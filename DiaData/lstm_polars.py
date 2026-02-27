from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for LSTM training. Install it with: pip install torch"
    ) from exc


@dataclass
class TrainConfig:
    data_path: str
    output_dir: str = "DiaData/model_artifacts/lstm_polars"
    seq_len: int = 24
    horizon_steps: int = 12
    hypo_threshold: float = 70.0
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    epochs: int = 20
    tune_trials: int = 8
    patience: int = 4
    max_patients: int | None = None
    max_sequences_per_split: int | None = 300_000
    max_rows: int | None = None
    patient_sample_frac: float = 1.0
    feature_set: str = "auto"
    threshold: float = 0.5
    optimize_threshold: bool = True
    workers: int = 0
    device: str = "auto"


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, max(32, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_polars_frame(
    data_path: str, max_rows: int | None = None, patient_sample_frac: float = 1.0, seed: int = 42
) -> pl.DataFrame:
    lf = pl.scan_csv(
        data_path,
        infer_schema_length=2_000,
        ignore_errors=True,
        n_rows=max_rows,
    )
    cols = set(lf.collect_schema().names())
    required = {"ts", "PtID", "GlucoseCGM"}
    missing = required - cols
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    selected = ["ts", "PtID", "GlucoseCGM"]
    if "HR" in cols:
        selected.append("HR")
    if "Database" in cols:
        selected.append("Database")

    lf = lf.select(selected).with_columns(
        [
            pl.col("ts").str.to_datetime(strict=False).alias("ts"),
            pl.col("PtID").cast(pl.Utf8),
            pl.col("GlucoseCGM").cast(pl.Float32),
        ]
    )

    if not 0 < patient_sample_frac <= 1:
        raise ValueError("patient_sample_frac must be in (0, 1].")
    if patient_sample_frac < 1.0:
        mod = 10_000
        keep = max(1, int(mod * patient_sample_frac))
        lf = lf.filter((pl.col("PtID").hash(seed=seed) % mod) < keep)

    df = (
        lf.drop_nulls(subset=["ts", "PtID"])
        .sort(["PtID", "ts"])
        .collect(engine="streaming")
    )

    if "HR" in df.columns:
        df = df.with_columns(pl.col("HR").cast(pl.Float32))
    return df


def choose_features(df: pl.DataFrame, feature_set: str) -> list[str]:
    base = ["GlucoseCGM"]
    if feature_set == "glucose":
        return base
    if feature_set == "glucose_hr":
        if "HR" not in df.columns:
            raise ValueError("Requested glucose_hr, but HR column does not exist.")
        return ["GlucoseCGM", "HR"]
    if feature_set == "auto":
        return ["GlucoseCGM", "HR"] if "HR" in df.columns else base
    raise ValueError("feature_set must be one of: auto, glucose, glucose_hr")


def split_patients(
    patient_ids: list[str], val_ratio: float, test_ratio: float, seed: int
) -> tuple[set[str], set[str], set[str]]:
    if not 0 < val_ratio < 0.5:
        raise ValueError("val_ratio must be between 0 and 0.5")
    if not 0 < test_ratio < 0.5:
        raise ValueError("test_ratio must be between 0 and 0.5")
    if val_ratio + test_ratio >= 0.8:
        raise ValueError("val_ratio + test_ratio is too large")

    rng = np.random.default_rng(seed)
    ids = np.array(patient_ids)
    rng.shuffle(ids)

    n_total = len(ids)
    n_test = max(1, int(round(n_total * test_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_test - n_val
    if n_train < 1:
        raise ValueError("Not enough patients after split; reduce val/test ratios.")

    train_ids = set(ids[:n_train].tolist())
    val_ids = set(ids[n_train : n_train + n_val].tolist())
    test_ids = set(ids[n_train + n_val :].tolist())
    return train_ids, val_ids, test_ids


def split_sequences_temporal(
    x: np.ndarray, y: np.ndarray, val_ratio: float, test_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough sequences to create train/val/test split.")
    return (
        x[:n_train],
        y[:n_train],
        x[n_train : n_train + n_val],
        y[n_train : n_train + n_val],
        x[n_train + n_val :],
        y[n_train + n_val :],
    )


def clean_group(group: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for col in feature_cols:
        exprs.append(
            pl.col(col)
            .cast(pl.Float32)
            .interpolate()
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .alias(col)
        )
    group = group.with_columns(exprs).drop_nulls(subset=feature_cols)
    return group


def build_sequences_for_groups(
    groups: Iterable[pl.DataFrame],
    feature_cols: list[str],
    seq_len: int,
    horizon_steps: int,
    hypo_threshold: float,
    max_sequences: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    glucose_idx = feature_cols.index("GlucoseCGM")
    for group in groups:
        g = clean_group(group, feature_cols)
        n = g.height
        if n <= seq_len + horizon_steps:
            continue

        values = g.select(feature_cols).to_numpy().astype(np.float32)
        glucose = values[:, glucose_idx]

        x_local: list[np.ndarray] = []
        y_local: list[np.ndarray] = []
        stop = n - horizon_steps
        for end_idx in range(seq_len - 1, stop):
            start = end_idx - seq_len + 1
            future_slice = glucose[end_idx + 1 : end_idx + 1 + horizon_steps]
            label = 1.0 if np.nanmin(future_slice) <= hypo_threshold else 0.0
            x_local.append(values[start : end_idx + 1])
            y_local.append(np.float32(label))

        if x_local:
            x_parts.append(np.stack(x_local, axis=0))
            y_parts.append(np.array(y_local, dtype=np.float32))

    if not x_parts:
        raise RuntimeError("No sequences were produced. Reduce seq_len or horizon_steps.")

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    if max_sequences and x.shape[0] > max_sequences:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], size=max_sequences, replace=False)
        x = x[idx]
        y = y[idx]
    return x, y


def build_split_sequences(
    df: pl.DataFrame,
    patient_ids: set[str],
    feature_cols: list[str],
    seq_len: int,
    horizon_steps: int,
    hypo_threshold: float,
    max_sequences: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    filtered = df.filter(pl.col("PtID").is_in(patient_ids)).sort(["PtID", "ts"])
    groups = filtered.partition_by("PtID", maintain_order=True)
    return build_sequences_for_groups(
        groups=groups,
        feature_cols=feature_cols,
        seq_len=seq_len,
        horizon_steps=horizon_steps,
        hypo_threshold=hypo_threshold,
        max_sequences=max_sequences,
        seed=seed,
    )


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true_i = y_true.astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true_i == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true_i == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true_i == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true_i == 1)))

    total = max(1, y_true_i.size)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    specificity = tn / max(1, tn + fp)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, y_prob, threshold=0.5)
    best_acc = best_metrics["accuracy"]
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = binary_metrics(y_true, y_prob, threshold=float(threshold))
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += float(loss.item()) * xb.shape[0]
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(yb.detach().cpu().numpy())

    n_samples = max(1, len(loader.dataset))
    epoch_loss = total_loss / n_samples
    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0)
    return epoch_loss, y_true, y_prob


def train_one_config(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, float | int | bool],
    device: torch.device,
    epochs: int,
    patience: int,
    workers: int,
    threshold: float,
) -> tuple[LSTMClassifier, dict[str, float]]:
    train_ds = SequenceDataset(x_train, y_train)
    val_ds = SequenceDataset(x_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=workers,
    )

    model = LSTMClassifier(
        input_dim=x_train.shape[-1],
        hidden_dim=int(params["hidden_dim"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
        bidirectional=bool(params["bidirectional"]),
    ).to(device)

    pos = float(np.sum(y_train == 1.0))
    neg = float(np.sum(y_train == 0.0))
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_acc = -math.inf
    best_metrics: dict[str, float] = {}
    patience_left = patience

    for _ in range(epochs):
        run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, y_val_true, y_val_prob = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        val_metrics = binary_metrics(y_val_true, y_val_prob, threshold=threshold)
        val_metrics["val_loss"] = float(val_loss)
        scheduler.step(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_metrics = val_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, best_metrics


def sample_hparams(trial_idx: int, seed: int) -> dict[str, float | int | bool]:
    rng = np.random.default_rng(seed + trial_idx)
    return {
        "hidden_dim": int(rng.choice([64, 96, 128, 192])),
        "num_layers": int(rng.choice([1, 2, 3])),
        "dropout": float(rng.choice([0.1, 0.2, 0.3, 0.4])),
        "batch_size": int(rng.choice([128, 192, 256])),
        "lr": float(rng.choice([1e-3, 7e-4, 5e-4, 3e-4])),
        "weight_decay": float(rng.choice([1e-4, 5e-5, 1e-5])),
        "bidirectional": bool(rng.choice([True, False], p=[0.8, 0.2])),
    }


def tune_hyperparameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[LSTMClassifier, dict[str, float | int | bool], dict[str, float]]:
    best_model: LSTMClassifier | None = None
    best_params: dict[str, float | int | bool] = {}
    best_metrics: dict[str, float] = {}
    best_acc = -math.inf

    for trial_idx in range(cfg.tune_trials):
        params = sample_hparams(trial_idx, cfg.seed)
        model, metrics = train_one_config(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            params=params,
            device=device,
            epochs=cfg.epochs,
            patience=cfg.patience,
            workers=cfg.workers,
            threshold=cfg.threshold,
        )
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_model = model
            best_params = params
            best_metrics = metrics

    if best_model is None:
        raise RuntimeError("Hyperparameter tuning did not produce any model.")
    return best_model, best_params, best_metrics


def evaluate_model(
    model: LSTMClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    device: torch.device,
    threshold: float,
    workers: int,
) -> dict[str, float]:
    test_ds = SequenceDataset(x_test, y_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
    )
    criterion = nn.BCEWithLogitsLoss()
    _, y_true, y_prob = run_epoch(model, test_loader, criterion, device, optimizer=None)
    return binary_metrics(y_true, y_prob, threshold=threshold)


def predict_probabilities(
    model: LSTMClassifier,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    ds = SequenceDataset(x, y)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
    )
    criterion = nn.BCEWithLogitsLoss()
    _, y_true, y_prob = run_epoch(model, loader, criterion, device, optimizer=None)
    return y_true, y_prob


def save_artifacts(
    out_dir: Path,
    model: LSTMClassifier,
    cfg: TrainConfig,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    best_params: dict[str, float | int | bool],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    np.savez(out_dir / "normalization.npz", mean=mean, std=std)

    payload = {
        "config": asdict(cfg),
        "feature_cols": feature_cols,
        "best_params": best_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_training(cfg: TrainConfig) -> dict[str, object]:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    df = load_polars_frame(
        data_path=cfg.data_path,
        max_rows=cfg.max_rows,
        patient_sample_frac=cfg.patient_sample_frac,
        seed=cfg.seed,
    )
    feature_cols = choose_features(df, cfg.feature_set)

    patient_ids = df.select("PtID").unique().to_series().to_list()
    if cfg.max_patients and len(patient_ids) > cfg.max_patients:
        rng = np.random.default_rng(cfg.seed)
        patient_ids = rng.choice(
            np.array(patient_ids), size=cfg.max_patients, replace=False
        ).tolist()
        df = df.filter(pl.col("PtID").is_in(patient_ids))

    split_mode = "patient"
    if len(patient_ids) >= 3:
        train_ids, val_ids, test_ids = split_patients(
            patient_ids, cfg.val_ratio, cfg.test_ratio, cfg.seed
        )

        x_train, y_train = build_split_sequences(
            df=df,
            patient_ids=train_ids,
            feature_cols=feature_cols,
            seq_len=cfg.seq_len,
            horizon_steps=cfg.horizon_steps,
            hypo_threshold=cfg.hypo_threshold,
            max_sequences=cfg.max_sequences_per_split,
            seed=cfg.seed,
        )
        x_val, y_val = build_split_sequences(
            df=df,
            patient_ids=val_ids,
            feature_cols=feature_cols,
            seq_len=cfg.seq_len,
            horizon_steps=cfg.horizon_steps,
            hypo_threshold=cfg.hypo_threshold,
            max_sequences=max(10_000, (cfg.max_sequences_per_split or 100_000) // 3),
            seed=cfg.seed + 1,
        )
        x_test, y_test = build_split_sequences(
            df=df,
            patient_ids=test_ids,
            feature_cols=feature_cols,
            seq_len=cfg.seq_len,
            horizon_steps=cfg.horizon_steps,
            hypo_threshold=cfg.hypo_threshold,
            max_sequences=max(10_000, (cfg.max_sequences_per_split or 100_000) // 3),
            seed=cfg.seed + 2,
        )
    else:
        split_mode = "temporal"
        x_all, y_all = build_sequences_for_groups(
            groups=df.partition_by("PtID", maintain_order=True),
            feature_cols=feature_cols,
            seq_len=cfg.seq_len,
            horizon_steps=cfg.horizon_steps,
            hypo_threshold=cfg.hypo_threshold,
            max_sequences=cfg.max_sequences_per_split,
            seed=cfg.seed,
        )
        x_train, y_train, x_val, y_val, x_test, y_test = split_sequences_temporal(
            x_all, y_all, cfg.val_ratio, cfg.test_ratio
        )

    mean, std = fit_standardizer(x_train)
    x_train = apply_standardizer(x_train, mean, std)
    x_val = apply_standardizer(x_val, mean, std)
    x_test = apply_standardizer(x_test, mean, std)

    model, best_params, val_metrics = tune_hyperparameters(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        cfg=cfg,
        device=device,
    )

    eval_threshold = cfg.threshold
    if cfg.optimize_threshold:
        y_val_true, y_val_prob = predict_probabilities(
            model=model,
            x=x_val,
            y=y_val,
            batch_size=int(best_params["batch_size"]),
            device=device,
            workers=cfg.workers,
        )
        eval_threshold, val_metrics = find_best_threshold(y_val_true, y_val_prob)

    test_metrics = evaluate_model(
        model=model,
        x_test=x_test,
        y_test=y_test,
        batch_size=int(best_params["batch_size"]),
        device=device,
        threshold=eval_threshold,
        workers=cfg.workers,
    )

    out_dir = Path(cfg.output_dir)
    save_artifacts(
        out_dir=out_dir,
        model=model,
        cfg=cfg,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        best_params=best_params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    return {
        "output_dir": str(out_dir),
        "feature_cols": feature_cols,
        "split_mode": split_mode,
        "decision_threshold": float(eval_threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_params": best_params,
        "device": str(device),
        "train_size": int(x_train.shape[0]),
        "val_size": int(x_val.shape[0]),
        "test_size": int(x_test.shape[0]),
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="LSTM hypoglycemia prediction with Polars preprocessing."
    )
    parser.add_argument("--data-path", default="DiaData/datasets for T1D/maindatabase_sample.csv")
    parser.add_argument("--output-dir", default="DiaData/model_artifacts/lstm_polars")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--hypo-threshold", type=float, default=70.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tune-trials", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-sequences-per-split", type=int, default=300000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--patient-sample-frac", type=float, default=1.0)
    parser.add_argument("--feature-set", choices=["auto", "glucose", "glucose_hr"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--optimize-threshold", action="store_true")
    parser.add_argument("--no-optimize-threshold", dest="optimize_threshold", action="store_false")
    parser.set_defaults(optimize_threshold=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    results = run_training(config)
    print(json.dumps(results, indent=2))
