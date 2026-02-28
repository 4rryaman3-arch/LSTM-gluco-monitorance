from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from lstm_polars import (
    TrainConfig,
    apply_standardizer,
    binary_metrics,
    build_sequences_for_groups,
    build_split_sequences,
    choose_features,
    find_best_threshold,
    fit_standardizer,
    load_polars_frame,
    split_patients,
    split_sequences_temporal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare classical baselines to LSTM-ready data splits.")
    parser.add_argument("--data-path", default="DiaData/datasets for T1D/maindatabase_sample.csv")
    parser.add_argument("--output", default="DiaData/model_artifacts/baseline_compare_sample.json")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--hypo-threshold", type=float, default=70.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-sequences-per-split", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-set", choices=["auto", "glucose", "glucose_hr"], default="auto")
    return parser.parse_args()


def prepare_data(args: argparse.Namespace) -> tuple[np.ndarray, ...]:
    df = load_polars_frame(data_path=args.data_path, seed=args.seed)
    feature_cols = choose_features(df, args.feature_set)
    patient_ids = df.select("PtID").unique().to_series().to_list()

    if len(patient_ids) >= 3:
        train_ids, val_ids, test_ids = split_patients(
            patient_ids, args.val_ratio, args.test_ratio, args.seed
        )
        x_train, y_train = build_split_sequences(
            df=df,
            patient_ids=train_ids,
            feature_cols=feature_cols,
            seq_len=args.seq_len,
            horizon_steps=args.horizon_steps,
            hypo_threshold=args.hypo_threshold,
            max_sequences=args.max_sequences_per_split,
            seed=args.seed,
        )
        x_val, y_val = build_split_sequences(
            df=df,
            patient_ids=val_ids,
            feature_cols=feature_cols,
            seq_len=args.seq_len,
            horizon_steps=args.horizon_steps,
            hypo_threshold=args.hypo_threshold,
            max_sequences=max(10000, args.max_sequences_per_split // 3),
            seed=args.seed + 1,
        )
        x_test, y_test = build_split_sequences(
            df=df,
            patient_ids=test_ids,
            feature_cols=feature_cols,
            seq_len=args.seq_len,
            horizon_steps=args.horizon_steps,
            hypo_threshold=args.hypo_threshold,
            max_sequences=max(10000, args.max_sequences_per_split // 3),
            seed=args.seed + 2,
        )
        split_mode = "patient"
    else:
        x_all, y_all = build_sequences_for_groups(
            groups=df.partition_by("PtID", maintain_order=True),
            feature_cols=feature_cols,
            seq_len=args.seq_len,
            horizon_steps=args.horizon_steps,
            hypo_threshold=args.hypo_threshold,
            max_sequences=args.max_sequences_per_split,
            seed=args.seed,
        )
        x_train, y_train, x_val, y_val, x_test, y_test = split_sequences_temporal(
            x_all, y_all, args.val_ratio, args.test_ratio
        )
        split_mode = "temporal"

    mean, std = fit_standardizer(x_train)
    x_train = apply_standardizer(x_train, mean, std)
    x_val = apply_standardizer(x_val, mean, std)
    x_test = apply_standardizer(x_test, mean, std)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    return x_train, y_train, x_val, y_val, x_test, y_test, split_mode


def proba_of(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        scores = np.asarray(scores, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = model.predict(x)
    return np.asarray(preds, dtype=np.float64)


def run_model(name: str, model, x_train, y_train, x_val, y_val, x_test, y_test) -> dict[str, object]:
    model.fit(x_train, y_train)
    val_prob = proba_of(model, x_val)
    threshold, val_metrics = find_best_threshold(y_val, val_prob)
    test_prob = proba_of(model, x_test)
    test_metrics = binary_metrics(y_test, test_prob, threshold=threshold)
    return {
        "model": name,
        "threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def main() -> None:
    args = parse_args()
    x_train, y_train, x_val, y_val, x_test, y_test, split_mode = prepare_data(args)

    models = [
        (
            "Dummy (most_frequent)",
            DummyClassifier(strategy="most_frequent"),
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=args.seed,
                class_weight="balanced_subsample",
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=args.seed),
        ),
        (
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                random_state=args.seed,
            ),
        ),
    ]

    results = []
    for name, model in models:
        results.append(run_model(name, model, x_train, y_train, x_val, y_val, x_test, y_test))

    payload = {
        "data_path": args.data_path,
        "split_mode": split_mode,
        "seq_len": args.seq_len,
        "horizon_steps": args.horizon_steps,
        "train_size": int(x_train.shape[0]),
        "val_size": int(x_val.shape[0]),
        "test_size": int(x_test.shape[0]),
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
