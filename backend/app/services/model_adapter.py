from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class TrajectoryLSTM(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.15,
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class OptionalLSTMAdapter:
    def __init__(self, checkpoint_path: str, seq_len: int = 24) -> None:
        self.seq_len = seq_len
        self.model_active = False
        self.model_name = "heuristic"
        self.notes = "No trajectory checkpoint found; using heuristic fallback."
        self._model: TrajectoryLSTM | None = None

        if torch is None:
            self.notes = "PyTorch unavailable; using heuristic fallback."
            return

        path = Path(checkpoint_path)
        if not path.exists():
            return

        try:
            model = TrajectoryLSTM(input_dim=3, hidden_dim=64)
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self._model = model
            self.model_active = True
            self.model_name = "trajectory_lstm"
            self.notes = f"Loaded checkpoint from {path}"
        except Exception as exc:
            self.notes = f"Checkpoint load failed ({exc}); fallback enabled."

    def predict_delta(
        self,
        history: list[float],
        insulin_units: float,
        minute_of_day: int,
    ) -> float:
        if not self.model_active or self._model is None or torch is None:
            return 0.0

        seq = np.array(history[-self.seq_len :], dtype=np.float32)
        if seq.size < self.seq_len:
            seq = np.pad(seq, (self.seq_len - seq.size, 0), constant_values=seq[0])

        normalized_time = float(minute_of_day) / 1440.0
        features = np.column_stack(
            [
                seq,
                np.full(self.seq_len, insulin_units, dtype=np.float32),
                np.full(self.seq_len, normalized_time, dtype=np.float32),
            ]
        )
        x = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            delta = float(self._model(x).item())
        return float(np.clip(delta, -20.0, 20.0))
