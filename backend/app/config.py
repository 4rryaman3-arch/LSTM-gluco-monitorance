from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(slots=True)
class Settings:
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    redis_url: str = os.getenv("REDIS_URL", "").strip()
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "120"))
    model_checkpoint: str = os.getenv(
        "MODEL_CHECKPOINT", "DiaData/model_artifacts/trajectory_lstm/model.pt"
    )
    model_seq_len: int = int(os.getenv("MODEL_SEQ_LEN", "24"))
    cors_origins: list[str] = field(
        default_factory=lambda: [
            v.strip()
            for v in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        ]
    )


settings = Settings()
