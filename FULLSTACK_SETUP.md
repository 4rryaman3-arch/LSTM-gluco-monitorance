# FastAPI + Next.js CGM Forecast Stack

Architecture:

`LSTM Model -> FastAPI -> Redis (optional) -> Next.js -> WebSocket`

## 1) Backend (FastAPI)

```powershell
cd d:\personal_projects\college_project
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

Optional Redis cache:

1. Set `REDIS_URL` in `backend/.env.example` (copy to your local env).
2. Restart FastAPI.

Run API + Redis via Docker:

```powershell
cd d:\personal_projects\college_project
docker compose -f docker-compose.redis.yml up --build
```

This starts:

- FastAPI on `http://localhost:8000`
- Redis on `localhost:6379` (FastAPI uses `redis://redis:6379/0` inside Docker)

Health check:

`GET http://localhost:8000/api/v1/health`

## 2) Frontend (Next.js)

```powershell
cd d:\personal_projects\college_project\frontend
corepack enable
corepack prepare pnpm@10.10.0 --activate
pnpm install
copy .env.local.example .env.local
pnpm dev
```

Open:

`http://localhost:3000`

Node runtime:

- Use Node `24.x` (see `frontend/.nvmrc`).

## 3) API Contract

### REST

`POST /api/v1/forecast`

Example body:

```json
{
  "history_points": [
    { "timestamp": "2026-02-27T20:00:00.000Z", "glucose": 168 },
    { "timestamp": "2026-02-27T20:05:00.000Z", "glucose": 162 },
    { "timestamp": "2026-02-27T20:10:00.000Z", "glucose": 157 },
    { "timestamp": "2026-02-27T20:15:00.000Z", "glucose": 151 },
    { "timestamp": "2026-02-27T20:20:00.000Z", "glucose": 147 },
    { "timestamp": "2026-02-27T20:25:00.000Z", "glucose": 143 }
  ],
  "horizon_hours": 6,
  "step_minutes": 5,
  "insulin_units": 2.0,
  "include_without_insulin": true
}
```

### WebSocket

`ws://localhost:8000/ws/forecast`

Send the same JSON payload. The server emits:

- `start`
- repeated `tick`
- `complete`

## Notes

- Response includes:
  - `recommended_insulin` (optimal units from model sweep)
  - `with_insulin_events` and `without_insulin_events`
- If no trajectory checkpoint exists at `MODEL_CHECKPOINT`, backend uses a heuristic fallback and marks it in `model.notes`.
- You can plug your trained trajectory checkpoint into `MODEL_CHECKPOINT` without changing frontend code.
