from __future__ import annotations

from datetime import timedelta
from math import exp, pi, sin

from app.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    InsulinRecommendation,
    ModelInfo,
)
from app.services.model_adapter import OptionalLSTMAdapter


class ForecastEngine:
    def __init__(self, model_adapter: OptionalLSTMAdapter) -> None:
        self.model_adapter = model_adapter

    @staticmethod
    def _status(glucose: float) -> str:
        if glucose <= 70:
            return "hypo"
        if glucose >= 180:
            return "hyper"
        return "euglycemia"

    @staticmethod
    def _insulin_effect(units: float, elapsed_minutes: int, step_minutes: int) -> float:
        if units <= 0:
            return 0.0
        # Simplified rapid-acting insulin action curve.
        t_hours = elapsed_minutes / 60.0
        peak_hour = 1.3
        width = 0.9
        action = exp(-((t_hours - peak_hour) ** 2) / (2 * (width**2)))
        # 35 mg/dL per unit total effect distributed over the curve.
        return units * 35.0 * action * (step_minutes / 240.0)

    def _simulate_series(
        self,
        req: ForecastRequest,
        insulin_units: float,
    ) -> list[ForecastPoint]:
        steps = (req.horizon_hours * 60) // req.step_minutes
        current = float(req.current_glucose)
        points: list[ForecastPoint] = []
        history = [current] * max(self.model_adapter.seq_len, 8)

        for idx in range(1, steps + 1):
            elapsed = idx * req.step_minutes
            ts = req.timestamp + timedelta(minutes=elapsed)
            minute_of_day = ts.hour * 60 + ts.minute

            circadian = 2.0 * sin((2 * pi * minute_of_day) / 1440.0)
            homeostasis = (110.0 - current) * 0.03
            insulin_drop = self._insulin_effect(insulin_units, elapsed, req.step_minutes)
            lstm_delta = self.model_adapter.predict_delta(
                history=history,
                insulin_units=insulin_units,
                minute_of_day=minute_of_day,
            )

            next_cgm = current + homeostasis + (0.15 * circadian) + lstm_delta - insulin_drop
            next_cgm = float(min(400.0, max(40.0, next_cgm)))
            history.append(next_cgm)
            current = next_cgm

            points.append(
                ForecastPoint(
                    timestamp=ts,
                    cgm=round(next_cgm, 2),
                    status=self._status(next_cgm),
                    insulin_effect=round(insulin_drop, 3),
                )
            )

        return points

    @staticmethod
    def _score_series(points: list[ForecastPoint]) -> float:
        if not points:
            return float("inf")
        cgms = [p.cgm for p in points]
        n = len(cgms)
        target = 110.0
        mean_abs_error = sum(abs(v - target) for v in cgms) / n
        hypo = sum(1 for v in cgms if v < 70)
        severe_hypo = sum(1 for v in cgms if v < 54)
        hyper = sum(1 for v in cgms if v > 180)
        severe_hyper = sum(1 for v in cgms if v > 250)
        in_range = sum(1 for v in cgms if 70 <= v <= 180)
        tir_pct = (in_range / n) * 100.0

        score = (
            mean_abs_error
            + 1.8 * hypo
            + 3.0 * severe_hypo
            + 1.0 * hyper
            + 1.8 * severe_hyper
            - 0.25 * tir_pct
        )
        return score

    def _recommend_insulin(self, req: ForecastRequest) -> tuple[InsulinRecommendation, list[ForecastPoint]]:
        best_units = 0.0
        best_score = float("inf")
        best_points: list[ForecastPoint] = []

        # Search 0.0 - 10.0 U in 0.1 U increments.
        for step in range(0, 101):
            units = step / 10.0
            points = self._simulate_series(req=req, insulin_units=units)
            score = self._score_series(points)
            if score < best_score:
                best_score = score
                best_units = units
                best_points = points

        cgms = [p.cgm for p in best_points] if best_points else [req.current_glucose]
        in_range = sum(1 for v in cgms if 70 <= v <= 180)
        tir_pct = (in_range / max(1, len(cgms))) * 100.0

        recommendation = InsulinRecommendation(
            units=round(best_units, 1),
            expected_time_in_range_pct=round(tir_pct, 2),
            expected_min_cgm=round(min(cgms), 2),
            expected_max_cgm=round(max(cgms), 2),
            objective_score=round(best_score, 3),
            note="Recommendation from model simulation sweep (0.0-10.0U).",
        )
        return recommendation, best_points

    def forecast(self, req: ForecastRequest) -> ForecastResponse:
        with_insulin = None
        without_insulin = None
        recommended, optimal_points = self._recommend_insulin(req)

        if req.include_without_insulin:
            without_insulin = self._simulate_series(req=req, insulin_units=0.0)

        if req.insulin_units is not None and req.insulin_units > 0:
            with_insulin = self._simulate_series(req=req, insulin_units=req.insulin_units)
        else:
            with_insulin = optimal_points

        model_info = ModelInfo(
            model_active=self.model_adapter.model_active,
            model_name=self.model_adapter.model_name,
            notes=self.model_adapter.notes,
        )

        return ForecastResponse(
            request_id=req.request_id,
            input=req,
            with_insulin=with_insulin,
            without_insulin=without_insulin,
            recommended_insulin=recommended,
            model=model_info,
        )
