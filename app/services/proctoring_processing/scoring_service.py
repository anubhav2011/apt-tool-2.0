# scoring_service.py
"""
Scoring Service - Risk scoring and report generation

"""
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base_proctoring_processing_service import BaseService


class ScoringService(BaseService):

    def _setup(self) -> None:
        pass

    _HEAD_GESTURE         = "head_movement"
    _EYE_GESTURE          = "eye_movement"
    _FACE_MISSING_GESTURE = "face_missing"

    @staticmethod
    def parse_timestamp_to_seconds(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return 0.0
            parts = s.split(":")
            if len(parts) == 2:
                try:
                    return max(0.0, int(parts[0]) * 60 + float(parts[1]))
                except (ValueError, TypeError):
                    pass
            try:
                return max(0.0, float(s))
            except (ValueError, TypeError):
                return 0.0
        return 0.0

    @staticmethod
    def _count_bursts_non_overlapping(
        event_times_sec: List[float],
        window_sec: float,
        min_events: int = 4,
    ) -> int:
        times = sorted(t for t in event_times_sec if t >= 0)
        n     = len(times)
        if n < min_events:
            return 0
        bursts = 0
        i      = 0
        while i <= n - min_events:
            end_time = times[i] + window_sec
            j = i
            while j < n and times[j] <= end_time:
                j += 1
            if j - i >= min_events:
                bursts += 1
                i = j
            else:
                i += 1
        return bursts

    @staticmethod
    def cheating_likelihood_level_from_stored_score(stored: float) -> str:
        s = max(0.0, min(10.0, round(float(stored), 2)))
        if s < 3.0:
            return "LESS"
        if s < 6.0:
            return "MODERATE"
        return "HIGH"

    @staticmethod
    def compute_cheating_likelihood(
            gestures: List[Dict[str, Any]],
            video_duration_sec: float,
            *,
            classify_event_risk: Callable[..., Optional[str]],
            weight_hrd: float,
            weight_erd: float,
            weight_rtr: float,
            weight_bf: float,
            burst_window_sec: float,
            burst_min_events: int = 4,
            high_risk_event_multiplier: float = 2.0,
    ) -> Tuple[float, str, Dict[str, float]]:
        eps = 1e-6
        dur = max(float(video_duration_sec or 0.0), eps)
        dur_min = dur / 60.0

        head_suspicious = 0
        head_high_risk = 0
        eye_suspicious = 0
        eye_high_risk = 0
        head_risk_time = 0.0
        eye_risk_time = 0.0
        face_absent_time = 0.0
        risky_event_times: List[float] = []

        for gesture in gestures:
            name = gesture.get("name") or ""
            occurrences = gesture.get("occurrence") or []
            if not occurrences:
                continue

            for occ in occurrences:
                duration = float(occ.get("duration") or 0.0)
                direction = occ.get("direction")
                ts = ScoringService.parse_timestamp_to_seconds(occ.get("timestamp"))

                if name == ScoringService._HEAD_GESTURE:
                    risk = classify_event_risk(
                        gesture_name=name,
                        direction=direction if isinstance(direction, str) else None,
                        duration_seconds=duration,
                    )
                    if risk in (None, "ignore", "normal"):
                        continue
                    if risk == "suspicious":
                        head_suspicious += 1
                        head_risk_time += duration
                        risky_event_times.append(ts)
                    elif risk == "high_risk":
                        head_high_risk += 1
                        head_risk_time += duration
                        risky_event_times.append(ts)

                elif name == ScoringService._EYE_GESTURE:
                    risk = classify_event_risk(
                        gesture_name=name,
                        direction=direction if isinstance(direction, str) else None,
                        duration_seconds=duration,
                    )
                    if risk in (None, "ignore", "normal"):
                        continue
                    if risk == "suspicious":
                        eye_suspicious += 1
                        eye_risk_time += duration
                        risky_event_times.append(ts)
                    elif risk == "high_risk":
                        eye_high_risk += 1
                        eye_risk_time += duration
                        risky_event_times.append(ts)

                elif name == ScoringService._FACE_MISSING_GESTURE:
                    face_absent_time += duration
                    risky_event_times.append(ts)

                elif name in ("multiple_faces", "face_occluded"):
                    risky_event_times.append(ts)

        hr_mult = float(high_risk_event_multiplier)
        hrd = min((head_suspicious + hr_mult * head_high_risk) / max(dur_min, eps), 6.0)
        erd = (eye_suspicious + hr_mult * eye_high_risk) / max(dur_min, eps)
       
        rtr = min(((head_risk_time + eye_risk_time) / 60) / dur_min, 1.0)

        bursts = ScoringService._count_bursts_non_overlapping(
            risky_event_times, burst_window_sec, min_events=burst_min_events,
        )
        bf = bursts / max(dur_min, eps)

        raw = (
                weight_hrd * hrd
                + weight_erd * erd
                + weight_rtr * rtr
                + weight_bf * bf
        )

        base_score = (raw / 2.5) * 10.0  # Normalize to 0-10 (2.5 = max expected raw score)

        safe_sec = 5  # ≤5 sec → no penalty
        max_sec = 60  # ≥60 sec → max penalty

        # Linear penalty based on total missing face duration
        if face_absent_time <= safe_sec:
            face_penalty = 0.0
        elif face_absent_time <= max_sec:
            face_penalty = (face_absent_time - safe_sec) / (max_sec - safe_sec) * 2.0
        else:
            face_penalty = 2.0

        score = base_score + face_penalty
        stored = max(0.0, min(10.0, round(float(score), 2)))
        level = ScoringService.cheating_likelihood_level_from_stored_score(stored)

        return stored, level, {
            "hrd": float(hrd),
            "erd": float(erd),
            "rtr": float(rtr),
            "bf": float(bf),
            "raw": float(raw),
            "face_penalty": float(face_penalty),
            "score_0_10": float(stored),
            "bursts": float(bursts),
            "head_suspicious": float(head_suspicious),
            "head_high_risk": float(head_high_risk),
            "eye_suspicious": float(eye_suspicious),
            "eye_high_risk": float(eye_high_risk),
            "face_absent_sec": float(face_absent_time),
            "high_risk_multiplier": float(hr_mult),
        }
