"""
Small helpers used by `ProctoringService` that don't need class state.
"""

from __future__ import annotations

from typing import Optional, Protocol

from app.utils.logger import debug_logger
from app.core.config import ProctoringConfig


class _StatusRepoProtocol(Protocol):
    def increment_proctoring_retry(self, interview_id: str) -> None: ...


def record_proctoring_failure(
    status_repository: _StatusRepoProtocol,
    interview_id: str,
) -> None:
    try:
        status_repository.increment_proctoring_retry(interview_id)
    except Exception as exc:
        debug_logger.error(
            f"Failed to increment proctoring_retry_count for "
            f"interview {interview_id}: {exc}",
            exc_info=True,
        )


class EventRiskClassifier:
    """Classify a single head/eye occurrence into ignore | normal | suspicious | high_risk."""

    def __init__(self, config: ProctoringConfig) -> None:
        self._config = config

    def classify(
        self,
        gesture_name: str,
        direction: Optional[str],
        duration_seconds: float,
    ) -> Optional[str]:
        """
        Secondary safety net for summary/event-log paths; ViolationTracker may
        already filter very short events.
        """
        if not direction:
            return None

        min_dur_head = float(self._config.MIN_EVENT_DURATION)
        min_dur_eye = float(self._config.EYE_MIN_EVENT_DURATION)
        d = direction.lower()

        if gesture_name == "head_movement":
            return self._classify_head(d, duration_seconds, min_dur_head)

        if gesture_name == "eye_movement":
            return self._classify_eye(d, duration_seconds, min_dur_eye)

        return None

    def _classify_head(self, d: str, dur: float, min_dur: float) -> Optional[str]:
        if d in ("left", "right", "down"):
            if dur < min_dur:
                return "ignore"
            if dur < 2.0:
                return "normal"
            if dur < 4.0:
                return "suspicious"
            return "high_risk"

        if d == "up":
            if dur < min_dur:
                return "ignore"
            if dur < 2.5:
                return "normal"
            if dur < 5.0:
                return "suspicious"
            return "high_risk"

        if d in ("up-left", "up-right", "down-left", "down-right"):
            if dur < min_dur:
                return "ignore"
            if dur < 2.0:
                return "normal"
            if dur < 4.0:
                return "suspicious"
            return "high_risk"

        return "normal"

    def _classify_eye(self, d: str, dur: float, min_dur: float) -> Optional[str]:
        if d in ("left", "right"):
            if dur < min_dur:
                return "ignore"
            if dur < 1.5:
                return "normal"
            if dur < 3.0:
                return "suspicious"
            return "high_risk"

        if d == "down":
            if dur < min_dur:
                return "ignore"
            if dur < 1.5:
                return "normal"
            if dur < 3.5:
                return "suspicious"
            return "high_risk"

        if d == "up":
            if dur < min_dur:
                return "ignore"
            if dur < 2.5:
                return "normal"
            if dur < 5.0:
                return "suspicious"
            return "high_risk"

        if d in ("up-left", "up-right", "down-left", "down-right"):
            if dur < min_dur:
                return "ignore"
            if dur < 1.5:
                return "normal"
            if dur < 3.0:
                return "suspicious"
            return "high_risk"

        return "normal"

