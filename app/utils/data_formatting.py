"""
Data formatting helpers: JSON API responses, numeric parsing, time strings (M:SS),
velocity labels.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.core.config import ProctoringConfig


# ---------------------------
# Interview ID
# ---------------------------
def normalize_interview_id(interview_id: object):
    """
    Convert external interview_id values to integers where possible so they
    align with the BIGINT schema in the database.
    """
    try:
        return int(interview_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return interview_id


# ---------------------------
# JSON
# ---------------------------
def format_json_response(data: Any) -> str:
    """Serialize payload to indented JSON; flattens ``timestamps`` arrays in the text."""
    if hasattr(data, "dict"):
        data = data.dict()
    elif hasattr(data, "model_dump"):
        data = data.model_dump()

    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    def replace_timestamps(match):
        array_content = match.group(1)
        numbers = re.findall(r"[\d.]+", array_content)
        if numbers:
            return '"timestamps": [' + ", ".join(numbers) + "]"
        return match.group(0)

    return re.sub(
        r'"timestamps":\s*\[(.*?)\]',
        replace_timestamps,
        json_str,
        flags=re.DOTALL,
    )


# ---------------------------
# Numeric Parsing
# ---------------------------
def parse_numeric_value(value) -> Optional[float]:
    """Extract a float from numbers or strings (first numeric substring)."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+\.?\d*", value)
        if match:
            try:
                return float(match.group())
            except (ValueError, AttributeError):
                return None
    return None

# --------------------------- 
# Velocity Helpers 
# ---------------------------
def float_to_velocity_label(velocity: float, config: ProctoringConfig) -> str:
    """Map numeric gaze/head velocity (deg/s) to a string label for DB storage."""
    if velocity < config.MIN_VELOCITY_THRESHOLD:
        return "negligible"
    if velocity < config.SUSPICIOUS_VELOCITY_THRESHOLD:
        return "slow"
    if velocity < config.HIGH_RISK_VELOCITY_THRESHOLD:
        return "moderate"
    return "rapid"

# --------------------------- 
# Time Formatting
# ---------------------------
def seconds_to_mmss(total_seconds: float) -> str:
    seconds_int = max(0, int(total_seconds))
    minutes = seconds_int // 60
    seconds = seconds_int % 60
    return f"{minutes}:{seconds:02d}"


def to_video_timestamp(value: object) -> str:
    """Normalize a loose occurrence timestamp to ``M:SS`` for logs or reports."""
    if value is None:
        return "0:00"
    if isinstance(value, (int, float)):
        return seconds_to_mmss(float(value))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return "0:00"
        parts = s.split(":")
        if len(parts) == 2:
            try:
                m = int(parts[0])
                sec = float(parts[1])
                return seconds_to_mmss(m * 60 + sec)
            except Exception:
                pass
        try:
            return seconds_to_mmss(float(s))
        except Exception:
            return "0:00"
    return "0:00"


def estimate_duration_from_gestures(gestures: list[dict[str, Any]]) -> float:
    """Estimate video duration from gesture timestamps + durations."""
    max_end = 0.0
    for gesture in gestures:
        for occ in gesture.get("occurrence") or []:
            start = parse_numeric_value(occ.get("timestamp")) or 0.0
            ts = occ.get("timestamp")
            if start == 0.0 and isinstance(ts, str) and ":" in ts:
                parts = ts.split(":")
                if len(parts) == 2:
                    try:
                        start = max(0.0, int(parts[0]) * 60 + float(parts[1]))
                    except (ValueError, TypeError):
                        start = 0.0
            max_end = max(max_end, start + float(occ.get("duration") or 0.0))
    return max_end
