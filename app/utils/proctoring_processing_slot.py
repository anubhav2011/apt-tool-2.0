"""
In-process single-flight guard for proctoring video processing.

Ensures at most one scheduler worker runs at a time within this process.
Not shared across multiple app workers or hosts; use a distributed lock if needed.
"""

from __future__ import annotations

import threading

_guard = threading.Lock()
_in_flight = False


def try_claim_processing_slot() -> bool:
    """If no job is in flight, mark the slot taken and return True; else False."""
    global _in_flight
    with _guard:
        if _in_flight:
            return False
        _in_flight = True
        return True


def release_processing_slot() -> None:
    """Clear the slot; safe to call once after a successful claim."""
    global _in_flight
    with _guard:
        _in_flight = False
