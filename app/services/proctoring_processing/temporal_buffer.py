# temporal_buffer.py
"""
Rolling buffer for temporal landmark sequences.
Stores landmark vectors per frame for TVT input.

"""
import numpy as np
import threading
from collections import deque
from typing import Optional, List

from app.utils.logger import debug_logger


class TemporalBuffer:
    """
    Thread-safe rolling buffer of landmark vectors for a fixed window size.
    Each frame adds one vector of shape (landmark_dim,).
    Default: (936,) = [x1, y1, …, x468, y468].
    """

    def __init__(
        self,
        window_size:  int = 24,
        landmark_dim: int = 936,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if landmark_dim < 1:
            raise ValueError("landmark_dim must be >= 1")
        self.window_size  = window_size
        self.landmark_dim = landmark_dim
        self._buffer:     deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)
        # CHANGE 2: lock protects the two-step push() from race conditions
        # in the parallel pipeline where analyzer and aggregator threads
        # may call push() and get_window() concurrently.
        self._lock = threading.Lock()

    def push(
        self, vector: np.ndarray, timestamp: float = 0.0
    ) -> None:
        """
        Append one frame's landmark vector.
        Accepts shape (landmark_dim,) or (468, 2) — auto-flattened.

        CHANGE 1: wrong-size vectors are now silently logged and skipped
        instead of raising ValueError. Rationale: partial face detections
        (e.g., only 132 landmarks visible) can return undersized arrays;
        crashing the frame loop is worse than skipping one TVT frame.
        """
        v = np.asarray(vector, dtype=np.float32).flatten()
        if v.size != self.landmark_dim:
            # CHANGE 1: log and skip instead of raising
            debug_logger.debug(
                f"TemporalBuffer.push: expected dim={self.landmark_dim}, "
                f"got {v.size} — skipping frame"
            )
            return
        # CHANGE 2: both appends under the same lock to prevent
        # get_window() reading a buffer with one new vector but
        # an old timestamp list (or vice versa).
        with self._lock:
            self._buffer.append(v)
            self._timestamps.append(float(timestamp))

    def is_ready(self) -> bool:
        """True when the buffer holds a full window."""
        # CHANGE 2: lock for accurate length read in multi-thread context
        with self._lock:
            return len(self._buffer) >= self.window_size

    def get_window(self) -> Optional[np.ndarray]:
        """
        Return (T, D) array with T=window_size, D=landmark_dim.
        Returns None if not ready.

        CHANGE 3: acquires lock before snapshotting to ensure buffer and
        timestamps are consistent (no partial push() in flight).
        """
        with self._lock:
            if len(self._buffer) < self.window_size:
                return None
            # list() creates a point-in-time snapshot — safe to release
            # lock after this line since we work on the copy
            return np.stack(list(self._buffer), axis=0)

    def get_timestamps(self) -> List[float]:
        """Return timestamps for the current window (oldest first)."""
        with self._lock:
            return list(self._timestamps)

    def clear(self) -> None:
        """Reset the buffer."""
        with self._lock:
            self._buffer.clear()
            self._timestamps.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def __repr__(self) -> str:
        with self._lock:
            filled = len(self._buffer)
        return (
            f"TemporalBuffer("
            f"window={self.window_size}, "
            f"dim={self.landmark_dim}, "
            f"filled={filled})"
        )