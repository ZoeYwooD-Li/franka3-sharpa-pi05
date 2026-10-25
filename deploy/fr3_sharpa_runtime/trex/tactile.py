"""Collect real SDK fingertip frames in the T-Rex physical-finger order."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ROUTES = tuple((side, finger) for side in ("left", "right") for finger in FINGERS)


def _identity(value) -> str:
    return str(getattr(value, "value", value))


@dataclass(frozen=True)
class TactileSample:
    f6: np.ndarray
    deformation: np.ndarray
    received_monotonic_ns: int


class TactileHistory:
    """Require 16 genuine sequential samples; never fill missing data with zero."""

    def __init__(self, *, max_gap_ns=75_000_000, max_finger_spread_s=0.1):
        self.samples = deque(maxlen=16)
        self.last_indices = {}
        self.max_gap_ns = int(max_gap_ns)
        self.max_finger_spread_s = float(max_finger_spread_s)

    def append(self, frames, *, received_monotonic_ns=None) -> TactileSample:
        try:
            return self._append(frames, received_monotonic_ns=received_monotonic_ns)
        except Exception:
            self.samples.clear()
            self.last_indices.clear()
            raise

    def _append(self, frames, *, received_monotonic_ns=None) -> TactileSample:
        now = time.monotonic_ns() if received_monotonic_ns is None else int(received_monotonic_ns)
        if set(frames) != set(ROUTES):
            raise ValueError("T-Rex requires exactly ten physical fingertip frames")
        f6 = np.empty((10, 6), dtype=np.float32)
        deformation = np.empty((10, 240, 240), dtype=np.uint8)
        timestamps = []
        next_indices = {}
        for index, (side, finger) in enumerate(ROUTES):
            frame = frames[(side, finger)]
            if frame is None or _identity(frame.side) != side or _identity(frame.finger) != finger:
                raise ValueError(f"Tactile frame identity mismatch: {side}/{finger}")
            frame_index = int(frame.frame_index)
            if frame_index < 0 or frame_index <= self.last_indices.get((side, finger), -1):
                raise ValueError(f"Non-advancing tactile frame: {side}/{finger}")
            next_indices[(side, finger)] = frame_index
            stamp = float(frame.timestamp_s)
            if not np.isfinite(stamp):
                raise ValueError(f"Non-finite tactile timestamp: {side}/{finger}")
            timestamps.append(stamp)
            wrench = np.asarray(frame.wrench, dtype=np.float32)
            if wrench.shape != (6,) or not np.isfinite(wrench).all():
                raise ValueError(f"Invalid F6 wrench: {side}/{finger}")
            image = frame.deformation
            if (image is None or image.width != 240 or image.height != 240
                    or image.encoding != "mono8"):
                raise ValueError(f"Missing 240x240 mono8 deformation: {side}/{finger}")
            pixels = np.frombuffer(image.data, dtype=np.uint8)
            if pixels.size != 240 * 240:
                raise ValueError(f"Invalid deformation byte count: {side}/{finger}")
            f6[index] = wrench
            deformation[index] = pixels.reshape(240, 240)
        if max(timestamps) - min(timestamps) > self.max_finger_spread_s:
            raise ValueError("Ten fingertip timestamps are not synchronized")
        if self.samples and (now <= self.samples[-1].received_monotonic_ns
                             or now - self.samples[-1].received_monotonic_ns > self.max_gap_ns):
            raise ValueError("Tactile sample gap; 16-frame history reset")
        sample = TactileSample(f6, deformation, now)
        self.samples.append(sample)
        self.last_indices = next_indices
        return sample

    def read_from_hands(self, left_hand, right_hand, *, received_monotonic_ns=None) -> TactileSample:
        """Call the physical-finger SDK API, which owns left/right channel routing."""
        try:
            frames = {}
            for side, hand in (("left", left_hand), ("right", right_hand)):
                if not hand.tactile_available:
                    raise RuntimeError(f"{side} Sharpa tactile sensor is unavailable")
                for finger in FINGERS:
                    frames[(side, finger)] = hand.read_tactile(finger, timeout_s=0.0)
            return self.append(frames, received_monotonic_ns=received_monotonic_ns)
        except Exception:
            self.samples.clear()
            self.last_indices.clear()
            raise

    def snapshot(self, *, now_monotonic_ns=None, max_age_ns=200_000_000) -> tuple[np.ndarray, np.ndarray, int]:
        if len(self.samples) != 16:
            raise ValueError(f"T-Rex F6 history has {len(self.samples)}/16 real samples")
        latest = self.samples[-1]
        now = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        if not 0 <= now - latest.received_monotonic_ns <= max_age_ns:
            raise ValueError("T-Rex tactile history is stale or future-dated")
        return (np.stack([sample.f6 for sample in self.samples]),
                latest.deformation.copy(), latest.received_monotonic_ns)
