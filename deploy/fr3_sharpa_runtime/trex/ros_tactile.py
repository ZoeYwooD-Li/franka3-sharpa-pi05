"""Read existing physical-finger ROS telemetry without opening another SDK.

ROS headers identify host acquisition time, not SDK frame numbers. Exact
wrench/image header pairing and advancing timestamps are required here; this
does not claim to detect losses before the existing publisher.
"""

from collections import deque

import numpy as np

from ..ros_observation import MAX_AGE_NS, MAX_SPREAD_NS, stamp_ns
from .tactile import ROUTES

MAX_GAP_NS = 75_000_000


class RosTactileBuffer:
    def __init__(self):
        self.pending = {route: {} for route in ROUTES}
        self.history = {route: deque(maxlen=16) for route in ROUTES}
        self.errors = {}

    def remember(self, route, kind, message, *, now_ros_ns, now_monotonic_ns):
        """Pair only messages from the same acquisition, never nearest-neighbor."""
        if route not in self.pending or kind not in ("wrench", "deformation"):
            raise ValueError("Unexpected tactile route")
        try:
            self._remember(route, kind, message, now_ros_ns, now_monotonic_ns)
        except (ValueError, TypeError, AttributeError) as error:
            self.pending[route].clear()
            self.history[route].clear()
            self.errors[route] = str(error)

    def _remember(self, route, kind, message, now_ros, now_mono):
        side, finger = route
        if message.header.frame_id != f"{side}_{finger}_fingertip":
            raise ValueError(f"Tactile identity mismatch: {side}/{finger}")
        stamp = stamp_ns(message)
        if stamp <= 0 or not 0 <= now_ros - stamp <= MAX_AGE_NS:
            raise ValueError(f"Stale/future tactile timestamp: {side}/{finger}")
        history = self.history[route]
        if history and stamp <= history[-1][0]:
            raise ValueError(f"Non-advancing tactile timestamp: {side}/{finger}")
        if kind == "wrench":
            w = message.wrench
            value = np.array([w.force.x, w.force.y, w.force.z,
                              w.torque.x, w.torque.y, w.torque.z], dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError("Non-finite tactile wrench")
        else:
            if (message.encoding != "mono8" or message.height != 240
                    or message.width != 240 or message.step < 240):
                raise ValueError("Expected 240x240 mono8 tactile deformation")
            raw = np.frombuffer(message.data, dtype=np.uint8)
            if raw.size != message.height * message.step:
                raise ValueError("Invalid tactile deformation byte count")
            value = raw.reshape(240, message.step)[:, :240].copy()
        pending = self.pending[route]
        # A bounded reorder buffer handles DDS delivery order without repeating data.
        for old in list(pending):
            if now_ros - old > MAX_AGE_NS:
                del pending[old]
        pair = pending.setdefault(stamp, {})
        if kind in pair:
            raise ValueError("Duplicate tactile message")
        pair[kind] = (value, now_mono)
        if len(pending) > 32:
            raise ValueError("Unpaired tactile telemetry overflow")
        if len(pair) != 2:
            return
        received = tuple(pair[k][1] for k in ("wrench", "deformation"))
        if history and (stamp - history[-1][0] > MAX_GAP_NS
                        or max(received) - max(history[-1][1]) > MAX_GAP_NS):
            history.clear()
        history.append((stamp, received, pair["wrench"][0], pair["deformation"][0]))
        for old in list(pending):
            if old <= stamp:
                del pending[old]
        self.errors.pop(route, None)

    def snapshot(self, *, now_ros_ns, now_monotonic_ns):
        for route in ROUTES:
            if len(self.history[route]) != 16:
                raise ValueError(f"Tactile {route} has {len(self.history[route])}/16 paired samples; "
                                 f"last error: {self.errors.get(route)}")
        rows = [list(self.history[route]) for route in ROUTES]
        stamps = np.array([[entry[0] for entry in row] for row in rows], dtype=np.int64).T
        if np.any(np.diff(stamps, axis=0) <= 0) or np.any(np.diff(stamps, axis=0) > MAX_GAP_NS):
            raise ValueError("Tactile history is not continuous")
        if np.any(np.ptp(stamps, axis=1) > MAX_SPREAD_NS):
            raise ValueError("Ten-finger tactile histories are not synchronized")
        latest_received = np.array([entry[-1][1] for entry in rows], dtype=np.int64)
        if (np.any(now_ros_ns - stamps[-1] < 0)
                or np.any(now_ros_ns - stamps[-1] > MAX_AGE_NS)
                or np.any(now_monotonic_ns - latest_received < 0)
                or np.any(now_monotonic_ns - latest_received > MAX_AGE_NS)):
            raise ValueError("Tactile snapshot is stale/future-dated")
        return {
            "tactile_f6": np.stack([np.stack([e[2] for e in row]) for row in rows], axis=1),
            "tactile_deform": np.stack([row[-1][3] for row in rows]),
            "tactile_stamp_ns": stamps,
            "tactile_received_ns": latest_received,
        }
