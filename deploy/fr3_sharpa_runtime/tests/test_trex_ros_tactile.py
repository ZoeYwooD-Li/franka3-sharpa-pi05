from types import SimpleNamespace as NS

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.ros_tactile import RosTactileBuffer, MAX_GAP_NS
from deploy.fr3_sharpa_runtime.trex.tactile import ROUTES

START = 10_000_000_000
STEP = 33_333_333


def message(route, kind, stamp, value=0, *, padding=0):
    header = NS(stamp=NS(sec=stamp // 10**9, nanosec=stamp % 10**9),
                frame_id=f"{route[0]}_{route[1]}_fingertip")
    if kind == "wrench":
        return NS(header=header, wrench=NS(force=NS(x=value, y=2, z=3), torque=NS(x=4, y=5, z=6)))
    return NS(header=header, encoding="mono8", width=240, height=240,
              step=240+padding, data=(bytes([value])*240 + b"\xff"*padding)*240)


def feed(buffer, index, *, routes=ROUTES, stamp=None):
    stamp = START + index*STEP if stamp is None else stamp
    for route in routes:
        for kind in ("deformation", "wrench"):
            buffer.remember(route, kind, message(route, kind, stamp, ROUTES.index(route)),
                            now_ros_ns=stamp+1_000, now_monotonic_ns=stamp+2_000)
    return stamp


def complete():
    buffer = RosTactileBuffer()
    for i in range(16):
        stamp = feed(buffer, i)
    return buffer, stamp


def snapshot(buffer, stamp):
    return buffer.snapshot(now_ros_ns=stamp+3_000, now_monotonic_ns=stamp+3_000)


def test_exact_pairing_order_and_physical_routes():
    buffer, stamp = complete()
    saved = snapshot(buffer, stamp)
    assert saved["tactile_f6"].shape == (16, 10, 6)
    np.testing.assert_array_equal(saved["tactile_f6"][-1, :, 0], np.arange(10))
    np.testing.assert_array_equal(saved["tactile_deform"][:, 0, 0], np.arange(10))
    assert np.all(np.diff(saved["tactile_stamp_ns"], axis=0) == STEP)


def test_missing_finger_or_unpaired_message_cannot_fill_history():
    buffer = RosTactileBuffer()
    for i in range(20):
        stamp = feed(buffer, i, routes=ROUTES[:-1])
        buffer.remember(ROUTES[-1], "wrench", message(ROUTES[-1], "wrench", stamp),
                        now_ros_ns=stamp, now_monotonic_ns=stamp)
    with pytest.raises(ValueError, match="0/16"):
        snapshot(buffer, stamp)


@pytest.mark.parametrize("fault", ["identity", "duplicate", "nan", "future", "size"])
def test_corrupt_data_invalidates_route(fault):
    buffer, stamp = complete()
    route = ROUTES[0]
    next_stamp = stamp + STEP
    kind = "deformation" if fault == "size" else "wrench"
    msg = message(route, kind, next_stamp)
    now = next_stamp
    if fault == "identity":
        msg.header.frame_id = "right_pinky_fingertip"
    elif fault == "duplicate":
        msg = message(route, kind, stamp)
    elif fault == "nan":
        msg.wrench.force.x = float("nan")
    elif fault == "future":
        now -= 1
    elif fault == "size":
        msg.data = b"bad"
    buffer.remember(route, kind, msg, now_ros_ns=now, now_monotonic_ns=now)
    assert not buffer.history[route]
    assert route in buffer.errors
    with pytest.raises(ValueError, match="0/16"):
        snapshot(buffer, next_stamp)


def test_gap_resets_instead_of_padding_or_repeating_samples():
    buffer, stamp = complete()
    stamp = feed(buffer, 20, stamp=stamp+MAX_GAP_NS+1)
    with pytest.raises(ValueError, match="1/16"):
        snapshot(buffer, stamp)


def test_pair_requires_equal_source_timestamp():
    buffer = RosTactileBuffer()
    route = ROUTES[0]
    for kind, stamp in (("wrench", START), ("deformation", START+1)):
        buffer.remember(route, kind, message(route, kind, stamp),
                        now_ros_ns=START+2, now_monotonic_ns=START+2)
    assert len(buffer.history[route]) == 0


def test_stale_snapshot_and_misaligned_fingers_rejected():
    buffer, stamp = complete()
    with pytest.raises(ValueError, match="stale"):
        snapshot(buffer, stamp+200_000_001)
    for index in range(4):
        feed(buffer, 16+index, routes=(ROUTES[0],))
    with pytest.raises(ValueError, match="not synchronized"):
        snapshot(buffer, stamp+4*STEP)


def test_padded_images_do_not_treat_padding_as_sensor_pixels():
    buffer = RosTactileBuffer()
    route = ROUTES[0]
    for kind in ("wrench", "deformation"):
        buffer.remember(route, kind, message(route, kind, START, 23, padding=8),
                        now_ros_ns=START, now_monotonic_ns=START)
    assert buffer.history[route][-1][3].shape == (240, 240)
    assert np.all(buffer.history[route][-1][3] == 23)
