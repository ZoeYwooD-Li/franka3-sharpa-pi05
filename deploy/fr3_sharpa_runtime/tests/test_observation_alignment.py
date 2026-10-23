from collections import deque
from types import SimpleNamespace

import pytest

from deploy.fr3_sharpa_runtime.ros_observation import MAX_AGE_NS
from deploy.fr3_sharpa_runtime.ros_observation import MAX_SPREAD_NS
from deploy.fr3_sharpa_runtime.ros_observation import ObservationProvider
from deploy.fr3_sharpa_runtime.ros_observation import select_aligned_observation
from deploy.fr3_sharpa_runtime.ros_observation import stamp_ns

NOW = 10_000_000_000


def entry(age_ms, *, received_age_ms=None):
    stamp = NOW - int(age_ms * 1e6)
    message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9)))
    return message, NOW - int((age_ms if received_age_ms is None else received_age_ms) * 1e6)


def test_delayed_camera_uses_original_causal_history_without_relaxing_limits():
    latest = {"cam0": entry(162, received_age_ms=2), "cam1": entry(26), "left_arm": entry(3)}
    histories = {
        "cam0": deque([latest["cam0"]]),
        "cam1": deque([entry(180), latest["cam1"]]),
        "left_arm": deque([entry(165), latest["left_arm"]]),
    }
    selected, mode = select_aligned_observation(latest, histories, now_ros=NOW, now_mono=NOW)
    assert mode == "source_time_buffer"
    assert selected["cam1"][0] is histories["cam1"][0][0]
    assert stamp_ns(selected["cam0"][0]) == NOW - 162_000_000
    stamps = [stamp_ns(value[0]) for value in selected.values()]
    assert max(stamps) - min(stamps) == 18_000_000 <= MAX_SPREAD_NS
    assert all(0 <= NOW - stamp <= MAX_AGE_NS for stamp in stamps)
    assert all(stamp <= NOW - 162_000_000 for stamp in stamps)


@pytest.mark.parametrize("age", [250, -20])
def test_buffer_does_not_hide_stale_or_future_latest_stream(age):
    latest = {"cam0": entry(age), "cam1": entry(20)}
    histories = {"cam0": deque([entry(170), latest["cam0"]]), "cam1": deque([entry(175), latest["cam1"]])}
    with pytest.raises(ValueError, match="Stale/future"):
        select_aligned_observation(latest, histories, now_ros=NOW, now_mono=NOW)


def test_buffer_waits_when_no_genuine_matching_sample_exists():
    latest = {"cam0": entry(162), "cam1": entry(20)}
    with pytest.raises(ValueError, match="Waiting for time-aligned history"):
        select_aligned_observation(latest, {key: [value] for key, value in latest.items()}, now_ros=NOW, now_mono=NOW)


def test_camera_startup_requires_stable_duration_frequency_and_current_frames():
    provider = ObservationProvider.__new__(ObservationProvider)
    provider.startup_ready = False
    provider.startup_stable_s = 3.0
    provider.camera_windows = dict.fromkeys(
        ("cam0", "cam1", "cam2"), (NOW - 2900000000, NOW - 2900000000, NOW, NOW, 88)
    )
    assert not provider.cameras_ready(NOW)
    provider.camera_windows = dict.fromkeys(
        ("cam0", "cam1", "cam2"), (NOW - 3100000000, NOW - 3100000000, NOW, NOW, 94)
    )
    assert provider.cameras_ready(NOW)
