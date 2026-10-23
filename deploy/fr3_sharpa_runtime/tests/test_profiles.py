import json
import sys

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import run
from deploy.fr3_sharpa_runtime.profiles import PROFILES
from deploy.fr3_sharpa_runtime.sdk_direct import scaled_action_frames


@pytest.mark.parametrize("name,scale,frames", [("slow", 0.1, 491), ("medium", 0.5, 99), ("fast", 1.0, 50)])
def test_profiles_keep_full_horizon_and_share_the_execution_stream(name, scale, frames):
    actions = np.linspace(0, 0.05, 50, dtype=np.float32)[:, None] * np.ones((1, 58), dtype=np.float32)
    profile = PROFILES[name]
    assert profile.execute_speed_scale == scale
    emitted = np.asarray(list(scaled_action_frames(actions, profile.execute_speed_scale)))
    assert len(emitted) == frames
    np.testing.assert_allclose(emitted[-1], actions[-1])


@pytest.mark.parametrize("name", ["slow", "medium", "fast"])
def test_offline_check_reports_effective_profile_without_devices(monkeypatch, capsys, name):
    monkeypatch.setattr(run, "execute_policy_prepare", lambda *args: pytest.fail("hardware must not start"))
    monkeypatch.setattr(run, "SharpaRuntime", lambda *args: pytest.fail("policy must not load"))
    monkeypatch.setattr(sys, "argv", ["run", "--check", "--speed", name])
    run.main()
    report = json.loads(capsys.readouterr().out)
    assert report["hardware_output"] is False
    assert report["speed"] == name
    assert report["execute_speed_scale"] == PROFILES[name].execute_speed_scale
    assert report["action_hz"] == 30


def test_profile_and_custom_scale_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run", "--check", "--speed", "medium", "--execute-speed-scale", "0.1"])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
