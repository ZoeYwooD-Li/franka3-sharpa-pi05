import json

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.shadow import load_observation


def save(tmp_path, **changes):
    data = {"state": np.zeros(58, np.float32),
            "tactile_f6": np.zeros((16, 10, 6), np.float32),
            "tactile_deform": np.zeros((10, 240, 240), np.uint8),
            "metadata_json": np.asarray(json.dumps({"source": "unit_test"})),
            **{camera: np.zeros((4, 4, 3), np.uint8) for camera in ("cam0", "cam1", "cam2")}}
    data.update(changes)
    path = tmp_path / "observation.npz"
    np.savez(path, **data)
    return path


def test_replay_loads_numeric_arrays_without_ros_or_sdk(tmp_path):
    obs, metadata = load_observation(save(tmp_path))
    assert obs["tactile_f6"].shape == (16, 10, 6)
    assert metadata["source"] == "unit_test"


@pytest.mark.parametrize("changes", [
    {"state": np.zeros(62)}, {"state": np.full(58, np.nan)},
    {"cam1": np.zeros((4, 4, 3), np.float32)},
    {"tactile_f6": np.zeros((1, 10, 6))},
    {"tactile_deform": np.full((10, 240, 240), 2.0)},
])
def test_invalid_recording_never_reaches_inference(tmp_path, changes):
    with pytest.raises(ValueError):
        load_observation(save(tmp_path, **changes))


def test_shadow_keeps_raw_prediction_when_ik_fails(tmp_path, monkeypatch, capsys):
    from deploy.fr3_sharpa_runtime.trex import shadow

    class Client:
        def __init__(self, *_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def hello(self, checkpoint): return {"checkpoint_kind": "full_finetune"}
        def slow_and_fast(self, *args): return 7, np.zeros((16, 62), np.float32)

    monkeypatch.setattr(shadow, "TrexClient", Client)
    monkeypatch.setattr(shadow, "validate_checkpoint", lambda path: {"checkpoint": str(path)})
    output = tmp_path / "result.npz"
    with pytest.raises(SystemExit) as error:
        shadow.main(["--checkpoint", "/test", "--observation", str(save(tmp_path)),
                     "--output", str(output), "--prompt", "test"])
    assert error.value.code == 2
    with np.load(output, allow_pickle=False) as result:
        assert result["raw_actions"].shape == (16, 62)
        assert "joint_actions" not in result
        report = json.loads(result["report_json"].item())
    assert report["published"] is False
    assert report["execute_ready"] is False
    assert report["ik"]["status"] == "FAIL"
    assert json.loads(capsys.readouterr().out)["chunk_id"] == 7
