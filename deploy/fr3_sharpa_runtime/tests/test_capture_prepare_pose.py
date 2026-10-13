import json
from pathlib import Path
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.capture_prepare_pose import make_config, save_config
from deploy.fr3_sharpa_runtime.feedback import LiveFeedback
from deploy.fr3_sharpa_runtime.prepare import PrepareConfig, GROUP_NAMES


def captured_document():
    template = json.loads((Path(__file__).resolve().parents[1]/"prepare_config.example.json").read_text())
    state = np.arange(58, dtype=np.float32)/100
    feedback = LiveFeedback(state, (1_000_000_000,)*4, (2_000_000_000,)*4)
    return state, make_config(feedback, template, pose_name="test_only",
                             ros_now_ns=1_010_000_000, monotonic_now_ns=2_010_000_000)


def test_capture_keeps_58d_order_timestamps_sources_and_verify_only(tmp_path):
    state, document = captured_document()
    output = tmp_path/"pose.json"
    save_config(document, output)
    config = PrepareConfig.load(output)
    np.testing.assert_array_equal(config.target, state)
    assert config.backend == "verify_only"
    for group, names in GROUP_NAMES.items():
        assert document["groups"][group]["joint_names"] == list(names)
        assert document["capture"]["sources"][group]["header_stamp_ns"] == 1_000_000_000
    assert len(document["capture"]["sources"]) == 4


def test_capture_does_not_overwrite_existing_pose(tmp_path):
    _, document = captured_document()
    output = tmp_path/"pose.json"
    output.write_text("existing")
    with pytest.raises(FileExistsError): save_config(document, output)
    assert output.read_text() == "existing"
    save_config(document, output, overwrite=True)
    assert PrepareConfig.load(output).target.shape == (58,)


def test_capture_rejects_stale_feedback():
    feedback = LiveFeedback(np.zeros(58,dtype=np.float32), (1_000_000_000,)*4, (2_000_000_000,)*4)
    with pytest.raises(ValueError, match="Stale"):
        make_config(feedback, {}, pose_name="test", ros_now_ns=2_000_000_000, monotonic_now_ns=3_000_000_000)
