import copy
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import ros_transport
from deploy.fr3_sharpa_runtime.dataset_sample import ORDER
from deploy.fr3_sharpa_runtime.dataset_sample import canonical_positions
from deploy.fr3_sharpa_runtime.setup_local import gateway_config


def test_gateway_override_preserves_constraints_and_source_config():
    source = {
        "teleop_safety_gateway": {
            "ros__parameters": {
                "allowed_sources": ["gello"],
                "command_timeout": 0.25,
                "max_initial_delta": 0.05,
                "max_joint_speed": 0.7,
            }
        }
    }
    original = copy.deepcopy(source)
    updated = gateway_config(source)
    assert source == original
    assert updated["teleop_safety_gateway"]["ros__parameters"] == {
        **original["teleop_safety_gateway"]["ros__parameters"],
        "allowed_sources": ["gello", "openpi"],
    }
    assert gateway_config(updated) == updated


def test_gateway_admission_rejects_unallowed_source(monkeypatch):
    node = SimpleNamespace(
        get_node_names=lambda: ["teleop_safety_gateway"],
        get_subscriptions_info_by_topic=lambda topic: [SimpleNamespace(node_name="teleop_safety_gateway")],
    )
    calls = []

    def require(node, remote, name, expected):
        calls.append((remote, name, expected))
        raise RuntimeError("not allowed")

    monkeypatch.setattr(ros_transport, "_require_parameter", require)
    with pytest.raises(RuntimeError, match="not allowed"):
        ros_transport.validate_gateway_admission(node)
    assert calls == [("teleop_safety_gateway", "allowed_sources", ["openpi"])]


def test_named_state_drops_velocity_and_reorders():
    names = [suffix for name in reversed(ORDER) for suffix in (name + ".velocity", name + ".position")]
    values = [(-1000 if name.endswith(".velocity") else ORDER.index(name.removesuffix(".position"))) for name in names]
    np.testing.assert_array_equal(canonical_positions(values, names), np.arange(58))
    with pytest.raises(ValueError, match="unique"):
        canonical_positions([0, 0], ["same", "same"])
    with pytest.raises(ValueError, match="missing"):
        canonical_positions([], [])


def test_lerobot_sample_uses_named_positions_and_physical_camera_roles(tmp_path):
    import io
    import json

    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq

    from deploy.fr3_sharpa_runtime.dataset_sample import load_sample

    names = [name + suffix for name in reversed(ORDER) for suffix in (".position", ".velocity")]
    values = [ORDER.index(name.removesuffix(".position")) if name.endswith(".position") else -999 for name in names]
    metadata = {"fps": 30, "features": {"observation.state": {"names": names}}}
    (tmp_path / "meta").mkdir()
    info = tmp_path / "meta/info.json"
    info.write_text(json.dumps(metadata))
    row = {"observation.state": values}
    for camera, value in (("cam0", 10), ("cam1", 20), ("cam2", 30)):
        buffer = io.BytesIO()
        Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
        row[f"observation.images.{camera}"] = {"bytes": buffer.getvalue()}
    path = tmp_path / "episode.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    sample = load_sample(path, "tomato")
    np.testing.assert_array_equal(sample["observation/state"], np.arange(58))
    assert sample["observation/left_wrist_image"][0, 0, 0] == 30
    assert sample["observation/right_wrist_image"][0, 0, 0] == 20
    metadata["fps"] = 20
    info.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="30 Hz"):
        load_sample(path, "tomato")
