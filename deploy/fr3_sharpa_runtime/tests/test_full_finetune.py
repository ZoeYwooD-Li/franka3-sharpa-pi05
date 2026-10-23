import json

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import config
from deploy.fr3_sharpa_runtime.full_finetune import checkpoint_contract


@pytest.fixture
def checkpoint(monkeypatch, tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    (params / "manifest.ocdbt").write_bytes(b"test manifest")
    entries = {}
    for name, shape in (("action_in_proj", [58, 128]), ("action_out_proj", [58])):
        kind = "kernel" if name == "action_in_proj" else "bias"
        entries[name] = {"key_metadata": [{"key": name}, {"key": kind}],
                         "value_metadata": {"write_shape": shape}}
    (params / "_METADATA").write_text(json.dumps({"tree_metadata": entries}))
    stats = tmp_path / "assets" / "ab4_normal_absolute_30hz" / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(json.dumps({"norm_stats": {
        name: {"mean": [0.] * 58, "std": [1.] * 58,
               "q01": [-1.] * 58, "q99": [1.] * 58} for name in ("state", "actions")}}))
    monkeypatch.setattr(config, "CHECKPOINT", tmp_path)
    monkeypatch.setattr(config, "NORM_STATS", stats)
    return params, stats


def test_native_58d_full_checkpoint_contract_and_handshake(checkpoint):
    metadata = checkpoint_contract()
    assert metadata["action_dim"] == 58
    assert metadata["action_representation"] == "absolute_joint_position"
    assert metadata["model_finetune_mode"] == "full"
    assert len(metadata["normalization_sha256"]) == 64
    metadata["warmed_up"] = True
    config.validate_policy_metadata(metadata)
    metadata["model_finetune_mode"] = "lora"
    with pytest.raises(ValueError, match="model_finetune_mode"):
        config.validate_policy_metadata(metadata)


def test_full_loader_rejects_lora_and_wuji_64d_checkpoint(checkpoint):
    params, _ = checkpoint
    path = params / "_METADATA"
    data = json.loads(path.read_text())
    data["tree_metadata"]["lora_a"] = {}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="LoRA"):
        checkpoint_contract()
    del data["tree_metadata"]["lora_a"]
    data["tree_metadata"]["action_out_proj"]["value_metadata"]["write_shape"] = [64]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="native 58D"):
        checkpoint_contract()


def test_full_loader_rejects_wrong_normalization_width(checkpoint):
    _, stats = checkpoint
    value = json.loads(stats.read_text())
    value["norm_stats"]["state"]["mean"] = [0.] * 64
    stats.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="state.mean"):
        checkpoint_contract()


def test_training_contract_rejects_delta_or_lora():
    from types import SimpleNamespace
    training = SimpleNamespace(data=SimpleNamespace(action_hz=30, extra_delta_transform=True),
        model=SimpleNamespace(action_dim=58, action_horizon=50,
                              paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"))
    with pytest.raises(ValueError, match="no delta transform"):
        config.validate_training_contract(training)
    training.data.extra_delta_transform = False
    training.model.paligemma_variant = "gemma_2b_lora"
    with pytest.raises(ValueError, match="full-finetune"):
        config.validate_training_contract(training)


def test_full_inference_transform_preserves_canonical_absolute_targets(checkpoint):
    from deploy.fr3_sharpa_runtime.full_finetune import build_config
    training = build_config()
    config.validate_training_contract(training)
    data = training.data.create(training.assets_dirs, training.model)
    assert [type(t).__name__ for t in data.data_transforms.inputs] == ["Fr3SharpaInputs"]
    assert [type(t).__name__ for t in data.data_transforms.outputs] == ["Fr3SharpaOutputs"]
    state = np.arange(58, dtype=np.float32)
    observation = {"observation/state": state, "prompt": "tomato"}
    for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        observation[key] = np.zeros((8, 8, 3), dtype=np.uint8)
    transformed = data.data_transforms.inputs[0](observation)
    np.testing.assert_array_equal(transformed["state"], state)
    targets = np.full((50, 58), 0.25, dtype=np.float32)
    output = data.data_transforms.outputs[0]({"actions": targets, "state": state})
    np.testing.assert_array_equal(output["actions"], targets)
