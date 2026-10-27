"""Small, explicit LoRA implementation for the T-Rex MoT post-training path.

Only linear maps used by the action/tactile experts and task heads are adapted.
The visual tower, latent expert, deform encoder, and embedded VQ-VAE stay frozen.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


_EXPERT = re.compile(
    r"^model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)_(?:action|tactile)"
    r"|mlp_(?:action|tactile)\.(?:gate_proj|up_proj|down_proj))$"
)
_HEADS = (
    "x_embedder.", "t_embedder.", "tacf6_embedder.", "deform_proj.",
    "final_layer.", "final_layer_tactile.", "flare_proj.",
)


def is_target(name: str) -> bool:
    return bool(_EXPERT.fullmatch(name) or name.startswith(_HEADS))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA requires rank > 0 and 0 <= dropout < 1")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # T-Rex calls some expert methods through the unwrapped model, outside
        # Accelerator's autocast context. Its new task heads remain fp32 while
        # the activations from the Qwen visual/text tower are often bf16.
        base = self.base(x.to(self.base.weight.dtype))
        delta = F.linear(F.linear(self.dropout(x).float(), self.lora_a), self.lora_b)
        return (base.float() + delta * self.scale).to(x.dtype)


def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float,
                gradient_checkpointing: bool = True) -> dict:
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and is_target(name)]
    names = [name for name, _ in targets]
    for required in ("self_attn.q_proj_action", "self_attn.q_proj_tactile",
                     "mlp_action.gate_proj", "mlp_tactile.gate_proj",
                     "final_layer.", "final_layer_tactile."):
        if not any(required in name for name in names):
            raise RuntimeError(f"T-Rex LoRA target absent: {required}; inspect model layout")
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in targets:
        parent_name, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child,
                LoRALinear(module, rank, alpha, dropout))
    if gradient_checkpointing:
        model.model.gradient_checkpointing = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"rank": rank, "alpha": alpha, "dropout": dropout,
            "gradient_checkpointing": bool(gradient_checkpointing),
            "target_modules": names, "trainable_parameters": trainable}


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {name: param.detach().cpu().clone()
             for name, param in model.named_parameters()
             if name.endswith((".lora_a", ".lora_b"))}
    if not state:
        raise RuntimeError("No LoRA parameters to save")
    return state


def save_lora(model: nn.Module, directory: str | Path, config: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), directory / "lora.pt")
    (directory / "lora_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")


def merge_lora_weights(base_state: dict[str, torch.Tensor],
                       lora_state: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    rank = int(config["rank"])
    scale = float(config["alpha"]) / rank
    prefixes = {name.removesuffix(".lora_a") for name in lora_state
                if name.endswith(".lora_a")}
    if not prefixes:
        raise ValueError("Adapter has no LoRA matrices")
    if len(lora_state) != 2 * len(prefixes):
        raise ValueError("Incomplete LoRA matrix pairs")
    expected = set(config["target_modules"])
    if prefixes != expected:
        raise ValueError("Adapter targets differ from lora_config.json")
    for prefix in sorted(prefixes):
        key = prefix + ".weight"
        a = lora_state[prefix + ".lora_a"].float()
        b = lora_state[prefix + ".lora_b"].float()
        if key not in base_state or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"Missing or incompatible base weight: {key}")
        weight = base_state[key]
        if tuple(weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f"Shape mismatch for {key}: {weight.shape}")
        base_state[key] = (weight.float() + scale * (b @ a)).to(weight.dtype)
    return base_state
