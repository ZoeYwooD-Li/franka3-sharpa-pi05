"""Sharpa execution speed presets, independent of model training method.

All profiles currently use serial chunks and the same 30 Hz SDK stream.
These are time-stretch presets, not migrated Wuji planners or qualified
hardware operating limits. Prepare speed remains a separate parameter.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeedProfile:
    name: str
    execute_speed_scale: float


PROFILES = {
    "slow": SpeedProfile("slow", 0.1),
    "medium": SpeedProfile("medium", 0.5),
    "fast": SpeedProfile("fast", 1.0),
}


def resolve_speed(name=None, scale=None):
    if name is not None and scale is not None:
        raise ValueError("Select --speed or --execute-speed-scale, not both")
    if scale is not None:
        return SpeedProfile("custom", scale)
    return PROFILES[name or "fast"]
