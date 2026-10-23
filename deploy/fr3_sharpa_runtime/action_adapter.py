"""Convert one Sharpa policy chunk into offline command records.

No ROS, SDK, controller, or hardware code belongs in this module.
"""

from __future__ import annotations

import numpy as np

from .protocol import (
    ACTION_HORIZON,
    ACTION_REPRESENTATION,
    DT_NS,
    FPS,
    FR3_COMMAND_NAMES,
    LEFT_FR3_SLICE,
    LEFT_SHARPA_NAMES,
    LEFT_SHARPA_SLICE,
    RIGHT_FR3_SLICE,
    RIGHT_SHARPA_NAMES,
    RIGHT_SHARPA_SLICE,
    SHARPA_JOINT_NAMES,
    UNIT,
    as_finite_action_chunk,
)
from .records import OfflineArmCommand, OfflineChunk, OfflineFrame, OfflineSharpaCommand


class SharpaActionAdapter:
    """Build a complete 30Hz, 50-frame offline command chunk."""

    def adapt_chunk(
        self,
        actions: np.ndarray,
        *,
        start_timestamp_ns: int = 0,
        sequence_start: int = 0,
        session_id: str = "offline",
        source: str = "openpi",
    ) -> OfflineChunk:
        if start_timestamp_ns < 0 or sequence_start < 0:
            raise ValueError("start_timestamp_ns and sequence_start must be non-negative")
        if not session_id or not source:
            raise ValueError("session_id and source must be non-empty")
        if sequence_start != 0:
            raise ValueError("The first offline chunk must start at sequence zero")
        chunk = as_finite_action_chunk(actions)
        frames = []
        for sequence in range(ACTION_HORIZON):
            timestamp_ns = start_timestamp_ns + sequence * DT_NS
            row = chunk[sequence]
            fr3 = OfflineArmCommand(
                timestamp_ns=timestamp_ns,
                sequence=sequence,
                source=source,
                session_id=session_id,
                active_sides=("left", "right"),
                joint_names=FR3_COMMAND_NAMES,
                positions=np.concatenate((row[LEFT_FR3_SLICE], row[RIGHT_FR3_SLICE])),
            )
            left = OfflineSharpaCommand(
                timestamp_ns=timestamp_ns,
                sequence=sequence,
                session_id=session_id,
                side="left",
                joint_names=LEFT_SHARPA_NAMES,
                position=row[LEFT_SHARPA_SLICE],
            )
            right = OfflineSharpaCommand(
                timestamp_ns=timestamp_ns,
                sequence=sequence,
                session_id=session_id,
                side="right",
                joint_names=RIGHT_SHARPA_NAMES,
                position=row[RIGHT_SHARPA_SLICE],
            )
            frames.append(OfflineFrame(timestamp_ns, sequence, fr3, left, right))
        return OfflineChunk(
            frames=tuple(frames),
            start_timestamp_ns=start_timestamp_ns,
            fps=FPS,
            causal=True,
            action_representation=ACTION_REPRESENTATION,
            unit=UNIT,
            session_id=session_id,
            source=source,
        )
