"""Keep one T-Rex chunk base fixed while applying fast tactile revisions."""

from __future__ import annotations

import time

import numpy as np

from .actions import TrexActionDecoder
from .contract import ACTION_HORIZON
from .kinematics import IKError


class TrexChunkSession:
    def __init__(self, chunk_id: int, measured_state, actions, *, decoder=None,
                 start_monotonic_ns=None, max_refinement_age_ns=200_000_000):
        if not isinstance(chunk_id, int) or chunk_id < 0:
            raise ValueError("T-Rex chunk_id must be a nonnegative integer")
        self.decoder = decoder or TrexActionDecoder()
        self.measured_state = np.asarray(measured_state, dtype=np.float64).copy()
        self.raw_actions = np.asarray(actions, dtype=np.float32).copy()
        self.joint_actions = self.decoder.decode(self.raw_actions, self.measured_state)
        self.chunk_id = chunk_id
        self.next_index = 0
        self.start_monotonic_ns = time.monotonic_ns() if start_monotonic_ns is None else start_monotonic_ns
        self.max_refinement_age_ns = max_refinement_age_ns

    def next_action(self) -> tuple[int, np.ndarray]:
        if self.next_index >= ACTION_HORIZON:
            raise StopIteration("T-Rex action chunk exhausted")
        index = self.next_index
        self.next_index += 1
        return index, self.joint_actions[index].copy()

    def refine(self, chunk_id: int, actions, *, sampled_monotonic_ns: int,
               now_monotonic_ns: int | None = None) -> int:
        """Replace unsent rows only; never rebase on a later measured pose."""
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        if chunk_id != self.chunk_id:
            raise ValueError("Fast refinement belongs to another T-Rex chunk")
        if (sampled_monotonic_ns < self.start_monotonic_ns
                or not 0 <= now - sampled_monotonic_ns <= self.max_refinement_age_ns):
            raise ValueError("Fast tactile observation is stale or predates the chunk")
        if self.next_index >= ACTION_HORIZON:
            raise ValueError("No unsent T-Rex actions remain")
        candidate = self.decoder.decode(actions, self.measured_state)
        if self.next_index:
            jump = np.max(np.abs(candidate[self.next_index, np.r_[:7, 29:36]]
                                 - self.joint_actions[self.next_index - 1, np.r_[:7, 29:36]]))
            if jump > self.decoder.max_joint_jump_rad:
                raise IKError("Fast refinement jumps from last emitted arm target")
        self.joint_actions[self.next_index:] = candidate[self.next_index:]
        self.raw_actions[self.next_index:] = np.asarray(actions, dtype=np.float32)[self.next_index:]
        return self.next_index
