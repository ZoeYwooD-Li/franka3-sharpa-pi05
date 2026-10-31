"""Explicit reduced hand control; never changes measured state or model inputs."""
from collections import deque
from dataclasses import replace

import numpy as np

from .execution_core import finite_state, JOINT_NAMES


class PolicyJointHold:
    indices = np.array([50,55])

    def __init__(self, measured):
        self.values=finite_state(measured)[self.indices].copy()
        self.audit=deque(maxlen=256)
        self.predictions_modified=0

    def metadata(self):
        return {'mode':'hold_at_policy_start_measured','joints':[JOINT_NAMES[i] for i in self.indices],
                'indices':self.indices.tolist(),'held_rad':self.values.tolist(),
                'feedback':'original_unmodified','audit_capacity':256}

    def apply(self, prediction):
        original=(prediction.original_joints if prediction.original_joints is not None else prediction.joints).copy()
        joints=prediction.joints.copy()
        joints[:,self.indices]=self.values
        self.predictions_modified+=1
        self.audit.append({'kind':prediction.kind,'chunk_id':prediction.chunk_id,
                           'sampled_mono_ns':prediction.sampled_mono_ns,
                           'original_targets_rad':original[:,self.indices].tolist()})
        return replace(prediction,joints=joints,original_joints=original)
