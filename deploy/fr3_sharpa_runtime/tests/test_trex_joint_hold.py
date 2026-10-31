import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.execution_core import Prediction,frame_check,acquisition_check
from deploy.fr3_sharpa_runtime.trex.joint_hold import PolicyJointHold


def test_hold_changes_only_two_policy_targets_and_preserves_original_data():
    measured=np.zeros(58);measured[[50,55]]=[-.035,-.04]
    raw=np.ones((16,62));joints=np.zeros((16,58));joints[:,[50,55]]=-.25
    prediction=Prediction('slow',1,measured.copy(),1,2,raw,joints)
    held=PolicyJointHold(measured).apply(prediction)
    assert np.all(held.joints[:,[50,55]]==[-.035,-.04])
    np.testing.assert_array_equal(held.original_joints,joints)
    np.testing.assert_array_equal(held.base_state,measured)
    assert held.raw is raw
    np.testing.assert_array_equal(prediction.joints,joints)
    other=np.setdiff1d(np.arange(58),[50,55])
    np.testing.assert_array_equal(held.joints[:,other],joints[:,other])
    assert measured[50]==-.035 and measured[55]==-.04


def test_hold_stays_fixed_across_fast_updates_and_does_not_bypass_feedback_guard():
    measured=np.zeros(58);hold=PolicyJointHold(measured)
    measured[50]=-.16
    prediction=Prediction('fast',1,measured.copy(),1,2,np.zeros((16,62)),np.ones((16,58)))
    held=hold.apply(prediction)
    assert np.all(held.joints[:,[50,55]]==0)
    with pytest.raises(ValueError,match='Hand command'):
        acquisition_check(np.zeros(58),measured)
    candidate=np.zeros(58);candidate[49]=.151
    with pytest.raises(ValueError,match='Hand command'):
        frame_check(candidate,np.zeros(58),np.zeros(58),np.full(58,-2),np.full(58,2),speed=.03)
