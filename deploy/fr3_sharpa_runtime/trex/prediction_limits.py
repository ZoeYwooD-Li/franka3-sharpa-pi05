"""Bounded hand-only range projection; arm targets are never corrected."""
import numpy as np

from .execution_core import HAND,JOINT_NAMES,validate_positions


def clip_hand_targets(joints, lower, upper, allowance):
    if not np.isfinite(allowance) or not 0<=allowance<=.05:
        raise ValueError('Prediction clip allowance must be 0..0.05 rad')
    original=np.asarray(joints)
    if original.shape!=(16,58) or not np.isfinite(original).all():
        raise ValueError('Expected finite decoded 16x58 prediction')
    result=original.copy()
    lo=np.asarray(lower)[HAND].astype(result.dtype)
    hi=np.asarray(upper)[HAND].astype(result.dtype)
    lo=np.where(lo.astype(float)<np.asarray(lower)[HAND],np.nextafter(lo,np.inf),lo)
    hi=np.where(hi.astype(float)>np.asarray(upper)[HAND],np.nextafter(hi,-np.inf),hi)
    clipped=np.clip(result[:,HAND],lo,hi)
    differences=np.abs(clipped.astype(float)-original[:,HAND])
    if np.any(differences>allowance):validate_positions(original,lower,upper)
    result[:,HAND]=clipped
    validate_positions(result,lower,upper)
    corrections=[]
    for row,column in np.argwhere(differences>0):
        joint=int(HAND[column])
        corrections.append({'row':int(row),'index':joint,'joint':JOINT_NAMES[joint],
                            'original_rad':float(original[row,joint]),'corrected_rad':float(result[row,joint]),
                            'correction_rad':float(differences[row,column])})
    return result,{'corrected_values':len(corrections),'corrections':corrections}
