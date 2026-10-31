"""Hardware-free T-Rex execution checks and immutable-base action timeline."""
from dataclasses import dataclass
import json
import time

import numpy as np

from ..protocol import LEFT_SHARPA_NAMES, RIGHT_SHARPA_NAMES

ARM = np.r_[0:7, 29:36]
HAND = np.r_[7:29, 36:58]
PERIOD = 1 / 30
JOINT_NAMES = (tuple(f'left_fr3_joint{i}' for i in range(1, 8)) + LEFT_SHARPA_NAMES
               + tuple(f'right_fr3_joint{i}' for i in range(1, 8)) + RIGHT_SHARPA_NAMES)


def joint_diagnostics(command, measured, target=None, indices=None):
    """Named errors, largest first; data only, with no hardware imports."""
    command, measured = finite_state(command), finite_state(measured)
    target = command if target is None else finite_state(target)
    indices = range(58) if indices is None else indices
    return [dict(index=int(i), joint=JOINT_NAMES[i], command_rad=float(command[i]),
                 measured_rad=float(measured[i]), target_rad=float(target[i]),
                 command_error_rad=float(abs(command[i]-measured[i])),
                 target_error_rad=float(abs(target[i]-measured[i])))
            for i in sorted(indices, key=lambda i: abs(target[i]-measured[i]), reverse=True)]


class ObservationUnavailable(ValueError):
    """No acceptable composite observation yet; no prediction was installed."""


def finite_state(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (58,) or not np.isfinite(result).all():
        raise ValueError('Expected finite 58-joint positions')
    return result.copy()


def validate_positions(values, lower, upper):
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2) or values.shape[-1] != 58 or not np.isfinite(values).all():
        raise ValueError('Expected finite joint target(s) with 58 columns')
    if np.any(values < lower) or np.any(values > upper):
        rows=values.reshape(-1,58);lo=np.asarray(lower);hi=np.asarray(upper)
        details=[]
        for row,joint in np.argwhere((rows<lo)|(rows>hi)):
            value=float(rows[row,joint])
            details.append({'row':int(row),'joint':JOINT_NAMES[joint],'index':int(joint),
                            'value_rad':value,'lower_rad':float(lo[joint]),'upper_rad':float(hi[joint]),
                            'excess_rad':float(max(lo[joint]-value,value-hi[joint]))})
        details.sort(key=lambda d:d['excess_rad'],reverse=True)
        raise ValueError('Prediction/command exceeds existing native joint limits; '+
                         json.dumps({'violation_count':len(details),'worst':details[:8]}))


def acquisition_check(command, measured):
    command, measured = finite_state(command), finite_state(measured)
    for label, indices, bound in (('Arm', ARM, .05), ('Hand', HAND, .15)):
        outside = indices[np.abs(command[indices]-measured[indices]) > bound]
        if len(outside):
            detail = '; '.join(f"{d['joint']}: command={d['command_rad']:.6f}, "
                               f"measured={d['measured_rad']:.6f}, error={d['command_error_rad']:.6f} rad"
                               for d in joint_diagnostics(command, measured, indices=outside))
            raise ValueError(f'{label} command exceeds {bound:g} rad live acquisition bound; {detail}')


def bounded_step(previous, target, speed=.03):
    if not np.isfinite(speed) or not 0 < speed <= .2:
        raise ValueError('Command speed must be finite and in (0, 0.2] rad/s')
    return finite_state(previous) + np.clip(finite_state(target)-previous, -speed*PERIOD, speed*PERIOD)


def prepare_step(previous, target):
    """Same straight joint-space 0.03rad/s step as pi05 SlowApproach."""
    previous=finite_state(previous)
    error=finite_state(target)-previous
    maximum=float(np.max(np.abs(error)))
    alpha=min(1.,.03*PERIOD/maximum) if maximum else 1.
    return previous+alpha*error


def frame_check(command, previous, measured, lower, upper, *, speed):
    validate_positions(command, lower, upper)
    acquisition_check(command, measured)
    delta = np.abs(command-previous)
    if np.max(delta) > speed*PERIOD+1e-8 or np.max(delta[HAND]) >= .1:
        raise ValueError('Emitted command slew exceeds execution bounds')


@dataclass(frozen=True)
class Prediction:
    kind: str
    chunk_id: int
    base_state: np.ndarray
    base_mono_ns: int
    sampled_mono_ns: int
    raw: np.ndarray
    joints: np.ndarray
    context_mono_ns: int | None = None
    timings: dict | None = None
    original_joints: np.ndarray | None = None


def installation_timing(prediction, now_ns):
    """Fresh tactile result plus a bounded, explicitly aged slow visual context.

    Live sensor acquisition still requires <=200ms source/receive age. Slow
    context may age during inference; it cannot install until a NEW fast call
    uses post-slow tactile. Keep both timestamps rather than retimestamping it.
    The 750ms context budget covers measured ~130ms camera age + ~300ms slow
    processing + ~100ms fast/IK/scheduling, with bounded scheduling margin.
    """
    if prediction.context_mono_ns is None:
        raise ValueError('Slow result has no original visual-context timestamp')
    context_age = now_ns-prediction.context_mono_ns
    tactile_age = now_ns-prediction.sampled_mono_ns
    if not 0 <= tactile_age <= context_age:
        raise ValueError('Invalid slow/fast acquisition timestamps')
    timing = {'context_age_ms': context_age/1e6, 'tactile_age_ms': tactile_age/1e6,
              'context_limit_ms': 750, 'tactile_limit_ms': 200,
              'worker': prediction.timings}
    return context_age <= 750_000_000 and tactile_age <= 200_000_000, timing


class ChunkTimeline:
    """16 rows, time stretched at 30 Hz; fast updates only change unsent rows."""
    def __init__(self, prediction, *, speed_scale=.1):
        if not np.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError('speed_scale must be finite and in (0,1]')
        if prediction.joints.shape != (16, 58):
            raise ValueError('T-Rex decoded horizon must be exactly 16x58')
        self.chunk_id = prediction.chunk_id
        self.base_state = finite_state(prediction.base_state)
        self.base_mono_ns = prediction.base_mono_ns
        self.joints = prediction.joints.copy()
        self.row = 0
        self.frames_in_row = 0
        self.frames_per_row = int(np.ceil(1/speed_scale))

    @property
    def done(self):
        return self.row >= 16

    def target(self):
        if self.done: raise StopIteration()
        return self.joints[self.row].copy()

    def emitted(self):
        if self.done: raise StopIteration()
        self.frames_in_row += 1
        if self.frames_in_row >= self.frames_per_row:
            self.row += 1
            self.frames_in_row = 0

    def refine(self, prediction, now_ns):
        if prediction.chunk_id != self.chunk_id:
            raise ValueError('Fast result belongs to another chunk')
        if not np.array_equal(prediction.base_state, self.base_state) or prediction.base_mono_ns != self.base_mono_ns:
            raise ValueError('Fast result changed the fixed chunk FK base')
        if not self.base_mono_ns <= prediction.sampled_mono_ns <= now_ns:
            raise ValueError('Fast tactile sample predates chunk or is future-dated')
        if now_ns-prediction.sampled_mono_ns > 200_000_000:
            return False  # Drop delayed inference, never replace a fresh live plan.
        if prediction.joints.shape != (16,58) or not np.isfinite(prediction.joints).all():
            raise ValueError('Invalid fast decoded actions')
        # A row already partly sent belongs to the emitted prefix too.
        start = self.row + int(self.frames_in_row > 0)
        if start >= 16: return False
        self.joints[start:] = prediction.joints[start:]
        return True


class Deadline:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.due = clock()

    def wait(self, sleep=time.sleep):
        remaining = self.due-self.clock()
        if remaining > 0: sleep(remaining)
        self.check()

    def check(self):
        if self.clock()-self.due > PERIOD:
            raise RuntimeError('30 Hz deadline missed; no catch-up burst permitted')

    def advance(self):
        self.due += PERIOD


class ExclusiveLease:
    """Protect the stateful slow/fast model cache from other clients."""
    def __init__(self, clock=time.monotonic):
        self.clock, self.token, self.expires = clock, None, 0

    def claim(self, token):
        if not isinstance(token, str) or not 16 <= len(token) <= 128:
            raise ValueError('Invalid inference session token')
        if self.token not in (None, token) and self.clock() < self.expires:
            raise ValueError('T-Rex server is owned by another execution session')
        self.token, self.expires = token, self.clock()+120

    def check(self, token):
        if self.token is not None and self.clock() >= self.expires:
            self.token = None
        if token is not None and self.token is None:
            raise ValueError('Inference session lease expired; reacquire before inference')
        if self.token is not None:
            if token != self.token:
                raise ValueError('Another session owns the T-Rex slow/fast cache')
            self.expires = self.clock()+120

    def release(self, token):
        self.check(token)
        self.token = None
