"""One model connection, one bounded job, background inference and IK."""
import queue
from collections import deque
import threading
import time

import numpy as np

from .actions import TrexActionDecoder
from .client import TrexClient
from .kinematics import IKError
from .execution_core import Prediction, validate_positions, ObservationUnavailable


def capture(snapshot):
    try:
        return snapshot() if callable(snapshot) else snapshot
    except ValueError as error:
        raise ObservationUnavailable(str(error)) from error


class InferenceWorker:
    def __init__(self, checkpoint, endpoint, lower, upper, *, timeout_ms=3000,
                 prediction_clip_rad=0.,held_hand_indices=()):
        if not np.isfinite(prediction_clip_rad) or not 0<=prediction_clip_rad<=.05:
            raise ValueError('Prediction clip allowance must be 0..0.05 rad')
        if any(i not in (50,55) for i in held_hand_indices):raise ValueError('Invalid held hand joint')
        self.prediction_clip_rad=prediction_clip_rad
        self.held_hand_indices=tuple(held_hand_indices)
        self.clipping_audit=deque(maxlen=256)
        self.clipped_values=0
        self.rejected_prediction=None
        self.ik_failure=None
        self.checkpoint, self.endpoint = checkpoint, endpoint
        self.lower, self.upper, self.timeout_ms = lower, upper, timeout_ms
        self.jobs, self.results = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.busy = False
        self.failure = None
        self.thread = threading.Thread(target=self.run, daemon=True, name='trex-inference-ik')
        self.thread.start()

    def submit(self, kind, snapshot, base, base_ns, chunk_id=None, *, refresh=None):
        if self.failure is not None: raise RuntimeError(f'Inference worker stopped: {self.failure}')
        if self.busy: return False
        if kind == 'slow' and refresh is None:
            raise ValueError('Slow planning requires a post-slow live tactile refresh')
        self.jobs.put_nowait((kind, snapshot, base.copy(), base_ns, chunk_id, refresh))
        self.busy = True
        return True

    def poll(self):
        if self.failure is not None:
            raise RuntimeError(f'Inference worker failed ({type(self.failure).__name__}): {self.failure}') from self.failure
        try: value = self.results.get_nowait()
        except queue.Empty: return None
        self.busy = False
        if isinstance(value, ObservationUnavailable): return value
        if isinstance(value, BaseException): raise RuntimeError(f'T-Rex inference failed: {value}') from value
        return value

    def run(self):
        try:
            with TrexClient(self.endpoint, timeout_ms=self.timeout_ms) as client:
                client.hello(self.checkpoint)
                client.claim()
                try:
                    decoder = TrexActionDecoder()
                    self.ready.set()
                    heartbeat=time.monotonic()+10
                    while not self.stop.is_set():
                        try: job = self.jobs.get(timeout=.1)
                        except queue.Empty:
                            if time.monotonic()>=heartbeat:
                                client.keepalive(); heartbeat=time.monotonic()+10
                            continue
                        kind, obs, base, base_ns, expected, refresh = job
                        started = time.monotonic_ns()
                        try:
                            obs = capture(obs)
                        except ObservationUnavailable as error:
                            self.results.put(error,timeout=1)
                            continue
                        captured = time.monotonic_ns()
                        context_ns = obs['sampled_mono_ns']
                        if kind == 'slow':
                            cid, raw = client.slow_and_fast(obs['cam0'],obs['cam1'],obs['cam2'],obs['prompt'],
                                                          obs['tactile_f6'],obs['tactile_deform'])
                            slow_done = time.monotonic_ns()
                            # Slow output is a cache seed, never directly installed.
                            # Re-read native tactile only AFTER slow has completed.
                            try:
                                obs = capture(refresh)
                            except ObservationUnavailable as error:
                                self.results.put(error,timeout=1)
                                continue
                            if obs['sampled_mono_ns'] <= context_ns:
                                raise ValueError('Post-slow tactile did not advance')
                            cid, raw = client.fast(obs['tactile_f6'],obs['tactile_deform'],cid)
                        else:
                            slow_done = captured
                            cid, raw = client.fast(obs['tactile_f6'],obs['tactile_deform'],expected)
                        inferred = time.monotonic_ns()
                        try:
                            original_joints=decoder.decode(raw, base)
                        except IKError as error:
                            self.ik_failure={
                                'schema_version':1,'kind':kind,'chunk_id':cid,'error':str(error),
                                'raw_actions':np.asarray(raw).tolist(),
                                'measured_state':base.tolist(),'base_mono_ns':base_ns,
                                'sampled_mono_ns':obs['sampled_mono_ns'],
                                'context_mono_ns':context_ns,'solver':error.diagnostics,
                            }
                            raise
                        joints=original_joints.copy()
                        if self.held_hand_indices:
                            # This staging result is never sent. The command
                            # owner applies its fixed post-Home hold anchor.
                            joints[:,self.held_hand_indices]=base[list(self.held_hand_indices)]
                        try:
                            if self.prediction_clip_rad:
                                from .prediction_limits import clip_hand_targets
                                joints,audit=clip_hand_targets(joints,self.lower,self.upper,self.prediction_clip_rad)
                                if audit['corrected_values']:
                                    self.clipped_values+=audit['corrected_values']
                                    self.clipping_audit.append({'kind':kind,'chunk_id':cid,**audit})
                            else:validate_positions(joints,self.lower,self.upper)
                        except ValueError as error:
                            self.rejected_prediction={'kind':kind,'chunk_id':cid,'error':str(error),
                                'raw':raw.copy(),'decoded':joints.copy(),'decoded_model':original_joints.copy(),
                                'base':base.copy(),'sampled_mono_ns':obs['sampled_mono_ns']}
                            raise
                        decoded = time.monotonic_ns()
                        timings = {'snapshot_ms': (captured-started)/1e6,
                                   'slow_ms': (slow_done-captured)/1e6,
                                   'fast_ms': (inferred-slow_done)/1e6,
                                   'ik_ms': (decoded-inferred)/1e6}
                        result = Prediction(kind,cid,base,base_ns,obs['sampled_mono_ns'],raw,joints,
                                            context_ns if kind == 'slow' else None,timings,
                                            original_joints if self.prediction_clip_rad or self.held_hand_indices else None)
                        self.results.put(result, timeout=1)
                finally:
                    client.release()
        except BaseException as error:
            self.failure = error
            try: self.results.put(error, timeout=.1)
            except queue.Full: pass

    def close(self):
        self.stop.set()
        self.thread.join(timeout=self.timeout_ms/1000+2)
