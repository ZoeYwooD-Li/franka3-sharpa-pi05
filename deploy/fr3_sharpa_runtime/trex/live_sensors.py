"""Shared ROS/native buffers; one dedicated acquisition thread reads the SDK."""
from collections import deque
import base64
import threading
import time

import numpy as np

from ..ros_observation import ordered_state, stamp_ns, rgb_image
from .execution_core import finite_state
from .tactile import ROUTES
from .native_tactile import NativeTactileBuffer


def before(entries, anchor):
    candidates = [e for e in entries if e[0] <= anchor]
    if not candidates: raise ValueError('Waiting for original time-aligned sensor history')
    return max(candidates, key=lambda e:e[0])


class SensorStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.arms = {s:deque(maxlen=512) for s in ('left','right')}
        self.hands = {s:deque(maxlen=256) for s in ('left','right')}
        self.cameras = {c:deque(maxlen=160) for c in ('cam0','cam1','cam2')}
        self.native = NativeTactileBuffer()
        self.markers = {}
        self.error = None

    def arm(self, side, message):
        try:
            stamp, q = stamp_ns(message), ordered_state(message,side,False)
            with self.lock:
                if stamp <= 0 or (self.arms[side] and stamp <= self.arms[side][-1][0]):
                    raise ValueError('Non-advancing arm state timestamp')
                self.arms[side].append((stamp,np.asarray(q),time.monotonic_ns()))
        except (ValueError,TypeError,AttributeError) as error:
            with self.lock: self.arms[side].clear(); self.error = str(error)

    def camera(self, key, message):
        with self.lock:
            stamp = stamp_ns(message)
            if stamp <= 0 or (self.cameras[key] and stamp <= self.cameras[key][-1][0]):
                self.cameras[key].clear()
            self.cameras[key].append((stamp,message,time.monotonic_ns()))

    def native_hands(self, executor):
        """One sampler owns native reads; preserve each finger independently."""
        now, wall = time.monotonic_ns(), time.time_ns()
        with self.lock:
            for side, hand in (('left',executor.left),('right',executor.right)):
                state = hand._wave.get_states()
                marker = (int(state.timestamp),int(state.sequence))
                q = np.asarray(state.angles,dtype=float)
                if marker == (0,0) or q.shape != (22,) or not np.isfinite(q).all():
                    continue
                if marker != self.markers.get(side):
                    self.markers[side] = marker
                    self.hands[side].append((wall,q.copy(),now))
            for side,finger in ROUTES:
                frame = (executor.left if side=='left' else executor.right).read_tactile(finger,timeout_s=0.)
                self.native.remember((side,finger),frame,time.time_ns(),time.monotonic_ns())

    def tactile_diagnostics(self):
        with self.lock:
            return self.native.diagnostics()

    @staticmethod
    def timing(entries, now_ros, now_mono, *, check_span=True, labels=None):
        stamps,received = [e[0] for e in entries],[e[2] for e in entries]
        details={'streams':labels,'source_age_ms':[(now_ros-s)/1e6 for s in stamps],
                 'receive_age_ms':[(now_mono-r)/1e6 for r in received]}
        if any(not 0 < s <= now_ros or now_ros-s > 200_000_000 for s in stamps):
            raise ValueError('Stale/future sensor source timestamp: '+str(details))
        if any(not 0 <= now_mono-r <= 200_000_000 for r in received):
            raise ValueError('Stale/future received sensor sample')
        if check_span and max(stamps)-min(stamps) > 100_000_000:
            raise ValueError('Sensor acquisition span exceeds 100 ms')

    def feedback(self):
        with self.lock:
            if any(not h for h in [*self.arms.values(),*self.hands.values()]):
                raise ValueError(f'Waiting for native hand/ROS arm feedback: {self.error}')
            rows = [self.arms['left'][-1],self.hands['left'][-1],self.arms['right'][-1],self.hands['right'][-1]]
            self.timing(rows,time.time_ns(),time.monotonic_ns(),labels=['left_arm','left_hand','right_arm','right_hand'])
            return finite_state(np.concatenate([e[1] for e in rows]))

    def camera_ready(self):
        """Validate source clocks and cadence without opening hand hardware."""
        with self.lock:
            rows=[]
            for key,buf in self.cameras.items():
                if not buf: raise ValueError(f'Missing camera {key}')
                rows.append(buf[-1])
            self.timing(rows,time.time_ns(),time.monotonic_ns(),check_span=False,labels=list(self.cameras))
            for key,buf in self.cameras.items():
                if len(buf)<3 or buf[-1][2]-buf[0][2]<2_000_000_000:
                    span=(buf[-1][2]-buf[0][2])/1e9
                    raise ValueError(f'Waiting for camera startup stability: {key}, frames={len(buf)}, receive_span_s={span:.3f}/2.000')
                stamps=np.array([e[0] for e in buf],dtype=np.int64)
                intervals=np.diff(stamps)
                hz=(len(stamps)-1)*1e9/(stamps[-1]-stamps[0])
                if np.any(intervals<=0) or max(intervals)>150_000_000 or hz<(18 if key=='cam0' else 25):
                    raise ValueError(f'Camera cadence invalid: {key}')

    def fresh(self, *, require_cameras=True):
        self.feedback()
        with self.lock:
            now, wall = time.monotonic_ns(), time.time_ns()
            rows = []
            if require_cameras:
                for key,buf in self.cameras.items():
                    if not buf: raise ValueError(f'Missing camera {key}')
                    rows.append(buf[-1])
            # Latest streams must be fresh even when a source-aligned snapshot
            # selects older samples. A buffer cannot cure a dead stream.
            if rows:self.timing(rows,wall,now,check_span=False,labels=list(self.cameras))
            self.native.validate_latest(now_ns=now)

    def snapshot(self, prompt, *, fast=False, encode_images=True):
        self.fresh(require_cameras=not fast)
        with self.lock:
            if fast:
                batch = self.native.snapshot()
                cameras = None
            else:
                self.camera_ready()
                anchor=min(buf[-1][0] for buf in self.cameras.values())
                cameras={key:before(buf,anchor) for key,buf in self.cameras.items()}
                arms=[before(buf,anchor) for buf in self.arms.values()]
                hands=[before(buf,anchor) for buf in self.hands.values()]
                batch=self.native.snapshot(anchor_wall=anchor)
                self.timing([*cameras.values(),*arms,*hands,
                             (batch['wall_ns'],None,batch['sampled_mono_ns'])],time.time_ns(),time.monotonic_ns(),
                            labels=['cam0','cam1','cam2','left_arm','right_arm','left_hand','right_hand','tactile'])
            result={'tactile_f6':batch['tactile_f6'],
                    'tactile_deform':batch['tactile_deform'],
                    'history_started_mono_ns':int(np.min(batch['received_mono_ns'][0])),
                    'sampled_mono_ns':batch['sampled_mono_ns'],'prompt':prompt}
        if cameras is not None and encode_images:
            for key,(_,message,_) in cameras.items():
                packed=rgb_image(message)
                result[key]=np.frombuffer(base64.b64decode(packed['rgb_b64']),dtype=np.uint8).reshape(packed['shape']).copy()
        return result
