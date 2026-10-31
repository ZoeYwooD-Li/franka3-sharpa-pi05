"""Independent physical-finger histories, matching the ROS tactile contract.

Asynchronous fingers never overwrite another finger's unconsumed samples.
Each model column contains 16 different original frames from that finger.
"""
from collections import deque
import time
import numpy as np
from .tactile import ROUTES


class TactileObservationPending(ValueError):
    """A live tactile stream is temporarily not eligible for model input."""


class TactileHistoryRebuilding(TactileObservationPending):
    """Positive source/receive gap cleared a real history; stream may recover."""


class TactileSynchronizationPending(TactileObservationPending):
    """All fingers are live, but their newest samples are temporarily skewed."""


class NativeTactileBuffer:
    def __init__(self, *, host_gap_limit_ms=75.):
        self.history = {r:deque(maxlen=64) for r in ROUTES}
        self.indices = {}
        self.faults = deque(maxlen=8)
        self.frame_count = 0
        self.reset_count = 0
        self.frame_id_rollovers = 0
        self.max_sdk_gap_ms = 0.
        self.max_host_gap_ms = 0.
        self.realigned_snapshots = 0
        self.last_alignment = None
        self.timing_archive = deque(maxlen=4096)
        self.configure_host_gap(host_gap_limit_ms)

    def configure_host_gap(self, milliseconds):
        if self.frame_count:
            raise ValueError('Configure tactile timing before acquisition starts')
        if not np.isfinite(milliseconds) or not 75 <= milliseconds <= 100:
            raise ValueError('Tactile host gap limit must be 75..100 ms')
        self.host_gap_limit_ns=int(milliseconds*1e6)

    def remember(self, route, frame, wall, mono):
        if frame is None: return
        index = int(frame.frame_index)
        previous = self.indices.get(route,-1)
        # Sharpa's raw frame_id wraps at 65536. Keep the original IDs in the
        # history/archive; source/receive clocks remain authoritative for gaps.
        serial_ids = 0 <= index < 65536 and 0 <= previous < 65536
        if serial_ids and not 0 < (index-previous) % 65536 < 32768: return
        rollover = serial_ids and index < previous
        if index <= previous and not rollover: return
        history = self.history[route]
        if rollover:
            # A delayed pre-wrap packet must never be accepted as new data.
            if not history or float(frame.timestamp_s) <= history[-1][0]: return
            self.frame_id_rollovers += 1
        self.indices[route] = index
        try:
            side,finger = route
            identity = lambda x:str(getattr(x,'value',x))
            if identity(frame.side)!=side or identity(frame.finger)!=finger:
                raise ValueError('Physical tactile identity mismatch')
            stamp = float(frame.timestamp_s)
            wrench = np.asarray(frame.wrench,dtype=np.float32)
            image = frame.deformation
            if not np.isfinite(stamp) or wrench.shape!=(6,) or not np.isfinite(wrench).all():
                raise ValueError('Invalid native tactile timestamp/F6')
            if image is None or image.encoding!='mono8' or (image.height,image.width)!=(240,240):
                raise ValueError('Missing native 240x240 mono8 deformation')
            pixels = np.frombuffer(image.data,dtype=np.uint8)
            if pixels.size!=240*240: raise ValueError('Invalid native deformation byte count')
            if history:
                sdk_gap = (stamp-history[-1][0])*1000
                host_gap = (mono-history[-1][2])/1e6
                self.max_sdk_gap_ms = max(self.max_sdk_gap_ms,sdk_gap)
                self.max_host_gap_ms = max(self.max_host_gap_ms,host_gap)
                if not 0 < sdk_gap <= 75 or not 0 < mono-history[-1][2] <= self.host_gap_limit_ns:
                    self.reset_count+=1
                    self.faults.append({'route':list(route),'error':'Native finger time discontinuity',
                        'sdk_gap_ms':sdk_gap,'host_gap_ms':host_gap,
                        'frame_index_gap':(index-history[-1][1]) % 65536 if rollover else index-history[-1][1],'mono_ns':mono})
                    history.clear()
            # Start a NEW genuine history after a discontinuity; never pad it.
            history.append((stamp,index,mono,wall,wrench.copy(),pixels.reshape(240,240).copy()))
            self.timing_archive.append((ROUTES.index(route),index,stamp,mono,wall))
            self.frame_count+=1
        except (ValueError,TypeError,AttributeError) as error:
            history.clear(); self.reset_count+=1
            self.faults.append({'route':list(route),'error':str(error),'mono_ns':mono})

    def validate_latest(self, *, now_ns=None):
        """Liveness for a command tick, independent of history-row selection."""
        now = time.monotonic_ns() if now_ns is None else now_ns
        latest=[]
        for route in ROUTES:
            h=self.history[route]
            if len(h)<16:
                message=f'Native tactile {route} has {len(h)}/16 real frames; '+str(self.diagnostics(now_ns=now))
                faults=[f for f in self.faults if f.get('route')==list(route)]
                if (faults and faults[-1]['error']=='Native finger time discontinuity'
                        and faults[-1].get('sdk_gap_ms',0)>0 and faults[-1].get('host_gap_ms',0)>0):
                    raise TactileHistoryRebuilding(message)
                raise ValueError(message)
            latest.append(h[-1])
        stale={f'{side}/{finger}':(now-e[2])/1e6
               for (side,finger),e in zip(ROUTES,latest) if not 0 <= now-e[2] <= 200_000_000}
        if stale:
            raise ValueError('Native tactile sample stale/future-dated; age_ms='+str(stale))
        sdk_span_ms=float(np.ptp([e[0] for e in latest])*1000)
        host_span_ms=float(np.ptp([e[3] for e in latest])/1e6)
        if sdk_span_ms>100 or host_span_ms>100:
            # Ages were checked above. The same bounded recovery may wait for
            # a live lagging stream without accepting an unsynchronized input.
            raise TactileSynchronizationPending('Native latest ten-finger acquisition span exceeds 100ms; '+
                str({'sdk_span_ms':sdk_span_ms,'host_span_ms':host_span_ms,
                     'latest_age_ms':{f'{s}/{f}':(now-e[2])/1e6 for (s,f),e in zip(ROUTES,latest)}}))

    @staticmethod
    def _match(reference, candidates, *, host_gap_limit_ns=75_000_000):
        """Minimum-error ordered match of real frames, with no reuse/interpolation.

        Every selected transition retains 75ms source continuity and the
        configured host bound. Pin the final sample to the newest candidate, so alignment cannot
        hide a dead stream by selecting an older, easier window.
        """
        stamps=np.array([e[0] for e in candidates])
        mono=np.array([e[2] for e in candidates],dtype=np.int64)
        wall=np.array([e[3] for e in candidates],dtype=np.int64)
        ref_sdk=np.array([e[0] for e in reference])
        ref_wall=np.array([e[3] for e in reference],dtype=np.int64)
        n=len(candidates)
        sdk_step=stamps[:,None]-stamps[None,:]
        mono_step=mono[:,None]-mono[None,:]
        transitions=(sdk_step>0)&(sdk_step<=.075)&(mono_step>0)&(mono_step<=host_gap_limit_ns)
        sdk_error=np.abs(ref_sdk[:,None]-stamps[None,:])
        host_error=np.abs(ref_wall[:,None]-wall[None,:])
        error=sdk_error*1000+host_error/1e6
        error[(sdk_error>.1)|(host_error>100_000_000)]=np.inf
        cost=error[0].copy(); parents=np.full((16,n),-1,dtype=int)
        for row in range(1,16):
            options=np.where(transitions,cost[None,:],np.inf)
            parent=np.argmin(options,axis=1)
            cost=options[np.arange(n),parent]+error[row]
            parents[row]=parent
        if not np.isfinite(cost[-1]): return None
        selected=[n-1]
        for row in range(15,0,-1):selected.append(int(parents[row,selected[-1]]))
        return [candidates[i] for i in reversed(selected)]

    def snapshot(self, *, anchor_wall=None, include_arrays=True, now_ns=None):
        now = time.monotonic_ns() if now_ns is None else now_ns
        self.validate_latest(now_ns=now)
        histories=[]
        rows=[]
        for route in ROUTES:
            history=self.history[route]
            selected=list(history) if anchor_wall is None else [e for e in history if e[3]<=anchor_wall]
            if len(selected)<16:
                raise ValueError(f'Native tactile {route} has {len(selected)}/16 real frames; '+str(self.diagnostics(now_ns=now)))
            histories.append(selected)
            rows.append(selected[-16:])
        # Equal ordinal positions are not equal times when one finger misses
        # a frame. Align original samples only when the direct window fails.
        direct_sdk=np.array([[e[0] for e in row] for row in rows]).T
        direct_wall=np.array([[e[3] for e in row] for row in rows],dtype=np.int64).T
        realigned=False
        if np.any(np.ptp(direct_sdk,axis=1)>.1) or np.any(np.ptp(direct_wall,axis=1)>100_000_000):
            # The widest real 16-frame interval is the sparse reference; dense
            # fingers can omit existing samples to match it, never invent data.
            references=sorted(range(10),key=lambda i:rows[i][-1][0]-rows[i][0][0],reverse=True)
            matched=None
            for reference_index in references:
                reference=rows[reference_index]
                trial=[reference if i==reference_index else self._match(reference,h,host_gap_limit_ns=self.host_gap_limit_ns) for i,h in enumerate(histories)]
                if any(r is None for r in trial):continue
                sdk_span=np.ptp(np.array([[e[0] for e in row] for row in trial]).T,axis=1)
                host_span=np.ptp(np.array([[e[3] for e in row] for row in trial],dtype=np.int64).T,axis=1)
                if np.max(sdk_span)<=.1 and np.max(host_span)<=100_000_000:
                    matched=trial;break
            if matched is None:
                raise ValueError(f'No 16-row original tactile alignment within 75ms source / {self.host_gap_limit_ns/1e6:g}ms host continuity / 100ms span; '+
                                 str({'ordinal_sdk_span_ms':float(np.ptp(direct_sdk,axis=1).max()*1000),
                                      'ordinal_host_span_ms':float(np.ptp(direct_wall,axis=1).max()/1e6)}))
            rows=matched;realigned=True;self.realigned_snapshots+=1
        sdk=np.array([[e[0] for e in row] for row in rows]).T
        mono=np.array([[e[2] for e in row] for row in rows],dtype=np.int64).T
        wall=np.array([[e[3] for e in row] for row in rows],dtype=np.int64).T
        if (np.any(np.diff(sdk,axis=0)<=0) or np.any(np.diff(sdk,axis=0)>.075)
                or np.any(np.diff(mono,axis=0)<=0) or np.any(np.diff(mono,axis=0)>self.host_gap_limit_ns)):
            raise ValueError('Native per-finger 16-frame history is not continuous')
        if np.any(np.ptp(sdk,axis=1)>.1) or np.any(np.ptp(wall,axis=1)>100_000_000):
            raise ValueError('Native ten-finger history acquisition span exceeds 100ms')
        if np.any(now-mono[-1]<0) or np.any(now-mono[-1]>200_000_000):
            raise ValueError('Native tactile sample stale/future-dated')
        self.last_alignment={'realigned':realigned,'sdk_span_ms':float(np.ptp(sdk,axis=1).max()*1000),
                             'host_span_ms':float(np.ptp(wall,axis=1).max()/1e6)}
        result={'sampled_mono_ns':int(min(mono[-1])), 'wall_ns':int(min(wall[-1])),
                'sdk_stamps':sdk,'received_mono_ns':mono,'frame_indices':np.array([[e[1] for e in row] for row in rows]).T}
        if include_arrays:
            result.update(tactile_f6=np.stack([np.stack([e[4] for e in row]) for row in rows],axis=1),
                          tactile_deform=np.stack([row[-1][5] for row in rows]))
        return result

    def diagnostics(self, *, now_ns=None):
        now=time.monotonic_ns() if now_ns is None else now_ns
        return {'frames_received':self.frame_count,'reset_count':self.reset_count,
                'source_gap_limit_ms':75.,'host_gap_limit_ms':self.host_gap_limit_ns/1e6,
                'frame_id_rollovers':self.frame_id_rollovers,
                'frames_per_finger':{f'{s}/{f}':len(h) for (s,f),h in self.history.items()},
                'latest_age_ms':{f'{s}/{f}':None if not h else (now-h[-1][2])/1e6 for (s,f),h in self.history.items()},
                'max_sdk_gap_ms':self.max_sdk_gap_ms,'max_host_gap_ms':self.max_host_gap_ms,
                'realigned_snapshots':self.realigned_snapshots,'last_alignment':self.last_alignment,
                'recent_faults':list(self.faults)}

    def timing_records(self):
        """Exact recent provenance for failure replay; never coerce ns to float."""
        rows=list(self.timing_archive)
        return {name:np.array([r[col] for r in rows],dtype=dtype) for col,name,dtype in
                ((0,'route_index',np.int8),(1,'frame_index',np.int64),
                 (2,'sdk_timestamp_s',np.float64),(3,'received_mono_ns',np.int64),
                 (4,'received_wall_ns',np.int64))}
