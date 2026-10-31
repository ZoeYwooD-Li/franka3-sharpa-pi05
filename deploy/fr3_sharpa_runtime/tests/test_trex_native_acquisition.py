import threading
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
from deploy.fr3_sharpa_runtime.trex.native_tactile import NativeTactileBuffer
from deploy.fr3_sharpa_runtime.trex.acquisition import NativeSampler
from deploy.fr3_sharpa_runtime.trex.tactile import ROUTES
from deploy.fr3_sharpa_runtime.tests.test_trex_tactile import _frames


def remember(buffer,route,index):
    frame=_frames(index)[route]
    frame.wrench=[index]*6
    stamp=1_000_000_000+index*33_333_333
    buffer.remember(route,frame,stamp,stamp)


def test_independent_fingers_keep_16_original_samples_despite_asynchronous_arrival():
    b=NativeTactileBuffer()
    # Faster thumb runs one frame ahead. Its first frame must not be overwritten
    # while waiting for the other nine physical fingers.
    for index in range(17):remember(b,ROUTES[0],index)
    for route in ROUTES[1:]:
        for index in range(16):remember(b,route,index)
    batch=b.snapshot(now_ns=1_000_000_000+16*33_333_333)
    assert batch['tactile_f6'].shape==(16,10,6)
    np.testing.assert_array_equal(batch['tactile_f6'][:,0,0],np.arange(1,17))
    np.testing.assert_array_equal(batch['tactile_f6'][:,1,0],np.arange(16))
    assert np.all(np.diff(batch['frame_indices'],axis=0)==1)
    remember(b,ROUTES[0],16)  # Duplicate fetch cannot create a new history row.
    assert len(b.history[ROUTES[0]])==17


def test_a_real_native_gap_resets_that_finger_without_padding_or_borrowing_samples():
    b=NativeTactileBuffer()
    for route in ROUTES:
        for index in range(16):remember(b,route,index)
    remember(b,ROUTES[-1],19)  # 133ms actual source/receive discontinuity.
    assert b.reset_count==1 and len(b.history[ROUTES[-1]])==1
    with pytest.raises(ValueError,match='1/16 real'):
        b.snapshot(now_ns=1_000_000_000+19*33_333_333)
    assert b.diagnostics()['recent_faults'][-1]['frame_index_gap']==4


def test_native_snapshot_rejects_stale_data_and_excessive_cross_finger_skew():
    b=NativeTactileBuffer()
    for route in ROUTES:
        for index in range(16):remember(b,route,index)
    end=1_000_000_000+15*33_333_333
    with pytest.raises(ValueError,match='stale'):
        b.snapshot(now_ns=end+201_000_000)
    for index in range(16,20):remember(b,ROUTES[0],index)
    with pytest.raises(ValueError,match='span'):
        b.snapshot(now_ns=1_000_000_000+19*33_333_333)


def test_sampler_is_the_only_reader_and_is_joined_before_sdk_cleanup():
    owners=[]
    def read(executor):owners.append(threading.get_ident())
    sampler=NativeSampler(NS(native_hands=read),NS())
    try:
        assert sampler.ready.wait(1)
        end=time.monotonic()+1
        while len(owners)<4 and time.monotonic()<end:time.sleep(.002)
        sampler.check()
        assert len(owners)>=4 and len(set(owners))==1
        assert owners[0]!=threading.get_ident()
    finally:sampler.close()
    count=len(owners);time.sleep(.01)
    assert not sampler.thread.is_alive() and len(owners)==count


def test_sampler_error_is_propagated_to_command_loop():
    def fail(executor):raise ValueError('SDK read failed')
    sampler=NativeSampler(NS(native_hands=fail),NS())
    try:
        sampler.thread.join(1)
        with pytest.raises(RuntimeError,match='SDK read failed'):sampler.check()
    finally:sampler.close()


def test_differing_real_frame_counts_are_time_aligned_without_repeats_or_padding():
    b=NativeTactileBuffer()
    # Sparse index finger has genuine 66ms gaps, while the others deliver 33ms.
    # Last-16-by-count puts the oldest columns >100ms apart despite live streams.
    for route in ROUTES:
        for index in range(64):
            if route==ROUTES[1] and index%4==2:continue
            remember(b,route,index)
    sdk=np.array([[e[0] for e in list(b.history[r])[-16:]] for r in ROUTES]).T
    assert np.ptp(sdk,axis=1).max()>.1
    batch=b.snapshot(now_ns=1_000_000_000+63*33_333_333)
    assert b.realigned_snapshots==1
    assert np.ptp(batch['sdk_stamps'],axis=1).max()<=.1
    assert np.all(np.diff(batch['frame_indices'],axis=0)>0)
    assert np.all(np.diff(batch['sdk_stamps'],axis=0)<=.075)
    for col,route in enumerate(ROUTES):
        # The actual F6 was tagged with its original frame number by remember().
        np.testing.assert_array_equal(batch['tactile_f6'][:,col,0],batch['frame_indices'][:,col])
        assert batch['frame_indices'][-1,col]==b.history[route][-1][1]


def test_alignment_keeps_camera_anchor_causal_and_never_uses_newer_frames():
    b=NativeTactileBuffer()
    for route in ROUTES:
        for index in range(64):
            if route==ROUTES[1] and index%4==2:continue
            remember(b,route,index)
    anchor=1_000_000_000+59*33_333_333
    batch=b.snapshot(anchor_wall=anchor,now_ns=1_000_000_000+63*33_333_333)
    assert np.max(batch['frame_indices'])<=59
    assert np.all(np.diff(batch['frame_indices'],axis=0)>0)
    assert b.last_alignment['sdk_span_ms']<=100


def test_alignment_handles_different_isolated_frame_losses_across_all_fingers():
    rng=np.random.default_rng(20261027)
    image=np.zeros((240,240),dtype=np.uint8)
    realigned=0
    for trial in range(100):
        b=NativeTactileBuffer()
        for col,route in enumerate(ROUTES):
            previous_dropped=False
            for index in range(64):
                dropped=index not in (0,63) and not previous_dropped and rng.random()<.15
                previous_dropped=dropped
                if dropped:continue
                stamp=1_000_000_000+index*33_333_333+col*500_000
                b.history[route].append((1000+index/30+col*.0005,index,stamp,stamp,
                                        np.full(6,index,dtype=np.float32),image))
        batch=b.snapshot(now_ns=1_000_000_000+63*33_333_333+10_000_000)
        realigned+=b.realigned_snapshots
        assert np.max(np.ptp(batch['sdk_stamps'],axis=1))<=.1
        assert np.all(np.diff(batch['sdk_stamps'],axis=0)<=.075)
        assert np.all(np.diff(batch['frame_indices'],axis=0)>0)
        np.testing.assert_array_equal(batch['frame_indices'][-1],np.full(10,63))
        np.testing.assert_array_equal(batch['tactile_f6'][:,:,0],batch['frame_indices'])
    assert realigned>90  # Exercises the formerly failing ordinal-window branch.


def test_fast_uses_live_tactile_while_slow_still_rejects_stale_rgb():
    from deploy.fr3_sharpa_runtime.trex.live_sensors import SensorStore
    store=SensorStore();mono=time.monotonic_ns();wall=time.time_ns()
    for side in ('left','right'):
        store.arms[side].append((wall,np.zeros(7),mono))
        store.hands[side].append((wall,np.zeros(22),mono))
    for key in store.cameras:store.cameras[key].append((wall-300_000_000,None,mono))
    for route in ROUTES:
        for index in range(16):
            offset=(15-index)*33_333_333
            store.native.remember(route,_frames(index)[route],wall-offset,mono-offset)
    assert store.snapshot('test',fast=True)['tactile_f6'].shape==(16,10,6)
    with pytest.raises(ValueError,match='Stale/future sensor source'):
        store.snapshot('test',fast=False)


def test_uint16_rollover_preserves_original_frames_and_continuous_timestamps():
    b=NativeTactileBuffer()
    for step in range(20):
        raw=(65528+step)%65536
        for route,frame in _frames(step).items():
            frame.frame_index=raw;frame.wrench=[step]*6
            tick=1_000_000_000+step*33_333_333
            b.remember(route,frame,tick,tick)
    result=b.snapshot(now_ns=tick)
    assert b.frame_id_rollovers==10 and b.reset_count==0
    np.testing.assert_array_equal(result['frame_indices'][:,0],np.arange(65532,65548)%65536)
    np.testing.assert_array_equal(result['tactile_f6'][:,0,0],np.arange(4,20))
    assert np.all(np.diff(result['sdk_stamps'],axis=0)>0)
    assert np.all(np.diff(result['received_mono_ns'],axis=0)>0)
    archive=b.timing_records()
    assert 65535 in archive['frame_index'] and 0 in archive['frame_index']
    route=ROUTES[0];count=b.frame_count
    old=_frames(7)[route];old.frame_index=65535
    b.remember(route,old,tick+1,tick+1)
    duplicate=_frames(19)[route];duplicate.frame_index=11
    b.remember(route,duplicate,tick+2,tick+2)
    assert b.frame_count==count and b.reset_count==0


def test_rollover_does_not_hide_a_real_source_or_receive_gap():
    b=NativeTactileBuffer();route=ROUTES[0]
    for step in range(16):
        frame=_frames(step)[route];frame.frame_index=65520+step
        tick=1_000_000_000+step*33_333_333
        b.remember(route,frame,tick,tick)
    frame=_frames(20)[route];frame.frame_index=4
    b.remember(route,frame,tick+166_666_665,tick+166_666_665)
    assert b.reset_count==1 and len(b.history[route])==1
    assert b.diagnostics()['recent_faults'][-1]['frame_index_gap']==5


@pytest.mark.parametrize('limit_ms,host_extra_ms,source_skip,accepted',[
    (75.,10.8,1,False),(85.,10.8,1,True),
    (85.,25.,1,False),(85.,0.,2,False),
])
def test_host_jitter_allowance_preserves_source_continuity(limit_ms,host_extra_ms,source_skip,accepted):
    b=NativeTactileBuffer(host_gap_limit_ms=limit_ms)
    middle=('right','middle');latest=0
    for route in ROUTES:
        for step in range(16):
            index=step+(source_skip if route==middle and step>=8 else 0)
            frame=_frames(index)[route]
            mono=1_000_000_000+index*33_333_333
            if route==middle and step>=8:mono+=int(host_extra_ms*1e6)
            b.remember(route,frame,mono,mono);latest=max(latest,mono)
    if accepted:
        batch=b.snapshot(now_ns=latest)
        assert b.reset_count==0 and batch['tactile_f6'].shape==(16,10,6)
        col=ROUTES.index(middle)
        assert np.max(np.diff(batch['received_mono_ns'][:,col]))>75_000_000
        assert np.max(np.diff(batch['received_mono_ns'][:,col]))<=85_000_000
        assert np.max(np.diff(batch['sdk_stamps'][:,col]))<=.075
        history=list(b.history[middle])
        assert b._match(history,history,host_gap_limit_ns=b.host_gap_limit_ns) is not None
        assert b._match(history,history) is None
    else:
        assert b.reset_count==1
        with pytest.raises(ValueError,match='real frames'):
            b.snapshot(now_ns=latest)


@pytest.mark.parametrize('limit_ms',[float('nan'),float('inf'),74.,101.])
def test_host_jitter_limit_rejects_invalid_configuration(limit_ms):
    with pytest.raises(ValueError,match='75..100'):
        NativeTactileBuffer(host_gap_limit_ms=limit_ms)


def test_latest_live_skew_is_recoverable_but_stale_or_future_samples_are_fatal():
    from deploy.fr3_sharpa_runtime.trex.native_tactile import TactileObservationPending,TactileSynchronizationPending
    b=NativeTactileBuffer()
    for route in ROUTES:
        for index in range(16):remember(b,route,index)
    for index in range(16,20):remember(b,ROUTES[0],index)
    now=1_000_000_000+19*33_333_333
    with pytest.raises(TactileSynchronizationPending,match='sdk_span_ms'):
        b.validate_latest(now_ns=now)
    assert b.reset_count==0 and all(len(h)>=16 for h in b.history.values())
    for stamp in [now+201_000_000,now-1]:
        with pytest.raises(ValueError,match='stale/future') as caught:
            b.validate_latest(now_ns=stamp)
        assert not isinstance(caught.value,TactileObservationPending)
