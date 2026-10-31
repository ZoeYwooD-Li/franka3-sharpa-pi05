import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.execution_core import (
    ChunkTimeline,Deadline,ExclusiveLease,Prediction,acquisition_check,bounded_step,frame_check,
)


def prediction(kind='slow',cid=1,base_ns=1_000_000_000,sample=1_100_000_000,value=0.):
    return Prediction(kind,cid,np.zeros(58),base_ns,sample,np.zeros((16,62)),np.full((16,58),value))


def test_timeline_keeps_partial_emitted_row_and_fixed_fk_base():
    timeline=ChunkTimeline(prediction(),speed_scale=.5)
    timeline.emitted()
    assert timeline.refine(prediction('fast',value=.01),1_150_000_000)
    assert np.all(timeline.joints[0]==0)
    assert np.all(timeline.joints[1:]==.01)
    timeline.emitted()
    assert timeline.row==1
    np.testing.assert_array_equal(timeline.base_state,np.zeros(58))


def test_fast_rejects_wrong_chunk_and_rebase_and_drops_delayed_results():
    timeline=ChunkTimeline(prediction())
    with pytest.raises(ValueError,match='another chunk'):
        timeline.refine(prediction('fast',cid=2),1_150_000_000)
    changed=prediction('fast'); changed.base_state[0]=.001
    with pytest.raises(ValueError,match='FK base'):
        timeline.refine(changed,1_150_000_000)
    assert not timeline.refine(prediction('fast',value=1),1_400_000_001)
    assert np.all(timeline.joints==0)
    with pytest.raises(ValueError,match='predates'):
        timeline.refine(prediction('fast',sample=900_000_000),1_150_000_000)


def test_horizon_is_16_not_legacy_50():
    timeline=ChunkTimeline(prediction(),speed_scale=1)
    for _ in range(16): timeline.target(); timeline.emitted()
    assert timeline.done
    with pytest.raises(StopIteration): timeline.target()


def test_emitted_slew_and_live_first_target_bounds_are_enforced():
    zero=np.zeros(58); goal=np.ones(58)
    step=bounded_step(zero,goal,.03)
    assert np.max(step)<=.001000001
    lower,upper=np.full(58,-2),np.full(58,2)
    frame_check(step,zero,zero,lower,upper,speed=.03)
    with pytest.raises(ValueError,match='slew'):
        frame_check(np.full(58,.01),zero,zero,lower,upper,speed=.03)
    arm=zero.copy(); arm[0]=.051
    with pytest.raises(ValueError,match='Arm command'): acquisition_check(arm,zero)
    hand=zero.copy(); hand[7]=.151
    with pytest.raises(ValueError,match='Hand command'): acquisition_check(hand,zero)


def test_deadlines_fail_without_catchup():
    now=[1.]
    clock=Deadline(lambda:now[0])
    clock.wait(sleep=lambda seconds:now.__setitem__(0,now[0]+seconds))
    clock.advance(); now[0]=clock.due+.034
    with pytest.raises(RuntimeError,match='catch-up'): clock.wait()


def test_exclusive_cache_lease_blocks_probe_and_other_execution():
    now=[1.]; lease=ExclusiveLease(lambda:now[0])
    lease.claim('a'*32)
    lease.check('a'*32)
    with pytest.raises(ValueError,match='another'): lease.claim('b'*32)
    with pytest.raises(ValueError,match='Another'): lease.check(None)
    with pytest.raises(ValueError): lease.release('b'*32)
    lease.release('a'*32); lease.claim('b'*32)
    now[0]+=121
    with pytest.raises(ValueError,match='expired'): lease.check('b'*32)
    lease.check(None)


@pytest.mark.parametrize('args',[
    ['--execute'],['--plan','--enable-hardware'],['--plan','--takeover-home','/test'],
    ['--plan','--command-speed-rad-s','nan'],['--plan','--num-chunks','-1'],
    ['--plan','--port','1'],['--plan','--fast-every','0'],
    ['--check-server','--enable-hardware'],['--check-server','--takeover-home','/test'],
    ['--home-only'],['--home-only','--enable-hardware','--approach-first-target'],
    ['--home-only','--enable-hardware','--hold-right-ring-pinky-aa'],
])
def test_cli_rejects_unsafe_or_inconsistent_flags_before_any_hardware(args):
    from deploy.fr3_sharpa_runtime.trex.execute import parser,check_args
    p=parser(); parsed=p.parse_args(['--checkpoint','/test',*args])
    with pytest.raises(SystemExit): check_args(parsed,p)


def test_takeover_refuses_stale_wrong_home_and_unrelated_pid(tmp_path,monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import hardware
    path=tmp_path/'home.json'
    report={'status':'HOLDING_HOME','pid':123,'home_sha256':'same','updated_unix_ns':time.time_ns()}
    monkeypatch.setattr(hardware.os,'kill',lambda *args:pytest.fail('Unexpected signal'))
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='does not match'): hardware.takeover_home(path,'other')
    report['updated_unix_ns']=1; path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='stale'): hardware.takeover_home(path,'same')
    report['updated_unix_ns']=time.time_ns(); path.write_text(json.dumps(report))
    monkeypatch.setattr(Path,'read_bytes',lambda self:b'python\0unrelated.py\0--hold\0')
    with pytest.raises(ValueError,match='not the verified'): hardware.takeover_home(path,'same')


def test_takeover_requires_confirmed_disable_and_disconnect(tmp_path,monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import hardware
    path=tmp_path/'home.json'
    report={'status':'HOLDING_HOME','pid':123,'home_sha256':'same','updated_unix_ns':time.time_ns()}
    path.write_text(json.dumps(report))
    monkeypatch.setattr(Path,'read_bytes',lambda self:b'python\0/tmp/trex_hand_home.py\0--hold\0')
    signals=[]
    def stopped(pid,number):
        signals.append((pid,number))
        report.update(status='STOPPED',shutdown={'status':'PASS'})
        path.write_text(json.dumps(report))
    monkeypatch.setattr(hardware.os,'kill',stopped)
    hardware.takeover_home(path,'same')
    assert len(signals)==1
    report.pop('shutdown'); report['status']='HOLDING_HOME'; path.write_text(json.dumps(report))
    def failed(pid,number):
        report['shutdown']={'status':'FAIL'}; path.write_text(json.dumps(report))
    monkeypatch.setattr(hardware.os,'kill',failed)
    with pytest.raises(RuntimeError,match='could not confirm'): hardware.takeover_home(path,'same')


def test_gateway_rejection_and_missing_ack_stop_commands(monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import hardware
    obj=hardware.Hardware.__new__(hardware.Hardware)
    obj.session='test'; obj.sequence=2; obj.status_lock=threading.Lock()
    obj.first_send_ns=time.monotonic_ns()-201_000_000
    obj.last_ack_ns=None; obj.ack_error=None; obj.last_ack_sequence=-1
    wall=time.time_ns()
    obj.node=NS(get_clock=lambda:NS(now=lambda:NS(nanoseconds=wall)))
    monkeypatch.setattr(hardware,'validate_gateway_graph',lambda node:None)
    with pytest.raises(RuntimeError,match='acknowledgement'): obj.check_gateway()
    obj.ack(NS(source='openpi',session_id='test',sequence=1,faults=['joint limit'],accepted_sides=[],
               header=NS(stamp=NS(sec=wall//10**9,nanosec=wall%10**9))))
    with pytest.raises(RuntimeError,match='rejected'): obj.check_gateway()


@pytest.mark.parametrize('expired,approach',[(False,False),(True,False),(False,True)])
def test_finite_execution_uses_16_rows_and_always_cleans_up(tmp_path,monkeypatch,expired,approach):
    from deploy.fr3_sharpa_runtime.trex import execute,hardware,inference_worker
    now=[2.]
    monkeypatch.setattr(execute.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(execute.time,'monotonic_ns',lambda:int(now[0]*1e9))
    class Clock:
        def __init__(self): self.due=now[0]
        def wait(self): now[0]=max(now[0],self.due)
        def check(self): pass
        def advance(self): self.due+=1/30
    monkeypatch.setattr(execute,'Deadline',Clock)
    class FakeHardware:
        instance=None
        def __init__(self):
            FakeHardware.instance=self; self.command=np.full(58,.03 if approach else 0.); self.sent=[]; self.closed=False
            self.rclpy=NS(ok=lambda:True)
            def fresh(**kwargs):
                if approach and .005<self.command[7]<.195 and Worker.instance.slow_count==1:
                    raise AssertionError('Model-input history must not gate pi05-style prepare')
            self.store=NS(feedback=lambda:self.command.copy(),fresh=fresh,
                          snapshot=lambda *args,**kwargs:{'sampled_mono_ns':int(now[0]*1e9)})
        def check_admission(self): pass
        def start_ros(self): pass
        def check_cameras(self): return {"status":"PASS"}
        def connect_hands(self): pass
        def poll(self): pass
        def send(self,q): self.command=q.copy(); self.sent.append(q.copy())
        def close(self): self.closed=True; return {'status':'PASS'}
    class Worker:
        instance=None
        def __init__(self,*args,**options):
            Worker.instance=self
            self.ready=NS(wait=lambda **kwargs:True); self.busy=False; self.result=None; self.slow_count=0
        def submit(self,kind,snapshot,base,base_ns,chunk_id=None,*,refresh=None):
            self.busy=True
            joints=np.tile(base,(16,1))
            if kind=='slow':self.slow_count+=1
            if approach and self.slow_count==1:joints[:,7]=.2
            self.result=Prediction(kind,1,base.copy(),base_ns,int(now[0]*1e9),np.zeros((16,62)),joints,int(now[0]*1e9)-(1_000_000_000 if expired else 0))
        def poll(self):
            value,self.result=self.result,None
            if value is not None:self.busy=False
            return value
        def close(self): pass
    monkeypatch.setattr(hardware,'Hardware',FakeHardware)
    monkeypatch.setattr(inference_worker,'InferenceWorker',Worker)
    args=execute.parser().parse_args(['--checkpoint','/test','--execute','--enable-hardware','--speed','fast']+(['--home','--approach-first-target'] if approach else []))
    if expired:
        with pytest.raises(RuntimeError,match='planning repeatedly expired'):
            execute.run(args,np.full(58,-5),np.full(58,5),np.zeros(58),{},tmp_path)
        report=json.loads((tmp_path/'report.json').read_text())
        assert FakeHardware.instance.closed
        assert report['status']=='FAIL' and report['completed_chunks']==0
        assert report['last_prediction_timing']['context_age_ms']>750
        return
    report=execute.run(args,np.full(58,-5),np.full(58,5),np.zeros(58),{},tmp_path)
    assert report['status']=='PASS'
    assert report['completed_chunks']==1
    assert FakeHardware.instance.closed
    rows=[json.loads(line) for line in (tmp_path/'frames.jsonl').read_text().splitlines()]
    active=[r['row'] for r in rows if r['phase']=='RUN' and r['row'] is not None]
    # Installation tick has row=0 but does not consume a model row.
    assert set(active)==set(range(16))
    if approach:
        phases=[r['phase'] for r in rows]
        assert phases.count('APPROACH')>30
        last_approach=max(i for i,p in enumerate(phases) if p=='APPROACH')
        assert 'WAIT_SENSORS' in phases[last_approach+1:]
        assert phases[0]=='HOME'
        assert report['policy_frames_sent']==16
        assert report['prepare_completion']['completion_criterion']=='policy_acquisition_bounds'
        assert not report['prepare_completion']['at_target']
        assert report['prepare_completion']['acquisition_ready']


def test_sensor_buffers_never_cure_stale_latest_stream(monkeypatch):
    from deploy.fr3_sharpa_runtime.trex.live_sensors import SensorStore
    now_mono=time.monotonic_ns(); wall=time.time_ns()
    store=SensorStore()
    with pytest.raises(ValueError,match='Waiting'): store.feedback()
    for side in ('left','right'):
        store.arms[side].append((wall,np.zeros(7),now_mono))
        store.hands[side].append((wall,np.zeros(22),now_mono))
    for camera in store.cameras:
        store.cameras[camera].append((wall-300_000_000,None,now_mono))
    with pytest.raises(ValueError,match='Stale'): store.fresh()


@pytest.mark.parametrize('offset_ms',[30,443,-250])
def test_camera_preflight_rejects_fresh_received_but_wrong_source_clock(monkeypatch,offset_ms):
    from deploy.fr3_sharpa_runtime.trex.live_sensors import SensorStore
    wall=time.time_ns(); mono=time.monotonic_ns()
    monkeypatch.setattr(time,'time_ns',lambda:wall)
    monkeypatch.setattr(time,'monotonic_ns',lambda:mono)
    store=SensorStore()
    for key,buf in store.cameras.items():
        for i in range(75,-1,-1):
            delay=i*33_333_333
            source=wall-30_000_000-delay
            if key=='cam1':source=wall+int(offset_ms*1e6)-delay
            buf.append((source,None,mono-delay))
    with pytest.raises(ValueError,match='source timestamp.*cam1'):
        store.camera_ready()


def test_camera_preflight_accepts_original_source_history_without_hands():
    from deploy.fr3_sharpa_runtime.trex.live_sensors import SensorStore
    store=SensorStore(); wall=time.time_ns(); mono=time.monotonic_ns()
    for buf in store.cameras.values():
        for i in range(75,-1,-1):
            delay=i*33_333_333
            buf.append((wall-30_000_000-delay,None,mono-delay))
    store.camera_ready()


def test_bad_camera_preflight_prevents_hand_connection_and_home(tmp_path,monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import execute,hardware,inference_worker
    closed=[]
    class Hardware:
        store=NS()
        def check_admission(self):pass
        def start_ros(self):pass
        def check_cameras(self):raise TimeoutError('cam1 future timestamp')
        def connect_hands(self):pytest.fail('Camera failure must precede hand connection')
        def send(self,q):pytest.fail('Camera failure must precede Home/motion')
        def close(self):closed.append(True);return {'status':'NOT_CONNECTED'}
    monkeypatch.setattr(hardware,'Hardware',Hardware)
    monkeypatch.setattr(inference_worker,'InferenceWorker',lambda *a:NS(ready=NS(wait=lambda **k:True),close=lambda:None))
    args=execute.parser().parse_args(['--checkpoint','/test','--execute','--enable-hardware','--home'])
    with pytest.raises(TimeoutError,match='cam1'):
        execute.run(args,np.full(58,-5),np.full(58,5),np.zeros(58),{},tmp_path)
    report=json.loads((tmp_path/'report.json').read_text())
    assert closed and not report['published']
    assert report['home_frames_sent']==report['policy_frames_sent']==0
    assert report['last_phase']=='CAMERA_PREFLIGHT'


def test_worker_failure_is_surfaced_even_if_an_old_result_is_queued():
    from deploy.fr3_sharpa_runtime.trex.inference_worker import InferenceWorker
    worker=InferenceWorker.__new__(InferenceWorker)
    worker.failure=RuntimeError('model socket lost')
    with pytest.raises(RuntimeError,match='socket lost'): worker.poll()


def test_installation_separates_original_context_and_new_tactile_deadlines():
    from dataclasses import replace
    from deploy.fr3_sharpa_runtime.trex.execution_core import installation_timing
    now=2_000_000_000
    p=replace(prediction(sample=now-100_000_000),context_mono_ns=now-500_000_000)
    accepted,timing=installation_timing(p,now)
    assert accepted and timing['context_age_ms']==500 and timing['tactile_age_ms']==100
    assert not installation_timing(replace(p,sampled_mono_ns=now-201_000_000),now)[0]
    assert not installation_timing(replace(p,context_mono_ns=now-751_000_000),now)[0]
    with pytest.raises(ValueError,match='original'):
        installation_timing(replace(p,context_mono_ns=None),now)
    with pytest.raises(ValueError,match='timestamps'):
        installation_timing(replace(p,sampled_mono_ns=now+1),now)


@pytest.mark.parametrize('initial_unavailable',[False,True])
def test_worker_refreshes_tactile_after_slow_and_decodes_only_refinement(monkeypatch,initial_unavailable):
    from deploy.fr3_sharpa_runtime.trex import inference_worker as module
    events=[]; now=time.monotonic_ns()
    def snapshot():
        events.append('snapshot')
        return dict(cam0=None,cam1=None,cam2=None,prompt='test',tactile_f6=np.ones((16,10,6)),
                    tactile_deform=None,sampled_mono_ns=now-400_000_000)
    def refresh():
        events.append('refresh')
        return dict(tactile_f6=np.full((16,10,6),2),tactile_deform=None,sampled_mono_ns=now-20_000_000)
    class Client:
        def __init__(self,*args,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def hello(self,*args): pass
        def claim(self): pass
        def release(self): pass
        def slow_and_fast(self,*args):
            events.append('slow'); return 7,np.ones((16,62))
        def fast(self,f6,deform,cid):
            assert cid==7 and np.all(f6==2)
            events.append('fast'); return 7,np.full((16,62),9)
    class Decoder:
        def decode(self,raw,base):
            assert np.all(raw==9)
            events.append('decode'); return np.zeros((16,58))
    monkeypatch.setattr(module,'TrexClient',Client)
    monkeypatch.setattr(module,'TrexActionDecoder',Decoder)
    w=module.InferenceWorker('/weights','tcp://127.0.0.1:5555',np.full(58,-5),np.full(58,5))
    try:
        assert w.ready.wait(1)
        with pytest.raises(ValueError,match='post-slow'): w.submit('slow',snapshot,np.zeros(58),now)
        if initial_unavailable:
            from deploy.fr3_sharpa_runtime.trex.execution_core import ObservationUnavailable
            def unavailable():raise ValueError('Camera-aligned history is temporarily too old')
            w.submit('slow',unavailable,np.zeros(58),now-500_000_000,refresh=refresh)
            retry=None;end=time.monotonic()+1
            while retry is None and time.monotonic()<end:
                retry=w.poll();time.sleep(.001)
            assert isinstance(retry,ObservationUnavailable)
            assert w.failure is None and not w.busy
        w.submit('slow',snapshot,np.zeros(58),now-500_000_000,refresh=refresh)
        end=time.monotonic()+2; result=None
        while result is None and time.monotonic()<end:
            result=w.poll(); time.sleep(.001)
        assert result is not None
        assert events==['snapshot','slow','refresh','fast','decode']
        assert result.context_mono_ns==now-400_000_000
        assert result.sampled_mono_ns==now-20_000_000
        assert result.chunk_id==7
        np.testing.assert_array_equal(result.base_state,np.zeros(58))
    finally: w.close()


def test_prepare_step_matches_pi05_slow_approach_math():
    from deploy.fr3_sharpa_runtime.policy_prepare import SlowApproach
    from deploy.fr3_sharpa_runtime.feedback import LiveFeedback
    from deploy.fr3_sharpa_runtime.trex.execution_core import prepare_step
    current=np.zeros(58);target=np.linspace(-.1,.2,58)
    feedback=LiveFeedback(current,(1_000_000_000,)*4,(1_000_000_000,)*4)
    reference=SlowApproach(target,current,speed_rad_s=.03)
    expected,_,_=reference.step(feedback,now_ros_ns=1_000_000_001,
                               now_monotonic_ns=1_000_000_001,enforce_execute_delta=False)
    np.testing.assert_allclose(prepare_step(current,target),expected,atol=1e-8)


def test_acquisition_failure_names_both_right_hand_joints_and_keeps_bound():
    measured=np.zeros(58); command=measured.copy()
    command[[50,55]]=-.15
    acquisition_check(command,measured)
    measured[[50,55]]=[-.041,-.046]
    command[[50,55]]=measured[[50,55]]-.149
    acquisition_check(command,measured)
    command[[50,55]]-=.002
    with pytest.raises(ValueError) as caught:
        acquisition_check(command,measured)
    message=str(caught.value)
    assert '0.15 rad' in message
    assert 'right_ring_MCP_AA' in message and 'right_pinky_MCP_AA' in message
    assert 'command=' in message and 'measured=' in message and 'error=' in message


@pytest.mark.parametrize('case',['stuck','moving','near_target','different_joints'])
def test_prepare_monitor_requires_persistent_lag_and_stationary_feedback(case):
    from deploy.fr3_sharpa_runtime.trex.execute_progress import PrepareMonitor
    monitor=PrepareMonitor(); measured=np.zeros(58); target=np.zeros(58)
    target[[50,55]]=-.2
    for tick in range(62):
        now=tick/30
        if case=='moving': measured[[50,55]]=-.01*now
        if case=='near_target': measured[[50,55]]=-.19
        command=measured.copy(); command[[50,55]]-=.05
        if case=='different_joints': command[50 if tick<30 else 55]=measured[50 if tick<30 else 55]
        stalled=monitor.update(now,command,measured,target)
        if case!='stuck' or now<2: assert stalled==[]
    if case=='stuck':
        assert {item['joint'] for item in stalled}=={'right_ring_MCP_AA','right_pinky_MCP_AA'}
        assert all(item['window_s']>=2 and item['feedback_span_rad']==0 for item in stalled)


@pytest.mark.parametrize('failure',['stall','bound'])
def test_blocked_approach_stops_before_send_and_keeps_shutdown_report(tmp_path,monkeypatch,failure):
    from deploy.fr3_sharpa_runtime.trex import execute,hardware,inference_worker
    now=[0.]
    monkeypatch.setattr(execute.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(execute.time,'monotonic_ns',lambda:int(now[0]*1e9))
    class Clock:
        def __init__(self): self.due=0.
        def wait(self): now[0]=self.due
        def check(self): pass
        def advance(self): self.due+=1/30
    class Hardware:
        instance=None
        def __init__(self):
            Hardware.instance=self; self.measured=np.zeros(58); self.measured[50]=-.04
            self.sent=[]; self.closed=False; self.rclpy=NS(ok=lambda:True)
            self.store=NS(feedback=lambda:self.measured.copy(),snapshot=lambda *a,**k:{})
        def check_admission(self): pass
        def start_ros(self): pass
        def check_cameras(self): return {"status":"PASS"}
        def connect_hands(self): pass
        def poll(self): pass
        def send(self,q): self.sent.append(q.copy())
        def close(self): self.closed=True; return {'status':'PASS'}
    class Worker:
        def __init__(self,*args,**options):
            self.ready=NS(wait=lambda **kwargs:True); self.busy=False; self.result=None
        def submit(self,kind,snapshot,base,base_ns,**kwargs):
            joints=np.tile(base,(16,1)); joints[:,50]=-.2
            self.result=Prediction(kind,3,base.copy(),base_ns,int(now[0]*1e9),
                                   np.zeros((16,62)),joints,int(now[0]*1e9))
            self.busy=True
        def poll(self):
            result,self.result=self.result,None; self.busy=False; return result
        def close(self): pass
    monkeypatch.setattr(execute,'Deadline',Clock)
    monkeypatch.setattr(hardware,'Hardware',Hardware)
    monkeypatch.setattr(inference_worker,'InferenceWorker',Worker)
    if failure=='bound':
        monkeypatch.setattr(execute,'PrepareMonitor',lambda:NS(update=lambda *args:[]))
    args=execute.parser().parse_args(['--checkpoint','/test','--execute','--enable-hardware','--approach-first-target'])
    expected='feedback stalled' if failure=='stall' else 'Hand command exceeds'
    with pytest.raises((RuntimeError,ValueError),match=expected):
        execute.run(args,np.full(58,-5),np.full(58,5),np.zeros(58),{},tmp_path)
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['status']=='FAIL' and report['last_phase']=='APPROACH'
    assert report['shutdown']['status']=='PASS' and Hardware.instance.closed
    assert report['policy_frames_sent']==report['completed_chunks']==0
    assert report['prepare_target'][50]==-.2
    progress=report['prepare_progress']
    assert progress['stable'] and not progress['at_target']
    assert progress['worst_joints'][0]['joint']=='right_ring_MCP_AA'
    if failure=='stall':
        assert now[0]<5
        assert progress['stalled_joints'][0]['joint']=='right_ring_MCP_AA'
        assert 'rejected_frame' not in report
    else:
        rejected=report['rejected_frame']
        assert rejected['sent'] is False
        assert abs(rejected['candidate'][50]-rejected['measured'][50])>.15
        np.testing.assert_array_equal(Hardware.instance.sent[-1],rejected['previous_command'])
    assert max(abs(q[50]+.04) for q in Hardware.instance.sent)<=.15
    rows=[json.loads(line) for line in (tmp_path/'frames.jsonl').read_text().splitlines()]
    assert len(rows)==len(Hardware.instance.sent)
    final=json.loads((tmp_path/'progress.json').read_text())
    assert final['status']=='FAIL' and final['shutdown']['status']=='PASS'


def test_progress_output_cannot_block_command_loop(tmp_path,monkeypatch):
    from deploy.fr3_sharpa_runtime.trex.execute_progress import ProgressReporter
    entered=threading.Event(); release=threading.Event()
    def blocked_print(*args,**kwargs):
        entered.set(); release.wait(2)
    monkeypatch.setattr('builtins.print',blocked_print)
    reporter=ProgressReporter(tmp_path)
    try:
        reporter.publish({'tick':0})
        assert entered.wait(1)
        for tick in range(1,100): reporter.publish({'tick':tick})
        assert reporter.pending.qsize()==1
    finally:
        release.set(); reporter.close()
    assert not reporter.thread.is_alive()
    assert json.loads((tmp_path/'progress.json').read_text())['tick']==99


@pytest.mark.parametrize('failure',[None,'timeout','checkpoint','sessions'])
def test_check_server_has_no_hardware_claim_or_model_request(monkeypatch,capsys,failure):
    from deploy.fr3_sharpa_runtime.trex import execute,client
    from deploy.fr3_sharpa_runtime.trex.client import TrexRequestTimeout
    calls=[]
    class Client:
        def __init__(self,endpoint,timeout_ms): calls.append((endpoint,timeout_ms))
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def hello(self,checkpoint):
            calls.append(('hello',str(checkpoint)))
            if failure=='timeout': raise TrexRequestTimeout('tcp://127.0.0.1:5555','hello','receive',3000)
            if failure=='checkpoint': raise ValueError('T-Rex server loaded another checkpoint')
            return {'exclusive_sessions':failure!='sessions','status':'success'}
    monkeypatch.setattr(client,'TrexClient',Client)
    monkeypatch.setattr(execute,'limits_and_home',lambda *args:pytest.fail('Hardware limits must not be imported'))
    monkeypatch.setattr(execute,'validate_checkpoint',lambda *args:pytest.fail('Health reads server metadata only'))
    args=['--checkpoint','/weights','--check-server']
    if failure:
        with pytest.raises(SystemExit) as caught:execute.main(args)
        assert caught.value.code==1
    else:execute.main(args)
    result=json.loads(capsys.readouterr().out)
    assert result['status']==('FAIL' if failure else 'PASS')
    assert result['published'] is result['hardware_opened'] is result['claimed'] is result['inference_requested'] is False
    assert calls==[('tcp://127.0.0.1:5555',3000),('hello','/weights')]
    if failure=='timeout': assert 'hello receive timed out' in result['error']


def test_model_handshake_failure_never_starts_ros_or_connects_hands(tmp_path,monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import execute,hardware,inference_worker
    from deploy.fr3_sharpa_runtime.trex.client import TrexRequestTimeout
    events=[]
    class Hardware:
        def __init__(self):self.store=NS(native=None)
        def check_admission(self):events.append('admission')
        def start_ros(self):pytest.fail('No ROS command path before model ready')
        def connect_hands(self):pytest.fail('No SDK connection before model ready')
        def close(self):events.append('closed');return {'status':'NOT_CONNECTED'}
    class Worker:
        def __init__(self,*args,**options):self.ready=NS(wait=lambda **kwargs:False)
        def poll(self):raise TrexRequestTimeout('tcp://127.0.0.1:5555','hello','receive',3000)
        def close(self):events.append('worker_closed')
    monkeypatch.setattr(hardware,'Hardware',Hardware)
    monkeypatch.setattr(inference_worker,'InferenceWorker',Worker)
    args=execute.parser().parse_args(['--checkpoint','/test','--execute','--enable-hardware'])
    with pytest.raises(TrexRequestTimeout,match='hello receive timed out'):
        execute.run(args,np.full(58,-5),np.full(58,5),np.zeros(58),{},tmp_path)
    result=json.loads((tmp_path/'report.json').read_text())
    assert not result['published'] and result['policy_frames_sent']==result['home_frames_sent']==0
    assert result['shutdown']['status']=='NOT_CONNECTED'
    assert events==['admission','closed','worker_closed']


@pytest.mark.parametrize('failure',[None,'feedback','gateway','shutdown'])
def test_home_only_moves_58_joints_and_exits_without_model(tmp_path,monkeypatch,failure):
    from deploy.fr3_sharpa_runtime.trex import execute,hardware,inference_worker
    now=[1.];target=np.linspace(-.02,.02,58)
    monkeypatch.setattr(execute.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(execute.time,'monotonic_ns',lambda:int(now[0]*1e9))
    class Clock:
        def __init__(self):self.due=now[0]
        def wait(self):now[0]=self.due
        def check(self):pass
        def advance(self):self.due+=1/30
    class Hardware:
        instance=None
        def __init__(self):
            Hardware.instance=self;self.position=np.zeros(58);self.sent=[];self.closed=False
            self.rclpy=NS(ok=lambda:True);self.store=NS(feedback=self.feedback)
        def feedback(self):
            if failure=='feedback' and len(self.sent)>=3:raise ValueError('feedback stale')
            return self.position.copy()
        def check_admission(self):pass
        def start_ros(self):pass
        def check_cameras(self): return {"status":"PASS"}
        def connect_hands(self):pass
        def poll(self):pass
        def send(self,q):
            if failure=='gateway' and len(self.sent)>=3:raise RuntimeError('gateway rejected')
            self.position=q.copy();self.sent.append(q.copy())
        def close(self):
            self.closed=True;return {'status':'FAIL' if failure=='shutdown' else 'PASS'}
    monkeypatch.setattr(execute,'Deadline',Clock)
    monkeypatch.setattr(hardware,'Hardware',Hardware)
    monkeypatch.setattr(inference_worker,'InferenceWorker',lambda *a,**k:pytest.fail('Home must not connect to model'))
    args=execute.parser().parse_args(['--home-only','--enable-hardware'])
    execute.check_args(args,execute.parser())
    assert args.checkpoint is None and args.home
    if failure:
        with pytest.raises((RuntimeError,ValueError)):
            execute.run(args,np.full(58,-2),np.full(58,2),target,{},tmp_path)
    else:
        execute.run(args,np.full(58,-2),np.full(58,2),target,{},tmp_path)
    report=json.loads((tmp_path/'report.json').read_text())
    instance=Hardware.instance
    assert instance.closed and report['mode']=='home_only'
    assert report['policy_frames_sent']==report['prepare_frames_sent']==report['completed_chunks']==0
    assert report['home_frames_sent']==len(instance.sent)>0
    rows=[json.loads(s) for s in (tmp_path/'frames.jsonl').read_text().splitlines()]
    assert {row['phase'] for row in rows}=={'HOME'}
    if not failure:
        assert report['status']=='PASS' and report['home_completed']
        np.testing.assert_allclose(instance.sent[-1],target,atol=1e-8)
        delta=np.diff(np.vstack([np.zeros(58),instance.sent]),axis=0)
        assert np.max(np.abs(delta))<=.001000001
    else:assert report['status']=='FAIL'


def test_home_only_main_skips_checkpoint_validation(tmp_path,monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import execute
    called=[]
    monkeypatch.setattr(execute,'validate_checkpoint',lambda *a:pytest.fail('No checkpoint for Home'))
    monkeypatch.setattr(execute,'limits_and_home',lambda args:(np.full(58,-2),np.full(58,2),np.zeros(58),{}))
    monkeypatch.setattr(execute,'run',lambda args,*a:called.append(args.home_only))
    monkeypatch.setattr(execute.fcntl,'flock',lambda *a:None)
    execute.main(['--home-only','--enable-hardware','--output-dir',str(tmp_path)])
    assert called==[True]


@pytest.mark.parametrize('target,refined_target,expected,arm_target,fast_target',[
    (-.16,-.16,'PASS',0.,None),(-.25,-.25,'FAIL',0.,None),(-.16,-.30,'FAIL',0.,None),
    (-.135,-.135,'PASS',.5,None),(-.16,-.16,'RUN_FAIL',0.,-.25),
    (-.25,-.25,'HELD_PASS',0.,-.30),
])
def test_policy_handoff_reobserves_without_requiring_home_precision(tmp_path,monkeypatch,target,refined_target,expected,arm_target,fast_target):
    from deploy.fr3_sharpa_runtime.trex import execute,hardware,inference_worker
    now=[1.]
    monkeypatch.setattr(execute.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(execute.time,'monotonic_ns',lambda:int(now[0]*1e9))
    class Clock:
        def __init__(self):self.due=now[0]
        def wait(self):now[0]=self.due
        def check(self):pass
        def advance(self):self.due+=1/30
    class Hardware:
        instance=None
        def __init__(self):
            Hardware.instance=self;self.position=np.zeros(58);self.sent=[];self.closed=False
            self.rclpy=NS(ok=lambda:True)
            self.store=NS(feedback=lambda:self.position.copy(),snapshot=lambda *a,**k:{},fresh=lambda **k:None,
                          native=NS(configure_host_gap=lambda ms:setattr(self,'host_gap_ms',ms)))
        def check_admission(self):pass
        def start_ros(self):pass
        def check_cameras(self): return {"status":"PASS"}
        def connect_hands(self):pass
        def poll(self):pass
        def send(self,q):
            self.sent.append(q.copy());self.position=q.copy();self.position[50]=max(q[50],-.04)
        def close(self):self.closed=True;return {'status':'PASS'}
    class Worker:
        instance=None
        def __init__(self,*a,**options):
            Worker.instance=self;self.ready=NS(wait=lambda **k:True)
            self.busy=False;self.result=None;self.slow_count=0;self.closed=False
        def submit(self,kind,snapshot,base,base_ns,chunk_id=None,**kwargs):
            if kind=='slow':self.slow_count+=1
            goal=target if self.slow_count==1 else refined_target
            if kind=='fast' and fast_target is not None:goal=fast_target
            joints=np.tile(base,(16,1));joints[:,50]=goal;joints[:,0]=arm_target
            if expected=='HELD_PASS':joints[:,55]=-.30
            self.result=Prediction(kind,self.slow_count,base.copy(),base_ns,int(now[0]*1e9),
                                   np.full((16,62),goal),joints,int(now[0]*1e9))
            self.busy=True
        def poll(self):
            result,self.result=self.result,None;self.busy=False;return result
        def close(self):self.closed=True
    monkeypatch.setattr(execute,'Deadline',Clock)
    monkeypatch.setattr(hardware,'Hardware',Hardware)
    monkeypatch.setattr(inference_worker,'InferenceWorker',Worker)
    args=execute.parser().parse_args(['--checkpoint','/test','--execute','--enable-hardware','--approach-first-target'])
    if expected=='HELD_PASS':
        args.hold_right_ring_pinky_aa=True
        args.tactile_host_gap_ms=85.
    if expected in ('PASS','HELD_PASS'):
        execute.run(args,np.full(58,-2),np.full(58,2),np.zeros(58),{},tmp_path)
    elif expected=='RUN_FAIL':
        with pytest.raises(ValueError,match='Hand command exceeds 0.15 rad'):
            execute.run(args,np.full(58,-2),np.full(58,2),np.zeros(58),{},tmp_path)
    else:
        with pytest.raises(RuntimeError,match='feedback stalled'):
            execute.run(args,np.full(58,-2),np.full(58,2),np.zeros(58),{},tmp_path)
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['status']==('PASS' if expected=='HELD_PASS' else 'FAIL' if expected=='RUN_FAIL' else expected)
    assert report['shutdown']['status']=='PASS'
    assert Hardware.instance.closed and Worker.instance.closed
    assert max(abs(q[50]-max(q[50],-.04)) for q in Hardware.instance.sent)<=.15
    if expected=='HELD_PASS':
        assert Hardware.instance.host_gap_ms==85.
        assert report['tactile_timing_limits']['source_gap_ms']==75.
        assert report['tactile_timing_limits']['host_gap_ms']==85.
        assert report['completed_chunks']==1 and report['completion']=='commands_sent_with_joint_hold'
        assert report['policy_frames_sent']==160
        assert report['degraded_execution']
        assert Worker.instance.slow_count==1
        assert all(q[50]==q[55]==0 for q in Hardware.instance.sent)
        assert report['policy_joint_hold']['predictions_modified']>1  # slow and fast
        assert any(row['kind']=='fast' for row in report['policy_joint_hold']['recent_prediction_audit'])
        with np.load(tmp_path/'chunk-0000.npz') as saved:
            assert np.all(saved['decoded'][:,[50,55]]==0)
            assert np.all(saved['decoded_model'][:,50]==-.25)
            assert np.all(saved['decoded_model'][:,55]==-.30)
        rows=[json.loads(s) for s in (tmp_path/'frames.jsonl').read_text().splitlines()]
        assert all(row['measured'][50]==row['measured'][55]==0 for row in rows)
    elif expected=='PASS':
        assert Worker.instance.slow_count==2  # Discard bridge seed; install a new prediction.
        assert report['completed_chunks']==1 and report['completion']=='commands_sent'
        assert report['policy_frames_sent']==160
        assert report['prepare_completion']['acquisition_ready']
        assert not report['prepare_completion']['at_target']
        assert abs(Hardware.instance.position[50]-refined_target)>.02
        if arm_target:
            assert report['prepare_completion']['stalled_joints']
            assert not report['prepare_completion']['blocking_stalled_joints']
    elif expected=='RUN_FAIL':
        assert report['last_phase']=='RUN' and report['policy_frames_sent']>0
        rejected=report['rejected_frame']
        assert rejected['sent'] is False
        assert abs(rejected['candidate'][50]-rejected['measured'][50])>.15
        assert Worker.instance.slow_count==2
    else:
        assert report['policy_frames_sent']==0
        assert report['prepare_progress']['blocking_stalled_joints']
        assert Worker.instance.slow_count==(1 if target==refined_target else 2)
    for name in report.get('prepare_prediction_files',[]):
        with np.load(tmp_path/name) as data:
            assert data['raw'].shape==(16,62) and data['decoded'].shape==(16,58)
            assert data['base'].shape==(58,)


@pytest.mark.parametrize('count',[0,2])
def test_gateway_error_records_actual_count_and_ros_configuration(monkeypatch,count):
    from deploy.fr3_sharpa_runtime.ros_transport import validate_gateway_graph
    monkeypatch.setenv('ROS_DOMAIN_ID','42')
    node=NS(get_node_names=lambda:['teleop_safety_gateway']*count+['cam1'])
    with pytest.raises(RuntimeError,match=f'observed_count={count}') as error:
        validate_gateway_graph(node)
    assert "'ROS_DOMAIN_ID': '42'" in str(error.value)
    assert 'cam1' in str(error.value)


def test_camera_preflight_allows_slow_cold_start_without_opening_hands(monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import hardware
    now=[0.]
    monkeypatch.setattr(hardware.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(hardware.time,'sleep',lambda dt:now.__setitem__(0,now[0]+dt))
    def ready():
        if now[0]<20:raise ValueError('Waiting for camera startup stability: cam0')
    h=hardware.Hardware.__new__(hardware.Hardware)
    h.store=NS(camera_ready=ready)
    result=h.check_cameras()
    assert result['status']=='PASS' and 22<=result['elapsed_s']<23


def test_camera_preflight_still_times_out_on_persistent_bad_camera(monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import hardware
    now=[0.]
    monkeypatch.setattr(hardware.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(hardware.time,'sleep',lambda dt:now.__setitem__(0,now[0]+dt))
    def ready():raise ValueError('cam1 future timestamp')
    h=hardware.Hardware.__new__(hardware.Hardware)
    h.store=NS(camera_ready=ready)
    with pytest.raises(TimeoutError,match='hands unopened.*cam1 future timestamp'):
        h.check_cameras()
    assert 60<=now[0]<60.1
