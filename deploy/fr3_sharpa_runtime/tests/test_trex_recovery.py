import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.execution_core import Prediction
from deploy.fr3_sharpa_runtime.trex.native_tactile import TactileHistoryRebuilding,TactileSynchronizationPending


@pytest.mark.parametrize('failure',[None,'timeout','feedback','following','busy','repeat','allowed_repeat','budget','skew'])
def test_interruption_holds_drops_old_chunk_and_requires_new_history(tmp_path,monkeypatch,failure):
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
            Hardware.instance=self;self.position=np.zeros(58);self.interruption=None
            self.fresh_calls=0;self.closed=False
            self.rclpy=NS(ok=lambda:True)
            self.store=NS(feedback=self.feedback,fresh=self.fresh,snapshot=self.snapshot)
        def feedback(self):
            if self.interruption is not None and now[0]-self.interruption>.1:
                if failure=='feedback':raise ValueError('Arm feedback stale')
                if failure=='following':self.position[0]=.1
            return self.position.copy()
        def fresh(self,**kw):
            self.fresh_calls+=1
            if self.fresh_calls==4 or (failure in ('repeat','allowed_repeat','budget') and self.fresh_calls in (8,12,16)):
                self.interruption=now[0]
                if failure=='skew':raise TactileSynchronizationPending('Newest finger samples exceed 100ms span')
                raise TactileHistoryRebuilding('left ring history reset')
        def snapshot(self,*a,**kw):
            if self.interruption is not None and failure=='timeout':raise ValueError('Only 1/16 frames')
            delay=2.6 if failure=='budget' else .6
            return {'sampled_mono_ns':int(now[0]*1e9),'history_started_mono_ns':int((now[0]-delay)*1e9)}
        def check_admission(self):pass
        def start_ros(self):pass
        def check_cameras(self):return {'status':'PASS'}
        def connect_hands(self):pass
        def poll(self):pass
        def send(self,q):self.position=q.copy()
        def close(self):self.closed=True;return {'status':'PASS'}
    class Worker:
        instance=None
        def __init__(self,*a):
            Worker.instance=self;self.ready=NS(wait=lambda **k:True)
            self.busy=False;self.result=None;self.cid=0;self.closed=False
        def submit(self,kind,snapshot,base,base_ns,chunk_id=None,**kw):
            if kind=='slow':self.cid+=1
            cid=self.cid if kind=='slow' else chunk_id
            joints=np.tile(base,(16,1));joints[:,7]=.02*cid
            self.result=Prediction(kind,cid,base.copy(),base_ns,int(now[0]*1e9),
                                   np.full((16,62),cid),joints,int(now[0]*1e9))
            self.busy=True
        def poll(self):
            if failure=='busy' and Hardware.instance.interruption is not None and self.cid==1:
                self.busy=True;return None
            result,self.result=self.result,None;self.busy=False;return result
        def close(self):self.closed=True
    monkeypatch.setattr(execute,'Deadline',Clock)
    monkeypatch.setattr(hardware,'Hardware',Hardware)
    monkeypatch.setattr(inference_worker,'InferenceWorker',Worker)
    args=execute.parser().parse_args(['--checkpoint','/test','--execute','--enable-hardware','--tactile-recovery-s','3'])
    if failure in ('allowed_repeat','budget'):args.tactile_max_recoveries=10
    fails=failure not in (None,'allowed_repeat','skew')
    if fails:
        expected={'timeout':'recovery timed out','feedback':'Arm feedback stale','following':'Arm command',
                  'busy':'recovery timed out','repeat':'More than 3 tactile interruptions',
                  'budget':'hold budget exceeded 10s'}[failure]
        with pytest.raises((ValueError,TimeoutError,RuntimeError),match=expected):
            execute.run(args,np.full(58,-2),np.full(58,2),np.zeros(58),{},tmp_path)
    else:execute.run(args,np.full(58,-2),np.full(58,2),np.zeros(58),{},tmp_path)
    report=json.loads((tmp_path/'report.json').read_text())
    rows=[json.loads(s) for s in (tmp_path/'frames.jsonl').read_text().splitlines()]
    holds=[r for r in rows if r['command_origin']=='tactile_recovery_hold']
    assert holds and Hardware.instance.closed and Worker.instance.closed
    expected_interruptions=3 if failure=='repeat' else 4 if failure in ('allowed_repeat','budget') else 1
    assert report['interrupted_chunks']==expected_interruptions and report['shutdown']['status']=='PASS'
    assert all(r['chunk_id'] is None for r in holds)
    from itertools import groupby
    for is_hold,group in groupby(rows,key=lambda r:r['command_origin']=='tactile_recovery_hold'):
        group=list(group)
        if is_hold:assert all(r['command']==group[0]['command'] for r in group)
    recovery=report['tactile_recoveries'][0]
    if fails:
        assert report['status']=='FAIL' and report['completed_chunks']==0
        assert recovery['status']==('REOBSERVED' if failure in ('repeat','budget') else 'STOPPED')
    else:
        assert report['status']=='PASS' and report['completed_chunks']==1
        assert recovery['status']=='REOBSERVED' and recovery['elapsed_s']>=.6
        after=[r for r in rows if r['tick']>holds[-1]['tick'] and r['command_origin']=='policy']
        assert after and all(r['chunk_id']==expected_interruptions+1 for r in after)
        assert report['installed_chunks']==expected_interruptions+1
        assert (tmp_path/'chunk-0000.npz').exists() and (tmp_path/'chunk-0001.npz').exists()


@pytest.mark.parametrize('value',['-1','3.1','nan','inf'])
def test_recovery_timeout_requires_finite_bounded_value(value):
    from deploy.fr3_sharpa_runtime.trex.execute import parser,check_args
    p=parser();args=p.parse_args(['--checkpoint','/test','--execute','--enable-hardware',f'--tactile-recovery-s={value}'])
    with pytest.raises(SystemExit):check_args(args,p)
