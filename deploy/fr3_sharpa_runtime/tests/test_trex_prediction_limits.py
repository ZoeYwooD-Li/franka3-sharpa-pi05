import time
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.execution_core import validate_positions
from deploy.fr3_sharpa_runtime.trex.prediction_limits import clip_hand_targets


@pytest.mark.parametrize('value',[-2.04,2.04])
def test_small_hand_violation_is_corrected_and_audited_without_touching_arms(value):
    joints=np.zeros((16,58),dtype=np.float32);joints[3,7]=value
    original=joints.copy();lo=np.full(58,-2.);hi=np.full(58,2.)
    clipped,audit=clip_hand_targets(joints,lo,hi,.05)
    assert audit['corrected_values']==1
    assert audit['corrections'][0]['joint']=='left_thumb_CMC_FE'
    assert clipped[3,7]==np.sign(value)*2
    np.testing.assert_array_equal(joints,original)
    np.testing.assert_array_equal(clipped[:,0:7],joints[:,0:7])
    validate_positions(clipped,lo,hi)


@pytest.mark.parametrize('joint,value',[(0,2.01),(29,-2.01),(7,2.06),(55,-2.06)])
def test_arm_or_large_hand_violation_is_never_hidden(joint,value):
    joints=np.zeros((16,58));joints[2,joint]=value
    with pytest.raises(ValueError,match='violation_count.*excess_rad'):
        clip_hand_targets(joints,np.full(58,-2.),np.full(58,2.),.05)


def test_float32_projection_stays_inside_native_float64_limits():
    joints=np.zeros((16,58),np.float32);joints[0,7]=.36
    lo=np.full(58,-.3491);hi=np.full(58,.3491)
    clipped,_=clip_hand_targets(joints,lo,hi,.05)
    assert float(clipped[0,7])<=hi[7]
    validate_positions(clipped,lo,hi)


@pytest.mark.parametrize('mode',['clip','reject','held'])
def test_worker_preserves_original_prediction_and_rejection_evidence(monkeypatch,mode):
    from deploy.fr3_sharpa_runtime.trex import inference_worker
    class Client:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def hello(self,*a):pass
        def claim(self):pass
        def release(self):pass
        def fast(self,*a):return 1,np.zeros((16,62),np.float32)
    original=np.zeros((16,58),np.float32)
    original[:,50 if mode=='held' else 7]=-3. if mode=='held' else 2.1 if mode=='reject' else 2.04
    class Decoder:
        def decode(self,*a):return original.copy()
    monkeypatch.setattr(inference_worker,'TrexClient',Client)
    monkeypatch.setattr(inference_worker,'TrexActionDecoder',Decoder)
    worker=inference_worker.InferenceWorker('/test','tcp://127.0.0.1:5555',np.full(58,-2.),np.full(58,2.),
        prediction_clip_rad=.05,held_hand_indices=(50,55) if mode=='held' else ())
    try:
        assert worker.ready.wait(1)
        worker.submit('fast',{'sampled_mono_ns':1,'tactile_f6':None,'tactile_deform':None},np.zeros(58),0,1)
        end=time.monotonic()+1;result=None
        while time.monotonic()<end:
            if worker.failure is not None:break
            result=worker.poll()
            if result is not None:break
            time.sleep(.001)
        if mode=='reject':
            with pytest.raises(RuntimeError,match='left_thumb_CMC_FE'):worker.poll()
            assert worker.rejected_prediction['decoded_model'][0,7]>2.05
            assert worker.rejected_prediction['raw'].shape==(16,62)
        else:
            assert result is not None
            np.testing.assert_array_equal(result.original_joints,original)
            assert result.joints[0,50]==0 if mode=='held' else result.joints[0,7]==2
            assert worker.clipped_values==(0 if mode=='held' else 16)
    finally:worker.close()


def test_failed_prediction_is_saved_after_cleanup_even_before_hand_connection(tmp_path,monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import execute,hardware,inference_worker
    events=[];joints=np.zeros((16,58));joints[0,7]=2.1
    class Hardware:
        store=NS()
        def check_admission(self):pass
        def close(self):events.append('hardware_closed');return {'status':'NOT_CONNECTED'}
    class Worker:
        def __init__(self,*a,**k):
            self.ready=NS(wait=lambda **k:False)
            self.rejected_prediction={'kind':'slow','chunk_id':3,'error':'left_thumb_CMC_FE exceeds native limit',
                'raw':np.zeros((16,62)),'decoded':joints,'decoded_model':joints.copy(),
                'base':np.zeros(58),'sampled_mono_ns':1}
            self.clipped_values=0;self.clipping_audit=[]
        def poll(self):raise RuntimeError('left_thumb_CMC_FE exceeds native limit')
        def close(self):events.append('worker_closed')
    monkeypatch.setattr(hardware,'Hardware',Hardware)
    monkeypatch.setattr(inference_worker,'InferenceWorker',Worker)
    args=execute.parser().parse_args(['--checkpoint','/test','--execute','--enable-hardware'])
    with pytest.raises(RuntimeError,match='left_thumb_CMC_FE'):
        execute.run(args,np.full(58,-2.),np.full(58,2.),np.zeros(58),{},tmp_path)
    assert events==['hardware_closed','worker_closed']
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['rejected_prediction']['file']=='rejected-prediction.npz'
    assert not report['published'] and report['policy_frames_sent']==0
    with np.load(tmp_path/'rejected-prediction.npz') as saved:
        np.testing.assert_array_equal(saved['decoded_model'],joints)
