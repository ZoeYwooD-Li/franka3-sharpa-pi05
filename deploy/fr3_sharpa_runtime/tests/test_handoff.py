import queue
import threading
import pytest
from deploy.fr3_sharpa_runtime.handoff import ControlServer,HandoffClient
from deploy.fr3_sharpa_runtime.guarded_motion_ros import promote_publishers


def test_readonly_client_cannot_request_motion(tmp_path):
    inbox=queue.Queue();server=ControlServer(tmp_path/'control.sock',inbox)
    client=HandoffClient(tmp_path/'control.sock')
    try:
        with pytest.raises(ValueError,match='enable-hardware'):
            client.request({'operation':'prepare','target':[0]*58})
        assert inbox.empty()
    finally:client.close();server.close()


def test_ipc_keeps_one_connection_across_inspection_prepare_and_chunk(tmp_path):
    inbox=queue.Queue();server=ControlServer(tmp_path/'control.sock',inbox)
    received=[]
    def consumer():
        for _ in range(3):
            payload,reply=inbox.get(timeout=2);received.append(payload)
            reply({'status':'PASS','sequence':len(received)})
    thread=threading.Thread(target=consumer);thread.start()
    client=HandoffClient(tmp_path/'control.sock',enable_hardware=True)
    try:
        for op in ('inspect','prepare','chunk'):client.request({'operation':op})
        thread.join(timeout=2)
        assert len({r['_client_id'] for r in received})==1
        assert 'enable_hardware' not in received[0]
        assert all(r['enable_hardware'] is True for r in received[1:])
    finally:client.close();server.close()


def test_promotion_reuses_both_hand_publishers():
    class Node:
        def __init__(self):self.created=[]
        def create_publisher(self,kind,topic,depth):
            result=object();self.created.append((kind,topic,result));return result
    node=Node();left=object();right=object()
    result=promote_publishers(node,(None,left,right),'Arm','Hand')
    assert result[1] is left and result[2] is right
    assert len(node.created)==1 and node.created[0][1]=='/teleop/arm_commands'


def test_existing_arm_owner_cannot_create_another_publisher():
    class Node:
        def create_publisher(self,*args):pytest.fail('must reject before creating publisher')
    with pytest.raises(RuntimeError,match='already exists'):
        promote_publishers(Node(),(object(),object(),object()),'Arm','Hand')


def test_legacy_transport_rejects_duplicate_hand_source_before_publishing():
    from deploy.fr3_sharpa_runtime.ros_transport import validate_no_existing_hand_source
    class Node:
        def get_publishers_info_by_topic(self,topic):return [object()]
    with pytest.raises(RuntimeError,match='Existing hand source'):
        validate_no_existing_hand_source(Node())


def test_readonly_verification_discards_expired_cold_start_output(monkeypatch):
    import time
    import numpy as np
    from types import SimpleNamespace
    from deploy.fr3_sharpa_runtime import run,handoff
    calls=[]
    class Client:
        def __init__(self,*args):self.seq=0
        def request(self,payload):
            assert payload=={'operation':'inspect'}
            self.seq+=1
            return {'session_id':'same','sequence_next':self.seq,'phase':'READY','motion_started':False,
                'arm_publishers':0,'supervisor':{'hardware_send':0,'hands':{'left':{'permit':False},'right':{'permit':False}}},
                'hand_publishers':{'left':[{'node':'fr3_sharpa_runtime','gid':'unchanged'}],
                                   'right':[{'node':'fr3_sharpa_runtime','gid':'unchanged2'}]}}
        def close(self):pass
    def observation(*args):
        calls.append('snapshot')
        return {'_captured_monotonic_ns':time.monotonic_ns()-(300_000_000 if len(calls)==1 else 0)}
    class Runtime:
        def __init__(self,*args):pass
        def infer_chunk(self,obs):
            calls.append('infer')
            return {'checkpoint':'ckpt','norm_stats':'stats'},np.zeros((50,58)),None
    monkeypatch.setattr(handoff,'HandoffClient',Client)
    monkeypatch.setattr(run,'live_observation',observation)
    monkeypatch.setattr(run,'SharpaRuntime',Runtime)
    report=run.verify_handoff(SimpleNamespace(handoff_socket='/socket',ros_overlay='/overlay',prompt='task',policy_url='ws://localhost'))
    assert calls==['snapshot','infer','snapshot','infer']
    assert report['status']=='PASS' and report['cold_start_output_discarded'] and not report['executed']
