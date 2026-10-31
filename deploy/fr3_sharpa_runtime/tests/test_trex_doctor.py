import threading
from types import SimpleNamespace as NS

import pytest

from deploy.fr3_sharpa_runtime.trex import doctor


@pytest.mark.parametrize('failure',[None,'server','gateway','camera','cleanup'])
def test_doctor_uses_observations_only_and_always_cleans_up(monkeypatch,failure):
    from deploy.fr3_sharpa_runtime import ros_transport
    now=[1.]; events=[]
    monkeypatch.setattr(doctor.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(doctor.time,'sleep',lambda dt:now.__setitem__(0,now[0]+dt))
    monkeypatch.setattr(doctor,'check_server',lambda *a:{'status':'FAIL','error':'server unavailable'} if failure=='server' else {'status':'PASS'})
    monkeypatch.setattr(ros_transport,'validate_gateway_graph',lambda n:events.append('graph'))
    class Store:
        cameras={k:[(1,None,1)] for k in ('cam0','cam1','cam2')}
        arms={s:[(1,None,1)] for s in ('left','right')}
        lock=threading.RLock()
        def camera_ready(self):
            if failure=='camera':raise ValueError('future source timestamp')
        def timing(self,*a,**k):pass
    class Hardware:
        store=Store(); arm_pub=None; hand_executor=None; owned={}
        node=NS(get_node_names=lambda:['teleop_safety_gateway'])
        def check_admission(self):
            events.append('admission')
            if failure=='gateway':raise RuntimeError('gateway missing')
        def start_ros(self):events.append('subscriptions')
        def check_cameras(self):return {'status':'PASS'}
        def connect_hands(self):pytest.fail('doctor must never connect/enable hands')
        def send(self,*a):pytest.fail('doctor must never send motion')
        def close(self):
            events.append('close')
            if failure=='cleanup':raise RuntimeError('executor cleanup failed')
            return {'status':'NOT_CONNECTED'}
    result=doctor.inspect(Hardware(),'/checkpoint',5555,.1)
    assert result['status']==('FAIL' if failure else 'PASS')
    assert result['joint_targets_sent']==0 and not result['published'] and not result['hardware_opened']
    assert events[-1]=='close'
    if failure=='server':assert events==['close']
    if not failure:assert result['gateway_count']==1 and result['checks']>=4
