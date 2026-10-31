import pytest

from deploy.fr3_sharpa_runtime.trex import hardware
from deploy.fr3_sharpa_runtime.trex.pi05_shutdown import DisableSettings,disable_and_confirm


class Hand:
    def __init__(self,error=None):
        self.error=error;self._wave=None;self.is_connected=False;self.connection_state='DISCONNECTED'
        self.stops=0;self.reads=0;self.disconnected=False

    def connect(self):
        if self.error:
            self.connection_state='FAILED'
            raise RuntimeError(self.error)
        self._wave=object();self.is_connected=True;self.connection_state='CONNECTED'

    def stop(self):
        assert self._wave is not None,'Cannot disable through a destroyed native handle'
        self.stops+=1

    def read_enable_state(self):
        assert self._wave is not None,'Cannot read through a destroyed native handle'
        self.reads+=1;return False

    def disconnect(self):
        self.disconnected=True;self._wave=None;self.is_connected=False


def settings():return DisableSettings(timeout_s=.2,max_attempts=3,retry_interval_s=.001)


@pytest.mark.parametrize('state',['FAILED','DISCONNECTED'])
def test_dead_handle_exits_without_retry_and_never_claims_disabled(state):
    hand=Hand();hand.connection_state=state
    result=disable_and_confirm({'left':hand,'right':None},settings(),emit=lambda *a:None)
    assert result['status']=='FAIL'
    left=result['hands']['left']
    assert left['status']=='UNCONFIRMED' and not left['confirmed_disabled']
    assert left['attempts']==0 and left['last_enabled'] is None
    assert 'handle unavailable' in left['last_error']
    assert hand.stops==hand.reads==0 and hand.disconnected
    assert result['hands']['right']['status']=='NOT_CONNECTED'


def test_live_hand_readback_is_still_required_and_preserved_on_other_failure():
    left=Hand();left.connect();right=Hand('Sharpa get_control_mode failed: Failure to connect to server')
    with pytest.raises(RuntimeError):right.connect()
    result=disable_and_confirm({'left':left,'right':right},settings(),emit=lambda *a:None)
    assert result['status']=='FAIL'
    assert result['hands']['left']['confirmed_disabled'] and left.stops==left.reads==1
    assert result['hands']['right']['status']=='UNCONFIRMED'
    assert left.disconnected and right.disconnected


def test_live_handle_still_retries_until_disabled_readback():
    hand=Hand();hand.connect()
    def readback():
        hand.reads+=1
        return hand.reads<3
    hand.read_enable_state=readback
    result=disable_and_confirm({'left':hand},settings(),emit=lambda *a:None)
    assert result['status']=='PASS'
    assert result['hands']['left']['attempts']==3
    assert hand.stops==hand.reads==3 and hand.disconnected


@pytest.mark.parametrize('mode',['recover','exhausted','enable_failure','partial_right_failure'])
def test_connection_retry_is_bounded_and_only_before_enable(monkeypatch,mode):
    obj=hardware.Hardware.__new__(hardware.Hardware)
    obj.owned={};obj.connection_attempts=[]
    calls=[];waits=[];created=[];attempt=[0]
    monkeypatch.setattr(hardware.time,'sleep',lambda value:waits.append(value))
    def factory(side,serial):
        if side=='left':attempt[0]+=1
        calls.append((attempt[0],side))
        error=None
        if side=='left':
            if (mode=='recover' and attempt[0]==1) or mode=='exhausted':
                error='Sharpa get_control_mode failed: Failure to connect to server'
            if mode=='enable_failure':error='Sharpa set_enable_state failed: Failure to connect to server'
        if side=='right' and mode=='partial_right_failure':
            error='Sharpa get_control_mode failed: Failure to connect to server'
        hand=Hand(error);created.append(hand);obj.owned[side]=hand;return hand
    if mode=='recover':obj._connect_native(factory)
    else:
        with pytest.raises(RuntimeError):obj._connect_native(factory)
    if mode=='recover':
        assert attempt[0]==2 and waits==[1.]
        assert obj.hand_executor.connected_sides=={'left','right'}
        assert obj.connection_attempts[-1]['status']=='CONNECTED'
    elif mode=='exhausted':
        assert attempt[0]==3 and waits==[1.,1.]
        assert not obj.connection_attempts[-1]['retry']
    else:
        assert attempt[0]==1 and not waits
        assert not obj.connection_attempts[-1]['retry']
    if mode=='partial_right_failure':
        # A connected (potentially enabled) left must retain its native handle
        # for confirmed shutdown, rather than disconnecting during connect().
        assert obj.hand_executor.connected_sides=={'left'}
        assert obj.owned['left']._wave is not None
        assert not obj.owned['left'].disconnected
        result=disable_and_confirm(obj.owned,settings(),emit=lambda *a:None)
        assert result['hands']['left']['confirmed_disabled']
        assert result['hands']['right']['status']=='UNCONFIRMED'
    assert all(hand.stops==0 for hand in created if hand is not obj.owned.get('left') or mode!='partial_right_failure')
