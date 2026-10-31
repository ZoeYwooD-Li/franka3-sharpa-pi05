from collections import deque
import json
import threading
import time
from types import SimpleNamespace as NS
from urllib.request import urlopen

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.monitor import LiveMonitor, camera_pixels, capture


def store_fixture():
    now, wall = time.monotonic_ns(), time.time_ns()
    msg = NS(encoding='rgb8', height=2, width=2, step=8,
             data=bytes([255,0,0,0,255,0,99,99]*2))
    return NS(lock=threading.Lock(), cameras={c:deque([(wall,msg,now)]) for c in ('cam0','cam1','cam2')},
              native=NS(history={(s,f):deque([(1.,3,now,wall,np.arange(6,dtype=float),
                                             np.full((240,240),90,np.uint8))])
                                 for s in ('left','right') for f in ('thumb','index','middle','ring','pinky')}))


def test_preview_reuses_latest_buffers_without_consuming_and_reports_staleness():
    store = store_fixture()
    route = ('left','ring')
    row = store.native.history[route][-1]
    store.native.history[route][-1] = (*row[:2], time.monotonic_ns()-500_000_000, *row[3:])
    before = store.native.history[route][-1]
    result = capture(store)
    assert set(result['cameras']) == {'cam0','cam1','cam2'}
    assert len(result['fingers']) == 10
    assert result['fingers']['left/ring']['age_ms'] >= 500
    assert result['fingers']['right/thumb']['wrench'] == list(range(6))
    import base64,io
    from PIL import Image
    encoded=result['fingers']['right/thumb']['image']
    assert encoded.startswith('data:image/png;base64,')
    np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(base64.b64decode(encoded.split(',')[1])))),
                                  store.native.history[('right','thumb')][-1][5])
    assert store.native.history[route][-1] is before
    assert len(store.native.history[route]) == 1
    with store.lock:
        assert capture(store) is None  # Preview must never wait for acquisition.


def test_camera_preview_handles_rgb_bgr_and_row_padding():
    msg = store_fixture().cameras['cam0'][-1][1]
    rgb = camera_pixels(msg)
    assert rgb.shape == (2,2,3)
    assert rgb[0,0].tolist() == [255,0,0]
    msg.encoding = 'bgr8'
    assert camera_pixels(msg)[0,0].tolist() == [0,0,255]
    msg.encoding = 'mono16'
    with pytest.raises(ValueError):camera_pixels(msg)


def test_loopback_http_preview_and_cleanup(tmp_path):
    (tmp_path/'progress.json').write_text(json.dumps({'phase':'RUN','completed_chunks':4,'total_chunks':10}))
    monitor = LiveMonitor(store_fixture(), tmp_path, 0)
    monitor.start()
    try:
        assert monitor.server.server_address[0] == '127.0.0.1'
        deadline=time.monotonic()+2
        while monitor.body == b'{}' and time.monotonic()<deadline:time.sleep(.01)
        with urlopen(monitor.url+'state',timeout=2) as response:
            data=json.load(response)
            assert response.headers['Cache-Control'] == 'no-store'
        assert data['progress']['completed_chunks'] == 4
        assert len(data['fingers']) == 10
        with urlopen(monitor.url,timeout=2) as response:
            assert 'T-Rex 实时观测' in response.read().decode()
    finally:monitor.close()
    assert not monitor.sample_thread.is_alive()
    assert not monitor.http_thread.is_alive()


def test_occupied_port_selects_free_loopback_port_without_stopping_owner(tmp_path):
    owner=LiveMonitor(store_fixture(),tmp_path,0)
    owner.start()
    other=None
    try:
        other=LiveMonitor(store_fixture(),tmp_path,owner.server.server_port)
        other.start()
        assert other.server.server_port!=owner.server.server_port
        assert other.server.server_address[0]=='127.0.0.1'
        with urlopen(owner.url,timeout=2) as response:assert response.status==200
        with urlopen(other.url,timeout=2) as response:assert response.status==200
    finally:
        if other:other.close()
        owner.close()


def test_record_and_replay_roundtrip_keeps_all_streams_and_timestamps(tmp_path):
    from deploy.fr3_sharpa_runtime.trex.replay import Recording,replay_server
    store=store_fixture()
    monitor=LiveMonitor(store,tmp_path,0,record=True)
    monitor.start()
    try:
        deadline=time.monotonic()+2
        while monitor.recorded_frames<2 and time.monotonic()<deadline:time.sleep(.01)
    finally:monitor.close()
    assert monitor.record_error is None
    recording=Recording(monitor.recording_path)
    assert len(recording.offsets)==monitor.recorded_frames>=2
    frame=recording.frame(1)
    assert len(frame['fingers'])==10 and len(frame['cameras'])==3
    assert frame['fingers']['left/ring']['frame_id']==3
    assert frame['fingers']['left/ring']['wrench']==list(range(6))
    assert frame['cameras']['cam0']['source_timestamp_ns']==str(store.cameras['cam0'][-1][0])
    assert frame['replay']['frame']==1
    server=replay_server(recording,0)
    thread=threading.Thread(target=lambda:server.serve_forever(poll_interval=.05),daemon=True);thread.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/meta') as response:
            assert json.load(response)['mode']=='replay'
        with urlopen(f'http://127.0.0.1:{server.server_port}/state?frame=1') as response:
            assert json.load(response)['fingers']['left/ring']==frame['fingers']['left/ring']
    finally:server.shutdown();server.server_close();thread.join(1)
    with monitor.recording_path.open('ab') as output:output.write(b'{"unfinished":')
    assert len(Recording(monitor.recording_path).offsets)==monitor.recorded_frames


def test_recording_limit_stops_recording_but_keeps_preview_available(tmp_path):
    monitor=LiveMonitor(store_fixture(),tmp_path,0,record=True)
    monitor.recorded_bytes=2*1024**3
    monitor.start()
    try:
        deadline=time.monotonic()+2
        while monitor.record_error is None and time.monotonic()<deadline:time.sleep(.01)
        assert '2 GiB' in monitor.record_error
        with urlopen(monitor.url+'state') as response:
            assert len(json.load(response)['cameras'])==3
    finally:monitor.close()


@pytest.mark.parametrize('port', ['-1','5555','80','65536'])
def test_monitor_rejects_invalid_or_inference_port(port):
    from deploy.fr3_sharpa_runtime.trex.execute import parser,check_args
    p=parser()
    args=p.parse_args(['--execute','--enable-hardware','--checkpoint','/test','--monitor-port',port])
    with pytest.raises(SystemExit):check_args(args,p)
