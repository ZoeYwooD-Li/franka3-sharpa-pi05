"""Failure evidence survives decoding, background inference, and shutdown."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.actions import TrexActionDecoder
from deploy.fr3_sharpa_runtime.trex.contract import URDF_PATH
from deploy.fr3_sharpa_runtime.trex.kinematics import ArmKinematics, IKError
from deploy.fr3_sharpa_runtime.trex.replay_ik import replay


def failed_chunk(side='left'):
    measured = np.r_[[0., -.35, 0., -1.8, 0., 2., 0.], np.zeros(22),
                     [0., .35, 0., -1.8, 0., 2., 0.], np.zeros(22)]
    raw = np.zeros((16, 62))
    raw[:, 3:9] = raw[:, 34:40] = [1, 0, 0, 0, 1, 0]
    raw[2, 0 if side == 'left' else 31] = 10
    with pytest.raises(IKError) as caught:
        TrexActionDecoder().decode(raw, measured)
    return dict(raw_actions=raw.tolist(), measured_state=measured.tolist(),
                solver=caught.value.diagnostics)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_failed_row_reconstructs_from_original_chunk(side):
    record = json.loads(json.dumps(failed_chunk(side)))
    assert record['solver']['side'] == side
    assert record['solver']['action_index'] == 2
    assert replay(record)['reproduced']
    record['solver']['urdf_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='SHA256'):
        replay(record)


@pytest.mark.parametrize('max_iterations,reason', [(100, 'line_search_no_improvement'), (0, 'iteration_limit')])
def test_full_precision_distinguishes_failure_near_threshold(monkeypatch, max_iterations, reason):
    model = ArmKinematics(URDF_PATH, 'left')
    seed = np.array([0., -.35, 0., -1.8, 0., 2., 0.])
    actual = model.forward(seed)
    target = actual.copy()
    target[0, 3] += .002000012345
    monkeypatch.setattr(model, 'forward_jacobian', lambda q: (actual.copy(), np.zeros((6, 7))))
    with pytest.raises(IKError) as caught:
        model.inverse(target, seed, max_iterations=max_iterations)
    d = json.loads(json.dumps(caught.value.diagnostics))
    assert d['stop_reason'] == reason
    assert .002 < d['position_error_m'] < .0020001
    assert not d['position_within_tolerance'] and d['orientation_within_tolerance']
    assert d['iterations'] == (1 if max_iterations else 0)
    np.testing.assert_array_equal(d['final_q'], seed)


def test_worker_keeps_failed_model_output_and_never_returns_prediction(monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import inference_worker as module
    record = failed_chunk()
    class Client:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def hello(self, *a): pass
        def claim(self): pass
        def release(self): pass
        def fast(self, *a): return 7, np.asarray(record['raw_actions'])
    monkeypatch.setattr(module, 'TrexClient', Client)
    worker = module.InferenceWorker('/test', 'unused', np.full(58, -5), np.full(58, 5))
    try:
        assert worker.ready.wait(1)
        worker.submit('fast', dict(sampled_mono_ns=12, tactile_f6=None, tactile_deform=None),
                      np.asarray(record['measured_state']), 10, 7)
        worker.thread.join(timeout=3)
        assert not worker.thread.is_alive()
        with pytest.raises(RuntimeError, match='IK failed'):
            worker.poll()
        assert worker.ik_failure['chunk_id'] == 7
        assert worker.ik_failure['base_mono_ns'] == 10
        assert worker.ik_failure['sampled_mono_ns'] == 12
        assert replay(worker.ik_failure)['reproduced']
    finally:
        worker.close()


def test_iteration_limit_after_a_real_solver_update():
    model = ArmKinematics(URDF_PATH, 'left')
    seed = np.array([0., -.35, 0., -1.8, 0., 2., 0.])
    target = model.forward(seed)
    target[0, 3] += 10
    with pytest.raises(IKError) as caught:
        model.inverse(target, seed, max_iterations=1)
    d = caught.value.diagnostics
    assert d['iterations'] == 1 and d['stop_reason'] == 'iteration_limit'
    assert not np.array_equal(d['final_q'], seed)


def test_ik_record_saved_after_hardware_cleanup(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_runtime.trex import execute, hardware, inference_worker
    record = failed_chunk()
    events = []
    class Hardware:
        store = NS()
        def check_admission(self): pass
        def close(self):
            events.append('hardware_closed')
            return {'status': 'NOT_CONNECTED'}
    class Worker:
        def __init__(self, *a, **kw):
            self.ready = NS(wait=lambda **kw: False)
            self.ik_failure = record
        def poll(self): raise RuntimeError('IK failed')
        def close(self): events.append('worker_closed')
    monkeypatch.setattr(hardware, 'Hardware', Hardware)
    monkeypatch.setattr(inference_worker, 'InferenceWorker', Worker)
    args = execute.parser().parse_args(['--checkpoint', '/test', '--execute', '--enable-hardware'])
    with pytest.raises(RuntimeError, match='IK failed'):
        execute.run(args, np.full(58, -5), np.full(58, 5), np.zeros(58), {}, tmp_path)
    assert events == ['hardware_closed', 'worker_closed']
    report = json.loads((tmp_path/'report.json').read_text())
    assert report['ik_failure']['file'] == 'ik-failure.json'
    assert not report['published'] and report['policy_frames_sent'] == 0
    assert replay(json.loads((tmp_path/'ik-failure.json').read_text()))['reproduced']
