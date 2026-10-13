import json
import sys

from deploy.fr3_sharpa_runtime.observation_worker import ObservationWorker


def test_worker_keeps_one_process_for_multiple_snapshots(tmp_path):
    code = """import sys,json,os
for line in sys.stdin:
 request=json.loads(line)
 if request.get('command')=='close':break
 print(json.dumps({'pid':os.getpid()}),flush=True)
"""
    worker = ObservationWorker([sys.executable, "-u", "-c", code], cwd=tmp_path)
    try:
        first = json.loads(worker.snapshot())
        second = json.loads(worker.snapshot())
        assert first["pid"] == second["pid"] == worker.process.pid
        assert worker.process.poll() is None
    finally:
        worker.close()
    assert worker.process.returncode == 0
