from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from deploy.fr3_sharpa_runtime import hardware_stack as stack


@pytest.fixture
def mocked_stack(monkeypatch, tmp_path):
    calls = []
    processes = []
    monkeypatch.setenv("ORBBEC_SDK_LICENSE_ACCEPTED", "YES")
    monkeypatch.setattr(stack.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(stack.subprocess, "Popen", lambda command, **kwargs: (
        processes.append((command, kwargs)) or SimpleNamespace(pid=12345, poll=lambda: None)))
    monkeypatch.setattr(stack.time, "sleep", lambda _: None)
    monkeypatch.setattr(stack, "guarded_running", lambda *_: False)
    return calls, processes, tmp_path


def test_background_stack_reuses_existing_entrypoints(mocked_stack):
    calls, processes, tmp = mocked_stack
    stack.start(Path("/gello"), Path("/private/full.json"), tmp)
    starts = [command[-1] for command in calls if command[-3:-1] == ["up", "-d"]]
    assert starts == ["franka-control", "teleop-control", "openpi-cameras"]
    assert calls[4][0] == "pixi"  # Permit is checked after all three services.
    assert "compose.openpi.yaml" in " ".join(calls[2])
    assert processes[0][0] == ["/gello/ops/run/run_sharpa_openpi_guarded.sh", "/private/full.json"]
    assert processes[0][1]["start_new_session"] is True
    assert not any("runtime.sh" in " ".join(command) for command in calls)


def test_invalid_existing_permit_blocks_only_sharpa(mocked_stack, monkeypatch):
    calls, processes, tmp = mocked_stack
    def fail(command, **kwargs):
        calls.append(command)
        if command[0] == "pixi":
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(stack.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        stack.start(Path("/gello"), Path("/private/full.json"), tmp)
    starts = [command[-1] for command in calls if command[-3:-1] == ["up", "-d"]]
    assert starts == ["franka-control", "teleop-control", "openpi-cameras"]
    assert calls[-1][0] == "pixi"
    assert not processes


def test_base_startup_failure_never_starts_sharpa(mocked_stack, monkeypatch):
    calls, processes, tmp = mocked_stack
    def fail(command, **kwargs):
        calls.append(command)
        if command[-3:] == ["up", "-d", "franka-control"]:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(stack.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        stack.start(Path("/gello"), Path("/private/full.json"), tmp)
    assert not any(command[0] == "pixi" for command in calls)
    assert not processes


def test_existing_guarded_process_not_started_twice(mocked_stack, monkeypatch):
    _, processes, tmp = mocked_stack
    monkeypatch.setattr(stack, "guarded_running", lambda *_: True)
    stack.start(Path("/gello"), Path("/private/full.json"), tmp)
    assert not processes


def test_plan_has_no_mutations(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.argv", ["stack", "--plan", "--full-config", "/private/full.json",
                                     "--state-dir", str(tmp_path / "not-created")])
    monkeypatch.setattr(stack.subprocess, "run", lambda *a, **k: pytest.fail("plan started a process"))
    stack.main()
    assert not (tmp_path / "not-created").exists()
    assert "openpi-cameras" in capsys.readouterr().out
