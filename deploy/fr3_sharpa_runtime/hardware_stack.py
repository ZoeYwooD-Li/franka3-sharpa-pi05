"""Start existing dependency services in the background; never send targets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def commands(repo: Path, full_config: Path):
    runtime = Path(__file__).resolve().parent
    compose = ["docker", "compose", "--profile", "openpi-hardware", "--project-directory", str(repo / "docker"),
               "--env-file", str(repo / "docker/.env"), "-f", str(repo / "docker/compose.yaml"),
               "-f", str(repo / "docker/compose.openpi.yaml"),
               "-f", str(runtime / "compose.hardware.yaml")]
    python_validation = ["python", "-c",
                "import sys; from adapters.litchibot.full_control import load_full_config; "
                "c=load_full_config(sys.argv[1]); "
                "assert c['source']=='openpi' and c['mode']=='supervised_hardware_validation', "
                "'OpenPI requires the existing supervised config and permit'", str(full_config)]
    # Pixi activation can reset PYTHONPATH; set it inside the activated shell.
    validate = ["pixi", "run", "-e", "ros2", "bash", "-lc",
                f"export PYTHONPATH={shlex.quote(str(repo))}:\"${{PYTHONPATH:-}}\"; exec {shlex.join(python_validation)}"]
    guarded = [str(repo / "ops/run/run_sharpa_openpi_guarded.sh"), str(full_config)]
    return compose, validate, guarded


def guarded_running(pid_file: Path, full_config: Path) -> bool:
    if not pid_file.exists():
        return False
    record = json.loads(pid_file.read_text())
    proc = Path(f"/proc/{int(record['pid'])}/cmdline")
    if not proc.exists():
        return False
    command = proc.read_bytes().replace(b"\0", b" ").decode()
    if "validation_driver.py" not in command and "run_sharpa_openpi_guarded.sh" not in command:
        raise RuntimeError("Recorded PID belongs to another process; refusing to reuse it")
    environment = (proc.parent / "environ").read_bytes().split(b"\0")
    bound_config = f"OPENPI_FULL_CONFIG={full_config}".encode() in environment
    if record["full_config"] != str(full_config) or not (bound_config or str(full_config) in command):
        raise RuntimeError("Guarded process already running with another configuration")
    return True


def start(repo: Path, full_config: Path, state_dir: Path):
    # Serialize invocations; the existing guarded runtime still owns device locking.
    import fcntl

    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state_dir / "startup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compose, validate, guarded = commands(repo, full_config)
        env = os.environ.copy()
        env["SHARPA_GELLO_REPO"] = str(repo)
        env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
        if env.get("ORBBEC_SDK_LICENSE_ACCEPTED") != "YES":
            raise RuntimeError("Existing camera launch requires ORBBEC_SDK_LICENSE_ACCEPTED=YES")
        subprocess.run(compose + ["config", "--quiet"], env=env, check=True)
        pid_file = state_dir / "guarded_process.json"
        running = guarded_running(pid_file, full_config)
        with (state_dir / "compose.log").open("ab") as log:
            for service in ("franka-control", "teleop-control", "openpi-cameras"):
                print(f"Starting dependency: {service}", flush=True)
                subprocess.run(compose + ["up", "-d", service], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        # The permit gates Sharpa only, not FR3, Gateway or camera startup.
        # Keep the authoritative check before entering the guarded hardware host.
        subprocess.run(validate, cwd=repo / "portable_deps/litchi_hardware", env=env, check=True)
        if not running:
            with (state_dir / "sharpa_guarded.log").open("ab") as log:
                process = subprocess.Popen(guarded, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pid_file.write_text(json.dumps({"pid": process.pid, "full_config": str(full_config)}))
            time.sleep(2)
            if process.poll() is not None:
                raise RuntimeError(f"Sharpa guarded startup exited {process.returncode}; see {state_dir / 'sharpa_guarded.log'}")
        # Keep the hand input owner resident from acquisition through execute.
        # This service only samples with permits OFF; no automatic ARM/motion.
        subprocess.run([str(Path(__file__).resolve().parent/'start_transport.sh'),
                        '--state-dir',str(state_dir)],env=env,check=True)
        subprocess.run(compose + ["ps", "-a"], env=env, check=True)
        print(f"Dependencies launched in background. Logs: {state_dir}")
        print("ROS readiness is NOT assumed. Next: ./serve.sh, then ./runtime.sh --handoff-verify")


def main():
    settings_path = Path.home() / ".config/fr3_sharpa_runtime/startup.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gello-repo", type=Path, default=Path(os.environ.get("SHARPA_GELLO_REPO", "/home/user/gello_upper_body_teleop")))
    parser.add_argument("--full-config", type=Path, default=Path(settings["full_config"]) if settings.get("full_config") else None,
                        help="Private supervised config; defaults to ~/.config/fr3_sharpa_runtime/startup.json")
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/fr3_sharpa_runtime")
    parser.add_argument("--plan", action="store_true", help="Print commands only; no files, processes or hardware are touched")
    parser.add_argument("--check-config", action="store_true", help="Validate existing config/permit only; no hardware startup")
    args = parser.parse_args()
    if args.full_config is None:
        parser.error("Specify --full-config or configure ~/.config/fr3_sharpa_runtime/startup.json")
    if "orbbec_sdk_license_accepted" in settings:
        os.environ.setdefault("ORBBEC_SDK_LICENSE_ACCEPTED", settings["orbbec_sdk_license_accepted"])
    repo, full_config = args.gello_repo.resolve(), args.full_config.resolve()
    if args.plan:
        compose, validate, guarded = commands(repo, full_config)
        for service in ("franka-control", "teleop-control", "openpi-cameras"):
            print(shlex.join(compose + ["up", "-d", service]))
        print("Existing Sharpa config/permit validation:", shlex.join(validate))
        print("Background:", shlex.join(guarded))
        print('Acquisition/handoff:',shlex.join([str(Path(__file__).resolve().parent/'start_transport.sh'),
                                               '--state-dir',str(args.state_dir)]))
        return
    try:
        if args.check_config:
            _, validate, _ = commands(repo, full_config)
            subprocess.run(validate, cwd=repo / "portable_deps/litchi_hardware", check=True)
            print("Existing guarded configuration: PASS. No hardware started.")
            return
        start(repo, full_config, args.state_dir.resolve())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        if args.check_config:
            parser.exit(1, f"Existing guarded configuration: FAIL ({error}). No hardware services started.\n")
        parser.exit(1, f"Hardware stack startup failed: {error}\nAlready started services are retained; inspect logs before retrying. No policy motion was requested.\n")


if __name__ == "__main__":
    main()
