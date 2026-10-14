"""One-chunk CLI. Default and ros-test modes cannot publish ROS commands."""

from __future__ import annotations

import argparse
import atexit
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from . import config
from .runtime import SharpaRuntime
from .preflight import offline_sanity_check

DEFAULT_PROMPT = (
    "Pick up a tomato with the right hand, move it to the center, "
    "let the left hand receive it, and place it in the bowl."
)
GELLO_REPO = Path(os.environ.get("SHARPA_GELLO_REPO", "/home/user/gello_upper_body_teleop"))
ROS_PROJECT = GELLO_REPO / "portable_deps/litchi_hardware"
_observation_worker = None
_observation_overlay = None


def load_sample(path: Path, prompt: str) -> dict:
    """Offline LeRobot sample for dry-run and ros-test only."""
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path).slice(0, 1)

    def image(key):
        return np.asarray(Image.open(io.BytesIO(table[key][0].as_py()["bytes"])).convert("RGB"))

    return {
        "observation/state": np.asarray(table["observation.state"][0].as_py(), dtype=np.float32),
        "observation/image": image("observation.images.cam0"),
        # Match the physical rig and LeRobotFr3SharpaDataConfig.
        "observation/left_wrist_image": image("observation.images.cam2"),
        "observation/right_wrist_image": image("observation.images.cam1"),
        "prompt": prompt,
    }


def _start_observation_worker(overlay: Path):
    """Create the runtime-owned ROS worker once, before preflight acquisition."""
    global _observation_worker, _observation_overlay
    import shlex
    from .observation_worker import ObservationWorker

    setup = overlay / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(setup)
    repo = str(Path(__file__).resolve().parents[2])
    inner = (
        f"source {shlex.quote(str(setup))}; "
        f"export PYTHONPATH={shlex.quote(repo)}:\"${{PYTHONPATH:-}}\"; "
        "exec python -m deploy.fr3_sharpa_runtime.ros_observation --persistent"
    )
    if _observation_worker is None:
        _observation_worker = ObservationWorker(
            ["pixi", "run", "--environment", "ros2", "bash", "-lc", inner], cwd=ROS_PROJECT)
        _observation_overlay = overlay.resolve()
        atexit.register(_observation_worker.close)
    elif overlay.resolve() != _observation_overlay:
        raise ValueError("Cannot change ROS overlay within one observation runtime")
    return _observation_worker


def live_observation(overlay: Path, prompt: str) -> dict:
    """Reuse the runtime's read-only ROS node across snapshot requests."""
    line = _start_observation_worker(overlay).snapshot()
    boundary_return_ns = time.monotonic_ns()
    start = time.monotonic_ns()
    snapshot = json.loads(line)
    if "error" in snapshot:
        raise RuntimeError(f"ROS observation failed: {snapshot['error']}")
    json_decode_ms = (time.monotonic_ns()-start)/1e6
    decode_start = time.monotonic_ns()

    def decode(key):
        payload = snapshot["images"][key]
        shape = tuple(payload["shape"])
        image = np.frombuffer(base64.b64decode(payload["rgb_b64"]), dtype=np.uint8)
        if len(shape) != 3 or shape[-1] != 3 or image.size != int(np.prod(shape)):
            raise ValueError(f"Invalid live {key} image")
        return image.reshape(shape).copy()

    observation = {
        "observation/state": np.asarray(snapshot["state"], dtype=np.float32),
        "observation/image": decode("cam0"),
        "observation/left_wrist_image": decode("cam2"),
        "observation/right_wrist_image": decode("cam1"),
        "prompt": prompt,
        "_captured_monotonic_ns": snapshot["captured_monotonic_ns"],
        "_topic_stamp_ns": snapshot["topic_stamp_ns"],
    }
    observation["_timing"] = {**snapshot.get("timing", {}), **snapshot.get("serialization_timing", {}),
                              "ros_boundary_return_monotonic_ns": boundary_return_ns,
                              "json_decode_ms": json_decode_ms,
                              "observation_base64_decode_ms": (time.monotonic_ns()-decode_start)/1e6}
    return observation


def prepare_before_observation(args):
    """Prepare must succeed before a newly sampled policy observation is used."""
    import shlex
    from .prepare import PrepareConfig
    PrepareConfig.load(args.prepare_config)  # Fail before ROS for missing target angles.
    setup = args.ros_overlay / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(setup)
    repo = str(Path(__file__).resolve().parents[2])
    core = str(GELLO_REPO / "ros_ws/src/teleop_core")
    command = ["python", "-m", "deploy.fr3_sharpa_runtime.prepare_ros", "--config", str(args.prepare_config.resolve())]
    inner = (f"source {shlex.quote(str(setup))}; export PYTHONPATH={shlex.quote(repo)}:{shlex.quote(str(GELLO_REPO))}:{shlex.quote(core)}:\"${{PYTHONPATH:-}}\"; exec {shlex.join(command)}")
    result = subprocess.run(["pixi", "run", "-e", "ros2", "bash", "-lc", inner],
                            cwd=ROS_PROJECT, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"PREPARE FAIL; execute blocked: {result.stderr[-1800:]}")
    report = json.loads(result.stdout.splitlines()[-1])
    if report.get("status") != "PASS":
        raise RuntimeError("PREPARE FAIL; execute blocked")
    return report, observation_after_completion(args, report)


def observation_after_completion(args, report):
    # Require all seven real sensor stamps to follow settling, not a cached
    # pre-prepare snapshot. Existing freshness/sync checks still apply.
    deadline = time.monotonic()+5
    while time.monotonic()<deadline:
        observation = live_observation(args.ros_overlay, args.prompt)
        stamps = observation["_topic_stamp_ns"]
        if (len(stamps) == 7 and min(stamps.values()) >= report["completed_ros_ns"]
                and observation["_captured_monotonic_ns"] >= report["completed_monotonic_ns"]):
            return observation
    raise TimeoutError("Prepare settled, but no new post-prepare observation arrived; execute blocked")


def execute_policy_prepare(args):
    """Explicit hardware path: first target approach, re-infer, then full chunks."""
    import shlex
    from .motion_worker import MotionWorker
    from .policy_prepare import prepare_and_run_chunks
    from .execution_trace import ExecutionTraceWriter
    setup = args.ros_overlay / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(setup)
    repo = str(Path(__file__).resolve().parents[2])
    core = str(GELLO_REPO / "ros_ws/src/teleop_core")
    worker_module = ("deploy.fr3_sharpa_runtime.sdk_direct_motion_ros"
                     if args.sharpa_backend == "sdk_direct"
                     else "deploy.fr3_sharpa_runtime.guarded_motion_ros")
    inner = (f"source {shlex.quote(str(setup))}; "
             f"export PYTHONPATH={shlex.quote(repo)}:{shlex.quote(str(GELLO_REPO))}:{shlex.quote(str(core))}:\"${{PYTHONPATH:-}}\"; "
             f"exec python -m {worker_module} --enable-hardware")
    worker = None
    trace_writer = (ExecutionTraceWriter(args.execution_trace_dir)
                    if args.sharpa_backend == "sdk_direct" else None)
    if args.sharpa_backend == "sdk_direct":
        # Direct SDK owns the Wave devices and publishes their feedback before
        # the first observation is captured.  This avoids starting a second
        # SDK connection after a legacy sharpa_driver snapshot.
        worker = MotionWorker(["pixi", "run", "-e", "ros2", "bash", "-lc", inner],
                              cwd=ROS_PROJECT, protocol_fd=True)
        atexit.register(worker.close)
        ready = worker.request({"operation": "ready"})
        if ready.get("operation") != "ready":
            raise RuntimeError(f"sdk_direct worker did not become ready: {ready}")
    _start_observation_worker(args.ros_overlay)
    runtime = SharpaRuntime(args.policy_url)
    def infer(observation):
        # Policy cold start can make the first read-only inference stale.  Drop
        # that result, reacquire all sensors, and retry once.  The 200 ms gate
        # itself remains unchanged for every command-capable result.
        current = observation
        for attempt in range(2):
            result = runtime.infer_chunk(current)
            age = time.monotonic_ns() - current['_captured_monotonic_ns']
            if 0 <= age <= 200_000_000:
                return result
            if attempt == 0:
                print("POLICY WARMUP: discarded stale inference; reacquiring observation", file=sys.stderr)
                current = live_observation(args.ros_overlay, args.prompt)
        raise RuntimeError("Observation expired during inference; no new target/chunk will be sent")
    initial = live_observation(args.ros_overlay, args.prompt)
    # Infer and validate freshness before starting any command-capable worker.
    first = infer(initial)
    initial_used = False
    def cached_first_infer(observation):
        nonlocal initial_used
        if not initial_used:
            initial_used = True
            return first
        return infer(observation)
    if args.sharpa_backend == "litchibot" and args.handoff_socket.exists():
        from .handoff import HandoffClient
        worker=HandoffClient(args.handoff_socket,enable_hardware=True)
    elif worker is None:
        worker = MotionWorker(["pixi", "run", "-e", "ros2", "bash", "-lc", inner],
                              cwd=ROS_PROJECT, protocol_fd=args.sharpa_backend == "sdk_direct")
    try:
        result = prepare_and_run_chunks(cached_first_infer,
            lambda completed: observation_after_completion(args, completed), worker, initial,
            num_chunks=args.num_chunks, speed_rad_s=args.prepare_speed_rad_s,
            execute_speed_scale=args.execute_speed_scale,
            trace_writer=trace_writer)
        if trace_writer is not None:
            result["execution_traces"] = trace_writer.paths
        return result
    finally:
        worker.close()


def verify_handoff(args):
    from .handoff import HandoffClient
    client=HandoffClient(args.handoff_socket)
    try:
        before=client.request({'operation':'inspect'})
        observation=live_observation(args.ros_overlay,args.prompt)
        metadata,actions,_=SharpaRuntime(args.policy_url).infer_chunk(observation)
        age=time.monotonic_ns()-observation['_captured_monotonic_ns']
        warmup_discarded=age>200_000_000
        if warmup_discarded:
            # Read-only cold-start warmup: discard its output, reacquire all
            # live sensors and keep the original 200ms freshness requirement.
            observation=live_observation(args.ros_overlay,args.prompt)
            metadata,actions,_=SharpaRuntime(args.policy_url).infer_chunk(observation)
            age=time.monotonic_ns()-observation['_captured_monotonic_ns']
        if not 0<=age<=200_000_000:
            raise RuntimeError('Observation expired during handoff policy verification')
        after=client.request({'operation':'inspect'})
        if (before['session_id']!=after['session_id'] or after['sequence_next']<=before['sequence_next']
                or after['phase']!='READY' or after['motion_started'] or after['arm_publishers']!=0
                or after['supervisor'].get('hardware_send')!=0
                or any(after['supervisor']['hands'][s]['permit'] is not False for s in ('left','right'))):
            raise RuntimeError('Handoff verification requires continuously sampled READY, same session, disabled hands and zero hardware sends')
        if before['hand_publishers']!=after['hand_publishers'] or any(
                len(v)!=1 or v[0]['node']!='fr3_sharpa_runtime' for v in after['hand_publishers'].values()):
            raise RuntimeError('Handoff publisher ownership/GUID changed or duplicate source found')
        return {'status':'PASS','mode':'handoff-verify','policy_action_shape':list(actions.shape),
                'finite':bool(np.isfinite(actions).all()),'checkpoint':metadata['checkpoint'],
                'norm_stats':metadata['norm_stats'],'transport':after,'executed':False,
                'snapshot_to_action_ms':age/1e6,
                'cold_start_output_discarded':warmup_discarded,
                'sequence_before':before['sequence_next'],'sequence_after':after['sequence_next']}
    finally:client.close()


def ros_boundary(actions: np.ndarray | None, session_id: str, mode: str, overlay: Path | None,
                 enabled: bool, sharpa_backend: str = "litchibot"):
    import shlex

    transport = ["python", "-m", "deploy.fr3_sharpa_runtime.ros_transport", "--mode", mode]
    if enabled:
        transport.append("--enable-hardware")
    transport.extend(["--sharpa-backend", sharpa_backend])
    env = os.environ.copy()
    repo = str(Path(__file__).resolve().parents[2])
    setup_prefix = ""
    if overlay is not None:
        setup = overlay / "setup.bash"
        if not setup.is_file():
            raise FileNotFoundError(setup)
        setup_prefix = f"source {shlex.quote(str(setup))}; "
    # Pixi resets PYTHONPATH during activation, so source the ROS overlay inside it.
    gello = str(GELLO_REPO)
    core = f"{gello}/ros_ws/src/teleop_core"
    inner = f"{setup_prefix}export PYTHONPATH={shlex.quote(repo)}:{shlex.quote(gello)}:{shlex.quote(core)}:\"${{PYTHONPATH:-}}\"; exec {shlex.join(transport)}"
    command = ["pixi", "run", "--environment", "ros2", "bash", "-lc", inner]
    # Episode/inference state is deliberately absent: transport subscribes to
    # fresh ROS feedback after inference and checks again immediately at dispatch.
    payload = json.dumps({"actions": None if actions is None else actions.tolist(), "session_id": session_id})
    result = subprocess.run(command, cwd=ROS_PROJECT, env=env, input=payload, text=True, capture_output=True, timeout=30)
    if result.returncode and mode != "preflight-only":
        raise RuntimeError(f"ROS boundary failed ({result.returncode}): {result.stderr[-1500:]}")
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if not result.stdout.strip():
        raise RuntimeError(f"ROS boundary produced no report ({result.returncode}): {result.stderr[-1500:]}")
    report = json.loads(result.stdout.splitlines()[-1])
    if mode == "preflight-only":
        report["ros_probe_exit_code"] = result.returncode
    return report


def run_preflight_only(args) -> dict:
    """Real graph → real observations → policy → live acquisition check; no send."""
    report = ros_boundary(None, "preflight-readiness", "preflight-only", args.ros_overlay, False,
                          args.sharpa_backend)
    report["checks"]["policy"] = {"status": "NOT_CHECKED", "reason": "waiting for live graph and observations"}
    if not report["readiness_pass"]:
        return report
    try:
        observation = live_observation(args.ros_overlay, args.prompt)
    except Exception as error:
        report["checks"]["observation"] = {"status": "FAIL", "reason": str(error)}
        report["status"] = "FAIL"
        report["failures"]["observation"] = report["checks"]["observation"]
        return report
    try:
        runtime = SharpaRuntime(args.policy_url)
        metadata, actions, chunk = runtime.infer_chunk(observation)
    except Exception as error:
        report["checks"]["policy"] = {"status": "FAIL", "reason": str(error)}
        report["failures"]["policy"] = report["checks"]["policy"]
        report["status"] = "FAIL"
        return report
    observation_age_ns = time.monotonic_ns() - observation["_captured_monotonic_ns"]
    # Exactly the existing execute observation-age check, without hardware output.
    fresh = 0 <= observation_age_ns <= 200_000_000
    report = ros_boundary(actions, chunk.session_id, "preflight-only", args.ros_overlay, False,
                          args.sharpa_backend)
    report["timing"] = {"observation": observation.get("_timing", {}), "policy_client": runtime.last_timing,
                        "snapshot_to_action_ms": observation_age_ns/1e6}
    report["checks"]["policy"] = {"status": "PASS", "checkpoint": metadata["checkpoint"],
                                    "norm_stats": metadata["norm_stats"], "action_shape": list(actions.shape)}
    report["checks"]["observation"] = {
        "status": "PASS" if fresh else "FAIL", "age_after_inference_ms": observation_age_ns / 1e6,
        "reason": "fresh realtime observation" if fresh else "Observation expired during policy inference",
    }
    if not fresh:
        report["status"] = "FAIL"
        report["failures"]["observation"] = report["checks"]["observation"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "ros-test", "preflight-only", "prepare", "execute", "handoff-verify"), default="dry-run")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="Alias for --mode execute; requires --enable-hardware")
    modes.add_argument("--preflight-only", action="store_true", help="Inspect live graph, feedback and policy without command publishers")
    modes.add_argument("--prepare", action="store_true", help="Verify configured task pose and settling; motion backend pending")
    modes.add_argument('--handoff-verify',action='store_true',help='Verify persistent acquisition IPC + policy without enabling motion')
    from .handoff import DEFAULT_SOCKET
    parser.add_argument('--handoff-socket',type=Path,default=DEFAULT_SOCKET)
    parser.add_argument("--prepare-config", type=Path, help="Explicit 58D task initial pose; required before execute")
    parser.add_argument("--prepare-policy-first", action="store_true",
                        help="Execute-only: slowly approach initial policy first target through existing guards, then re-infer")
    parser.add_argument("--prepare-speed-rad-s", type=float, default=0.03,
                        help="Joint-space approach speed (rad/s); native guarded limits remain authoritative")
    parser.add_argument("--execute-speed-scale", type=float, default=1.0,
                        help="sdk_direct chunk time scale in (0,1]; 0.1 emits ten interpolation steps per policy step")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Finite number of full 50-frame chunks after policy-first prepare")
    parser.add_argument("--policy-url", default=config.POLICY_URL)
    parser.add_argument("--episode-parquet", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ros-overlay", type=Path, help="ROS overlay containing teleop_interfaces for Pixi ROS Python")
    parser.add_argument("--enable-hardware", action="store_true")
    parser.add_argument("--sharpa-backend", choices=("litchibot", "sdk_direct"), default="litchibot",
                        help="Sharpa command backend; sdk_direct bypasses the legacy litchibot validation path")
    parser.add_argument("--execution-trace-dir", type=Path, default=config.EXECUTION_TRACE_DIR,
                        help="Directory for sdk_direct per-chunk JSON execution traces")
    args = parser.parse_args()
    if args.execute or args.preflight_only or args.prepare or args.handoff_verify:
        if args.mode != "dry-run":
            parser.error("Use a mode flag or --mode, not both")
        args.mode = "execute" if args.execute else "prepare" if args.prepare else "handoff-verify" if args.handoff_verify else "preflight-only"
    if args.mode == "execute":
        if not args.enable_hardware or args.episode_parquet:
            parser.error("execute requires --enable-hardware and live ROS observations; recorded samples are forbidden")
        if args.prepare_config is None and not args.prepare_policy_first:
            parser.error("execute requires --prepare-config; prepare must finish before policy execution")
    elif args.mode == "prepare":
        if args.enable_hardware or args.episode_parquet or args.prepare_config is None:
            parser.error("prepare requires --prepare-config and forbids recorded input/hardware enable; guarded motion backend pending")
    elif args.mode in ("preflight-only","handoff-verify"):
        if args.enable_hardware or args.episode_parquet:
            parser.error("preflight-only accepts live ROS observations only and forbids --enable-hardware")
    elif args.enable_hardware:
        parser.error("offline modes do not accept hardware enable")
    elif not args.episode_parquet:
        parser.error("offline modes require --episode-parquet")
    if args.prepare_policy_first and (args.mode != 'execute' or args.prepare_config is not None):
        parser.error("--prepare-policy-first requires execute and cannot be mixed with --prepare-config")
    if (args.num_chunks < 1 or not np.isfinite(args.prepare_speed_rad_s)
            or args.prepare_speed_rad_s <= 0
            or not np.isfinite(args.execute_speed_scale)
            or not 0 < args.execute_speed_scale <= 1):
        parser.error("Positive finite prepare speed, execute speed scale in (0,1], and positive num-chunks required")
    if args.num_chunks != 1 and not args.prepare_policy_first:
        parser.error("Multiple chunks require the persistent --prepare-policy-first path")
    if args.mode != "dry-run" and args.ros_overlay is None:
        parser.error("ROS modes require --ros-overlay with teleop_interfaces built for the Pixi ROS Python")
    if args.mode=='handoff-verify':
        try:
            print(json.dumps(verify_handoff(args),indent=2))
        except Exception as error:
            print(json.dumps({'status':'FAIL','mode':'handoff-verify','executed':False,'reason':str(error)},indent=2))
            raise SystemExit(1)
        return
    if args.mode == "prepare":
        try:
            report, observation = prepare_before_observation(args)
            report["post_prepare_observation"] = "PASS"
            print(json.dumps(report, indent=2))
        except Exception as error:
            print(json.dumps({"status": "FAIL", "mode": "prepare", "published": False, "reason": str(error)}))
            raise SystemExit(1)
        return
    if args.mode == "preflight-only":
        try:
            _start_observation_worker(args.ros_overlay)
            report = run_preflight_only(args)
        except Exception as error:
            report = {"mode": "preflight-only", "status": "FAIL", "published": False,
                      "command_publishers_created": 0, "reason": str(error)}
        print(json.dumps(report, indent=2))
        if report["status"] != "PASS":
            raise SystemExit(1)
        return
    preparation = None
    if args.mode == 'execute' and args.prepare_policy_first:
        print(json.dumps(execute_policy_prepare(args), indent=2))
        return
    if args.mode == "execute":
        preparation, observation = prepare_before_observation(args)
    else:
        observation = load_sample(args.episode_parquet, args.prompt)
    metadata, actions, chunk = SharpaRuntime(args.policy_url).infer_chunk(observation)
    if args.mode == "execute" and not 0 <= time.monotonic_ns() - observation["_captured_monotonic_ns"] <= 200_000_000:
        raise RuntimeError("Observation expired during policy inference; no command will be published")
    offline_result = {"status": "NOT RUN", "blocking": False}
    if args.mode != "execute":
        offline_result = offline_sanity_check(actions, observation["observation/state"])
        print(f"OFFLINE SANITY: {offline_result['status']} (diagnostic only)", file=sys.stderr)
        print("EXECUTE PREFLIGHT: WAITING FOR LIVE ROS STATE", file=sys.stderr)
    result = {
        "mode": args.mode,
        "prepare": preparation,
        "policy_url": args.policy_url,
        "checkpoint": metadata["checkpoint"],
        "norm_stats": metadata["norm_stats"],
        "action_shape": list(actions.shape),
        "finite": bool(np.isfinite(actions).all()),
        "frames": len(chunk.frames),
        "fr3_dim": len(chunk.frames[0].fr3.positions),
        "sharpa_dims": [len(chunk.frames[0].left_sharpa.position), len(chunk.frames[0].right_sharpa.position)],
        "session_id": chunk.session_id,
        "sequence": [chunk.frames[0].sequence, chunk.frames[-1].sequence],
        "published": False,
        "offline_sanity": offline_result,
        "execute_preflight": {"status": "WAITING FOR LIVE ROS STATE"},
        "sharpa_backend": args.sharpa_backend,
    }
    if args.mode != "dry-run":
        result["ros"] = ros_boundary(actions, chunk.session_id, args.mode, args.ros_overlay,
                                      args.enable_hardware, args.sharpa_backend)
        result["published"] = result["ros"]["published"]
        result["execute_preflight"] = result["ros"]["execute_preflight"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
