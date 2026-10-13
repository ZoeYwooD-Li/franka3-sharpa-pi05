"""Execute-only persistent ROS sender; no SDK or controller-bus access."""
import argparse
import json
import queue
import sys
import threading
import time
import uuid

import numpy as np

from . import config
from .action_adapter import SharpaActionAdapter
from .feedback import ROSFeedbackProvider
from .policy_prepare import SlowApproach, target_chunk, check_validation_excursion
from .preflight import execute_preflight
from .protocol import DT_NS, LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE
from .ros_transport import (_require_parameter, construct_messages,
                            validate_action_ranges, validate_execution_graph)


def sequenced_messages(target, *, session_id, sequence, timestamp_ns):
    frame = SharpaActionAdapter().adapt_chunk(target_chunk(target),
        start_timestamp_ns=timestamp_ns, session_id=session_id, source=config.SOURCE).frames[0]
    arm, left, right = construct_messages(frame)
    arm.sequence = sequence
    for side, hand in [('left',left),('right',right)]:
        hand.header.frame_id = f"openpi:{side}:{session_id}:{sequence}"
    return arm, left, right


def promote_publishers(node, publishers, arm_type, hand_type):
    if publishers is not None and publishers[0] is not None:
        raise RuntimeError('Arm publisher already exists')
    arm=node.create_publisher(arm_type,config.FR3_TOPIC,10)
    if publishers is None:
        return (arm,node.create_publisher(hand_type,config.SHARPA_LEFT_TOPIC,10),
                node.create_publisher(hand_type,config.SHARPA_RIGHT_TOPIC,10))
    return (arm,publishers[1],publishers[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-hardware", action="store_true")
    parser.add_argument("--acquisition-only", action="store_true")
    parser.add_argument("--control-socket")
    args=parser.parse_args()
    if not args.enable_hardware and not (args.acquisition_only and args.control_socket):
        parser.error('Explicit hardware enable or acquisition-only private IPC required')
    import rclpy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from teleop_interfaces.msg import ArmCommand

    rclpy.init()
    node = rclpy.create_node("fr3_sharpa_runtime")
    feedback = ROSFeedbackProvider(node)
    inbox = queue.Queue()
    supervisor = {}
    def status(message):
        try:
            value = json.loads(message.data)
            supervisor.clear()
            supervisor.update(value=value, received=time.monotonic_ns())
        except (ValueError, TypeError):
            supervisor.clear()
    node.create_subscription(String, "/teleop/sharpa/supervisor/state", status, 10)
    def reader():
        try:
            for line in sys.stdin:
                inbox.put((json.loads(line),lambda report: print(json.dumps(report),flush=True)))
        except Exception as error:
            inbox.put(({"operation": "error", "reason": str(error)},lambda _:None))
        finally:
            inbox.put(({"operation": "close"},lambda _:None))
    server=None
    if args.control_socket:
        from .handoff import ControlServer
        server=ControlServer(args.control_socket,inbox)
    else:
        threading.Thread(target=reader, daemon=True).start()
    publishers = None
    session = f"openpi-{uuid.uuid4().hex}"
    sequence = 0
    approach = None
    chunk = None
    chunk_index = 0
    hold = None
    initial_delta = None
    prepared = False
    motion_started = False
    owner = None
    motion_reply = None
    next_tick = time.monotonic_ns()
    def reply(report):
        print(json.dumps(report, allow_nan=False), flush=True)
    def guarded_status():
        if not supervisor or not 0 <= time.monotonic_ns()-supervisor['received'] <= 200_000_000:
            raise RuntimeError("Missing/stale Sharpa supervisor status")
        value = supervisor['value']
        if value.get('phase') == 'FAULT':
            raise RuntimeError(f"Existing Sharpa guarded runtime fault: {value.get('reason')}")
        return value
    def check_graph():
        validate_execution_graph(node)
        _require_parameter(node, "teleop_safety_gateway", "allowed_sources", [config.SOURCE])
        _require_parameter(node, "sharpa_driver", "supervised_validation_guard", True)
    def completion(operation, **extra):
        return {"status": "PASS", "operation": operation, "published": True,
                "session_id": session, "sequence_end": sequence-1,
                "completed_ros_ns": node.get_clock().now().nanoseconds,
                "completed_monotonic_ns": time.monotonic_ns(), **extra}
    try:
        deadline = time.monotonic()+3
        while time.monotonic()<deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
        check_graph()
        if args.acquisition_only:
            state=guarded_status()
            if state.get('phase') not in ('WARMUP','READY') or state.get('hardware_send')!=0 or any(
                    state.get('hands',{}).get(s,{}).get('permit') is not False for s in ('left','right')):
                raise RuntimeError('Acquisition-only requires disabled hands, WARMUP/READY and zero hardware sends')
            for topic in (config.SHARPA_LEFT_TOPIC,config.SHARPA_RIGHT_TOPIC):
                if node.get_publishers_info_by_topic(topic):
                    raise RuntimeError('Another OpenPI source exists; controlled service migration required')
            hold=feedback.wait().state.copy()
            publishers=(None,node.create_publisher(JointState,config.SHARPA_LEFT_TOPIC,10),
                        node.create_publisher(JointState,config.SHARPA_RIGHT_TOPIC,10))
            next_tick=time.monotonic_ns()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.002)
            try:
                request,request_reply = inbox.get_nowait()
            except queue.Empty:
                request = None
            if request:
                operation = request.get('operation')
                if operation=='client_disconnected':
                    if motion_started and request.get('_client_id')==owner:
                        raise RuntimeError('Execute client disconnected; stopped publishing')
                    continue
                if operation=='inspect':
                    state=guarded_status()
                    request_reply({'status':'PASS','phase':state.get('phase'),
                        'supervisor':state,'session_id':session,'sequence_next':sequence,
                        'motion_started':motion_started,'arm_publishers':int(publishers is not None and publishers[0] is not None),
                        'hand_publishers':{t:[{'node':x.node_name,'gid':bytes(x.endpoint_gid).hex()}
                            for x in node.get_publishers_info_by_topic(t)]
                            for t in (config.SHARPA_LEFT_TOPIC,config.SHARPA_RIGHT_TOPIC)},
                        'mode':'execute' if motion_started else 'acquisition_only'})
                    continue
                if operation == 'close':
                    break
                if operation == 'error':
                    raise ValueError(request['reason'])
                if server is not None and request.get('enable_hardware') is not True:
                    request_reply({'status':'FAIL','reason':'Explicit --enable-hardware request required'})
                    continue
                if motion_started and request.get('_client_id')!=owner:
                    request_reply({'status':'FAIL','reason':'Execution session owned by another client'})
                    continue
                if approach is not None or chunk is not None:
                    raise RuntimeError("Overlapping motion request refused")
                # Once streaming, do not block the watchdog while servicing IPC.
                current = feedback.wait() if publishers is None else feedback.snapshot()
                state = guarded_status()
                if publishers is None:
                    check_graph()
                else:
                    validate_execution_graph(node)
                if operation == 'prepare':
                    if motion_started:
                        raise RuntimeError("Prepare is only allowed once per execution session")
                    target = target_chunk(request['target'])
                    validate_action_ranges(target)
                    initial_delta = (target[0]-current.state).tolist()
                    print(f"PREPARE: live feedback -> frozen policy first target; "
                          f"max delta={np.max(np.abs(initial_delta)):.6f} rad; "
                          f"speed={request.get('speed_rad_s', 0.03)} rad/s", file=sys.stderr)
                    # Reuse the existing validation-mode total-excursion rule.
                    # Splitting a move must not circumvent its authorization baseline.
                    if state.get('mode') == 'supervised_hardware_validation':
                        from adapters.litchibot.validation_safety import EXCURSION_RAD
                        check_validation_excursion(target[0], current.state,
                            mode=state['mode'], excursion_rad=EXCURSION_RAD)
                    approach = SlowApproach(target[0], current.state,
                                            speed_rad_s=float(request.get('speed_rad_s', 0.03)))
                    hold = current.state.copy()
                    if publishers is None:
                        for topic in (config.SHARPA_LEFT_TOPIC,config.SHARPA_RIGHT_TOPIC):
                            if node.get_publishers_info_by_topic(topic):
                                raise RuntimeError('Another hand source exists; use private handoff IPC')
                    # No hand publisher/node/session replacement on handoff.
                    publishers=promote_publishers(node,publishers,ArmCommand,JointState)
                    motion_started=True;owner=request.get('_client_id');motion_reply=request_reply
                    next_tick = time.monotonic_ns()
                elif operation == 'chunk':
                    if not prepared or publishers is None:
                        raise RuntimeError("Prepare must settle before any policy chunk")
                    candidate = np.asarray(request['actions'], dtype=np.float32)
                    validate_action_ranges(candidate)
                    execute_preflight(candidate, current, now_ros_ns=node.get_clock().now().nanoseconds,
                                      now_monotonic_ns=time.monotonic_ns())
                    # Preserve guarded raw continuity across prepare/chunk boundaries.
                    for part in (LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE):
                        if np.max(np.abs(candidate[0, part]-hold[part])) >= 0.10:
                            raise ValueError("Existing guarded raw continuity bound at chunk boundary")
                    chunk, chunk_index = candidate, 0
                    motion_reply=request_reply
                else:
                    raise ValueError(f"Unknown guarded operation: {operation}")
            if publishers is None:
                continue
            now = time.monotonic_ns()
            if now < next_tick:
                continue
            if now-next_tick > DT_NS:
                raise RuntimeError("20Hz guarded motion deadline missed; stopped publishing")
            for _ in range(32):
                rclpy.spin_once(node, timeout_sec=0.0)
            current = feedback.snapshot()
            state = guarded_status()
            validate_execution_graph(node)
            authorized = state.get('phase') == 'ARMED' and all(
                state.get('hands', {}).get(side, {}).get('permit') is True for side in ('left','right'))
            settled = False
            error = None
            if not motion_started:
                if state.get('phase') not in ('WARMUP','READY') or state.get('hardware_send')!=0 or any(
                        state.get('hands',{}).get(s,{}).get('permit') is not False for s in ('left','right')):
                    raise RuntimeError('Acquisition-only authorization changed; stopped publishing')
                hold=current.state.copy()
            elif approach is not None:
                if authorized:
                    hold, settled, error = approach.step(current,
                        now_ros_ns=node.get_clock().now().nanoseconds, now_monotonic_ns=now)
                else:
                    # Only a stationary acquisition stream; never arm/enable automatically.
                    if approach.started_ns is None:
                        approach.started_ns = now
                    if now-approach.started_ns >= approach.timeout_s*1e9:
                        raise TimeoutError("Waiting for existing guarded operator authorization; no approach movement")
            elif not authorized:
                raise RuntimeError("Existing guarded authorization lost; stopped publishing")
            elif chunk is not None:
                hold = chunk[chunk_index]
            # Acquisition applies to prepare intermediates and the first chunk
            # frame. Do not turn it into a new limit on every model chunk frame.
            if approach is not None or (chunk is not None and chunk_index == 0):
                execute_preflight(target_chunk(hold), current, now_ros_ns=node.get_clock().now().nanoseconds,
                                  now_monotonic_ns=time.monotonic_ns())
            else:
                current.validate_freshness(now_ros_ns=node.get_clock().now().nanoseconds,
                                          now_monotonic_ns=time.monotonic_ns())
            stamp = node.get_clock().now().nanoseconds
            arm, left, right = sequenced_messages(hold, session_id=session,
                                                 sequence=sequence, timestamp_ns=stamp)
            for publisher, message in zip(publishers, (arm,left,right), strict=True):
                if publisher is not None:publisher.publish(message)
            sequence += 1
            next_tick += DT_NS
            if not motion_started and sequence%10==0:
                from pathlib import Path
                result={'status':state['phase'],'supervisor':state,'source_frames':sequence,
                    'session_id':session,'hardware_send':0,'hand_permits':False,
                    'arm_publishers':0,'mode':'persistent_acquisition_handoff'}
                path=Path('/tmp/sharpa_acquisition_only_status.json');temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(result,indent=2));temporary.replace(path)
            if settled:
                completed_target = approach.target.tolist()
                approach, prepared = None, True
                motion_reply(completion('prepare', maximum_pose_error_rad=error,
                    policy_first_target=completed_target, initial_delta_rad=initial_delta,
                    unit='rad', representation='absolute_joint_position'))
            elif chunk is not None:
                chunk_index += 1
                if chunk_index == 50:
                    chunk = None
                    motion_reply(completion('chunk', frames=50))
            # The same session keeps sending the last absolute target at 20Hz
            # during observation/inference; this retains the existing watchdog.
    except Exception as error:
        result={"status": "FAIL", "reason": str(error), "published": sequence > 0,
                "sequence_end": sequence-1}
        if motion_reply is not None:
            try:motion_reply(result)
            except OSError:pass
        reply(result)
        if args.acquisition_only:
            from pathlib import Path
            Path('/tmp/sharpa_acquisition_only_status.json').write_text(json.dumps(result))
        raise SystemExit(1)
    finally:
        feedback.close()
        node.destroy_node()
        rclpy.shutdown()
        if server is not None:server.close()


if __name__ == '__main__':
    main()
