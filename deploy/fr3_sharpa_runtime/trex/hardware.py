"""ROS Safety Gateway, pi05 hand commands and bounded connection cleanup."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time
import uuid

import numpy as np

from ..protocol import FR3_COMMAND_NAMES
from ..ros_observation import stamp_ns
from ..ros_transport import validate_gateway_admission, validate_gateway_graph
from .execution_core import ARM
from .live_sensors import SensorStore
from .acquisition import NativeSampler
from .pi05_sdk import DirectSdkConfig, SharpaSdkDirectExecutor
from .pi05_shutdown import disable_and_confirm, protect_disable_confirmation


def takeover_home(report_path, expected_sha, *, timeout=135):
    path=Path(report_path)
    report=json.loads(path.read_text())
    pid=int(report['pid'])
    if report.get('status')!='HOLDING_HOME' or report.get('home_sha256')!=expected_sha:
        raise ValueError('Home takeover record does not match the requested named Home')
    if not 0 <= time.time_ns()-int(report['updated_unix_ns']) < 2_000_000_000:
        raise ValueError('Home holder record is stale/future-dated')
    command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if not any(Path(os.fsdecode(p)).name=='trex_hand_home.py' for p in command if p) or b'--hold' not in command:
        raise ValueError('Refusing to signal a PID that is not the verified Home holder')
    os.kill(pid,signal.SIGINT)
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        current=json.loads(path.read_text())
        shutdown=current.get('shutdown')
        if shutdown:
            if shutdown.get('status')!='PASS':
                raise RuntimeError('Old SDK owner could not confirm disable/disconnect')
            print('Home SDK owner stopped; both hands confirmed disabled/disconnected',flush=True)
            return
        time.sleep(.1)
    raise TimeoutError('Home holder shutdown not confirmed; no new SDK connection opened')


def ensure_sdk_free():
    # Do not silently steal an SDK session from a GUI or another deployment.
    sockets=[]
    try:
        for port in (54321,50011,50021):
            sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            sockets.append(sock)
            sock.bind(('0.0.0.0',port))
    except OSError as error:
        raise RuntimeError('Sharpa SDK ports occupied; use the verified Home takeover or stop its owner') from error
    finally:
        for sock in sockets: sock.close()


class Hardware:
    def __init__(self):
        import rclpy
        from sensor_msgs.msg import Image,JointState
        from teleop_interfaces.msg import ArmCommand,ArmCommandStatus
        from rclpy.qos import QoSProfile,ReliabilityPolicy,qos_profile_sensor_data
        self.rclpy=rclpy
        self.store=SensorStore()
        self.owned={}
        self.hand_executor=None
        self.connection_attempts=[]
        self.sampler=None
        self.arm_pub=None
        self.spinner=None
        self.ros_executor=None
        self.sequence=0
        self.first_send_ns=None
        self.last_ack_ns=None
        self.last_ack_sequence=-1
        self.ack_error=None
        self.session='trex-'+uuid.uuid4().hex
        self.status_lock=threading.Lock()
        rclpy.init()
        self.node=rclpy.create_node('trex_fr3_sharpa_execute')
        self.subscriptions=[]
        self.ArmCommand=ArmCommand
        for side in ('left','right'):
            self.subscriptions.append(self.node.create_subscription(JointState,f'/{side}/franka/joint_states',
                lambda msg,s=side:self.store.arm(s,msg),qos_profile_sensor_data))
        for camera in ('cam0','cam1','cam2'):
            self.subscriptions.append(self.node.create_subscription(Image,f'/{camera}/color/image_raw',
                lambda msg,c=camera:self.store.camera(c,msg),QoSProfile(depth=5,reliability=ReliabilityPolicy.RELIABLE)))
        self.subscriptions.append(self.node.create_subscription(ArmCommandStatus,'/teleop/arm_command_status',self.ack,10))

    def check_admission(self):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            self.rclpy.spin_once(self.node,timeout_sec=.05)
            if self.node.get_node_names().count('teleop_safety_gateway')==1: break
        validate_gateway_admission(self.node)
        pubs=self.node.get_publishers_info_by_topic('/teleop/arm_commands')
        if pubs: raise RuntimeError('Another arm command publisher is active; refusing concurrent control')
        if any(self.node.get_publishers_info_by_topic(f'/sharpa/{side}/joint_states') for side in ('left','right')):
            raise RuntimeError('Another hand driver is active; refusing SDK ownership conflict')

    def start_ros(self):
        from rclpy.executors import SingleThreadedExecutor
        self.ros_executor=SingleThreadedExecutor()
        self.ros_executor.add_node(self.node)
        self.spinner=threading.Thread(target=self.ros_executor.spin,daemon=True,name='trex-ros-observations')
        self.spinner.start()

    def check_cameras(self, *, timeout_s=60., stable_s=2.):
        started=time.monotonic()
        deadline=started+timeout_s
        next_progress=started+5
        stable_since=None
        last_error='Waiting for camera streams'
        print('Checking camera source clocks/cadence before opening hands or sending Home',flush=True)
        while time.monotonic()<deadline:
            try:
                self.store.camera_ready()
                if stable_since is None: stable_since=time.monotonic()
                if time.monotonic()-stable_since>=stable_s:
                    return {'status':'PASS','stable_s':stable_s,'elapsed_s':time.monotonic()-started}
            except ValueError as error:
                stable_since=None
                last_error=str(error)
            if time.monotonic()>=next_progress:
                print(f'Camera preflight waiting ({time.monotonic()-started:.1f}s/{timeout_s:g}s): {last_error}',flush=True)
                next_progress=time.monotonic()+5
            time.sleep(.02)
        raise TimeoutError('Camera preflight failed; hands unopened, no motion sent: '+last_error)

    def connect_hands(self):
        from litchi_hardware.core.types import HandSide
        from litchi_hardware.hardware.sharpa import create_hand
        from litchi_hardware.hardware.sharpa.config import SharpaConfig
        ensure_sdk_free()
        def factory(side,serial):
            hand=create_hand(SharpaConfig(name='trex-'+side,side=HandSide(side),serial=serial))
            self.owned[side]=hand
            return hand
        self._connect_native(factory)
        self.sampler=NativeSampler(self.store,self.hand_executor)
        self.arm_pub=self.node.create_publisher(self.ArmCommand,'/teleop/arm_commands',10)

    def _connect_native(self,factory):
        for attempt in range(1,4):
            self.hand_executor=SharpaSdkDirectExecutor(DirectSdkConfig(
                left_serial='C5549534C555',right_serial='CC559039CC54'),hand_factory=factory)
            try:
                self.hand_executor.connect()
            except Exception as error:
                # The HAL reads these settings before _ensure_enabled(). A
                # failed read with all native handles already closed can be
                # retried before any target publication. Never retry an enable
                # or start failure, or while another hand may remain enabled.
                message=str(error)
                getter_failure=any(f'Sharpa {name} failed:' in message for name in
                    ('get_speed_coeff','get_current_coeff','get_control_source','get_control_mode'))
                inactive=(not self.hand_executor.connected_sides and
                          all(hasattr(hand,'_wave') and hand._wave is None
                              and not hand.is_connected for hand in self.owned.values()))
                retry=getter_failure and inactive and attempt<3
                self.connection_attempts.append({'attempt':attempt,'status':'FAIL',
                                                'error':message,'retry':retry})
                if not retry: raise
                print(f'[sharpa-connect] 初始化参数读取失败，1秒后重试 ({attempt}/3): {message}',flush=True)
                self.hand_executor.left=self.hand_executor.right=None
                self.owned.clear()
                time.sleep(1.)
            else:
                self.connection_attempts.append({'attempt':attempt,'status':'CONNECTED'})
                break

    def poll(self):
        # All native reads have one dedicated owner. The command loop only
        # checks it and consumes buffered feedback; it never competes for reads.
        self.sampler.check()

    def ack(self,msg):
        if msg.source!='openpi' or msg.session_id!=self.session: return
        age=self.node.get_clock().now().nanoseconds-stamp_ns(msg)
        if not 0<=age<=200_000_000: return
        with self.status_lock:
            if msg.sequence>=self.sequence or msg.sequence<=self.last_ack_sequence:
                return
            if msg.faults or set(msg.accepted_sides)!={'left','right'}:
                self.ack_error=f'Safety Gateway rejected command {msg.sequence}: {list(msg.faults)} / {list(msg.accepted_sides)}'
            self.last_ack_ns=time.monotonic_ns()
            self.last_ack_sequence=int(msg.sequence)

    def check_gateway(self):
        validate_gateway_graph(self.node)
        with self.status_lock:
            if self.ack_error: raise RuntimeError(self.ack_error)
            now=time.monotonic_ns()
            if self.first_send_ns is not None:
                reference=self.last_ack_ns if self.last_ack_ns is not None else self.first_send_ns
                if now-reference>200_000_000:
                    raise RuntimeError('Safety Gateway acknowledgement stale/missing')
                if self.sequence-1-self.last_ack_sequence>6:
                    raise RuntimeError('Safety Gateway acknowledgement sequence fell behind')

    def send(self,command):
        self.check_gateway()
        msg=self.ArmCommand()
        msg.header.stamp=self.node.get_clock().now().to_msg()
        msg.source='openpi'; msg.session_id=self.session; msg.sequence=self.sequence
        msg.active_sides=['left','right']; msg.joint_names=list(FR3_COMMAND_NAMES)
        msg.positions=[float(v) for v in command[ARM]]
        # Increment before publish so a synchronous/fast acknowledgement is valid.
        self.sequence+=1
        self.hand_executor.send(command[7:29],command[36:58])
        self.arm_pub.publish(msg)
        if self.first_send_ns is None: self.first_send_ns=time.monotonic_ns()

    def close(self):
        shutdown={'status':'NOT_CONNECTED'}
        sampler_error=None
        if self.sampler is not None:
            try: self.sampler.close()
            except BaseException as error: sampler_error=str(error)
        try:
            if self.owned:
                with protect_disable_confirmation():
                    shutdown=disable_and_confirm({s:self.owned.get(s) for s in ('left','right')})
        finally:
            if self.ros_executor is not None:
                self.ros_executor.shutdown(timeout_sec=2)
            if self.spinner is not None: self.spinner.join(timeout=2)
            self.node.destroy_node()
            if self.rclpy.ok(): self.rclpy.shutdown()
        shutdown['tactile_acquisition']=self.store.tactile_diagnostics()
        shutdown['connection_attempts']=self.connection_attempts
        if self.sampler is not None:
            shutdown['tactile_acquisition']['max_poll_gap_ms']=self.sampler.max_poll_gap_ms
        if sampler_error:
            shutdown.update(status='FAIL',acquisition_cleanup_error=sampler_error)
        return shutdown
