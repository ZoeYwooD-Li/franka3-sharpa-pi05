"""Start/reuse the one background acquisition/handoff owner; never ARM."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

from .handoff import HandoffClient


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--state-dir',type=Path,default=Path.home()/'.local/state/fr3_sharpa_runtime')
    args=parser.parse_args();state=args.state_dir;state.mkdir(parents=True,exist_ok=True)
    control=state/'motion_control.sock';pidfile=state/'transport_process.json'
    with (state/'transport_start.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if control.exists():
            try:
                client=HandoffClient(control)
            except ConnectionRefusedError:
                # A recorded live owner must never lose its socket on cleanup.
                if pidfile.exists() and Path(f"/proc/{json.loads(pidfile.read_text())['pid']}").exists():
                    raise RuntimeError('Recorded transport still alive; not removing its socket')
                control.unlink()
            else:
                try:print(json.dumps(client.request({'operation':'inspect'})))
                finally:client.close()
                return
        repo=Path(__file__).resolve().parents[2]
        gello=Path(os.environ.get('SHARPA_GELLO_REPO','/home/user/gello_upper_body_teleop'))
        overlay=Path(os.environ.get('SHARPA_ROS_MESSAGES_OVERLAY','/tmp/sharpa_ros_msg_build/install'))
        if not (overlay/'setup.bash').is_file():raise FileNotFoundError('ROS ArmCommand overlay missing')
        cmd=['python','-m','deploy.fr3_sharpa_runtime.guarded_motion_ros','--acquisition-only','--control-socket',str(control)]
        inner=(f"source {shlex.quote(str(overlay/'setup.bash'))}; "
               f"export PYTHONPATH={shlex.quote(str(repo))}:{shlex.quote(str(gello))}:{shlex.quote(str(gello/'ros_ws/src/teleop_core'))}:\"${{PYTHONPATH:-}}\"; exec {shlex.join(cmd)}")
        with (state/'transport.log').open('ab') as log:
            p=subprocess.Popen(['pixi','run','-e','ros2','bash','-lc',inner],cwd=gello/'portable_deps/litchi_hardware',
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pidfile.write_text(json.dumps({'pid':p.pid,'control_socket':str(control)}))
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            if p.poll() is not None:raise RuntimeError('Transport startup failed; inspect transport.log')
            if control.exists():
                client=HandoffClient(control)
                try:
                    report=client.request({'operation':'inspect'})
                    if report['phase']=='READY':
                        print(json.dumps(report));return
                finally:client.close()
            time.sleep(.05)
        raise TimeoutError('Transport control socket startup timeout')


if __name__=='__main__':main()
