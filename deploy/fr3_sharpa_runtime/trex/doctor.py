"""Read-only deployment checks. Never connects hands or creates command publishers."""
import argparse
import json
import os
from pathlib import Path
import time

from .execute import check_server


def inspect(hardware, checkpoint, port, duration_s):
    from ..ros_transport import validate_gateway_graph
    result={'status':'STARTING','published':False,'hardware_opened':False,
            'joint_targets_sent':0,'mode':'deployment_doctor',
            'ROS':{key:os.environ.get(key) for key in
                   ('ROS_DOMAIN_ID','ROS_LOCALHOST_ONLY','RMW_IMPLEMENTATION','CYCLONEDDS_URI')}}
    try:
        result['server']=check_server(checkpoint,port)
        if result['server']['status']!='PASS':raise RuntimeError(result['server']['error'])
        hardware.check_admission()
        result['gateway_admission']='PASS'
        hardware.start_ros()
        result['camera_preflight']=hardware.check_cameras()
        camera_stats={key:{'frames':0,'min_source_age_ms':None,'max_source_age_ms':None}
                      for key in hardware.store.cameras}
        markers={}
        started=time.monotonic(); deadline=started+duration_s; checks=0
        next_progress=started+10
        while time.monotonic()<deadline:
            validate_gateway_graph(hardware.node)
            hardware.store.camera_ready()
            with hardware.store.lock:
                wall=time.time_ns(); mono=time.monotonic_ns()
                for side,buf in hardware.store.arms.items():
                    if not buf:raise ValueError(f'Missing {side} arm state')
                    hardware.store.timing([buf[-1]],wall,mono,labels=[side+'_arm'])
                for key,buf in hardware.store.cameras.items():
                    row=buf[-1]
                    if markers.get(key)==row[0]:continue
                    markers[key]=row[0]; stats=camera_stats[key];stats['frames']+=1
                    age=(wall-row[0])/1e6
                    stats['min_source_age_ms']=age if stats['min_source_age_ms'] is None else min(age,stats['min_source_age_ms'])
                    stats['max_source_age_ms']=age if stats['max_source_age_ms'] is None else max(age,stats['max_source_age_ms'])
            checks+=1
            if time.monotonic()>=next_progress:
                print(f'Read-only deployment checks: {checks}, elapsed {time.monotonic()-started:.1f}s',flush=True)
                next_progress=time.monotonic()+10
            time.sleep(.02)
        result.update(status='PASS',duration_s=time.monotonic()-started,checks=checks,cameras=camera_stats,
                      gateway_count=hardware.node.get_node_names().count('teleop_safety_gateway'))
    except Exception as error:
        result.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    finally:
        # Neither inspect nor Hardware.__init__ creates a command publisher.
        if hardware.arm_pub is not None or hardware.hand_executor is not None or hardware.owned:
            result.update(status='FAIL',error='Unexpected command/hardware ownership in read-only check')
        try:
            result['shutdown']=hardware.close()
        except Exception as error:
            result['shutdown']={'status':'FAIL','error':f'{type(error).__name__}: {error}'}
        if result['shutdown'].get('status') not in ('PASS','NOT_CONNECTED'):
            result.update(status='FAIL',cleanup_error=result['shutdown'])
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--port',type=int,default=5555)
    p.add_argument('--duration',type=float,default=30.)
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    if not 0<args.duration<=300 or not 1024<=args.port<=65535:p.error('Invalid duration or port')
    from .hardware import Hardware
    result=inspect(Hardware(),args.checkpoint,args.port,args.duration)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
