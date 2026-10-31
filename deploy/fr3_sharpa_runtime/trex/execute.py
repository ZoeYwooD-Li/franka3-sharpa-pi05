"""Single-host dual-FR3/Sharpa T-Rex execution. Hardware requires two flags.

--check-server, --plan and --offline-replay never import ROS or open hand devices.
"""
import argparse
from collections import deque
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from .contract import validate_checkpoint
from .execution_core import (ARM,HAND,ChunkTimeline,Deadline,acquisition_check,
                             bounded_step,finite_state,frame_check,validate_positions,installation_timing,
                             ObservationUnavailable,prepare_step,joint_diagnostics)
from .execute_progress import PrepareMonitor,ProgressReporter

PI05=Path('/home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test')


def limits_and_home(args):
    from teleop_core.safety import LOWER_LIMITS,UPPER_LIMITS
    from litchi_hardware.hardware.sharpa.schema import SHARPA_WAVE_JOINT_SCHEMA
    from ..protocol import SHARPA_JOINT_NAMES
    from ..home_pose import load_home,home_target
    if tuple(SHARPA_WAVE_JOINT_SCHEMA.names)!=tuple(SHARPA_JOINT_NAMES):
        raise ValueError('Native SDK hand order differs from T-Rex deployment')
    limits=np.asarray(SHARPA_WAVE_JOINT_SCHEMA.limits,dtype=float)
    lower=np.r_[LOWER_LIMITS[:7],limits[:,0],LOWER_LIMITS[7:],limits[:,0]]
    upper=np.r_[UPPER_LIMITS[:7],limits[:,1],UPPER_LIMITS[7:],limits[:,1]]
    poses,metadata=load_home(args.home_poses,args.home_task,hand_path=args.hand_home)
    target=home_target(poses,np.zeros(58)).astype(float)
    validate_positions(target,lower,upper)
    return lower,upper,target,metadata


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true')
    mode.add_argument('--check-server',action='store_true',help='Check model hello only; no ROS, SDK, claim or inference')
    mode.add_argument('--offline-replay',type=Path)
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--home-only',action='store_true',help='Move both arms and hands to named Home and exit; no model service required')
    p.add_argument('--enable-hardware',action='store_true')
    p.add_argument('--prompt',default='Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.')
    p.add_argument('--port',type=int,default=5555)
    p.add_argument('--monitor-port',type=int,default=0,help='Read-only local camera/tactile preview HTTP port; 0 disables')
    p.add_argument('--monitor-record',action='store_true',help='Save 4 Hz camera/tactile previews for replay (maximum 2 GiB)')
    p.add_argument('--num-chunks',type=int,default=1,help='0 means continuous execution')
    p.add_argument('--speed',choices=('slow','medium','fast'),default='slow')
    p.add_argument('--command-speed-rad-s',type=float,default=.2,help='Maximum emitted joint target speed; capped at 0.2')
    p.add_argument('--fast-every',type=int,default=4,help='30Hz ticks between tactile refinements')
    p.add_argument('--home',action='store_true',help='Move to the same named pi05 arm/hand Home first')
    p.add_argument('--home-poses',type=Path,default=PI05/'.deployment/operator_poses.yaml')
    p.add_argument('--hand-home',type=Path,default=PI05/'.deployment/hand_home.json')
    p.add_argument('--home-task',default='powder_weighing')
    p.add_argument('--approach-first-target',action='store_true',help='If first-target acquisition fails, approach at 0.03rad/s, discard it and reobserve (max 3 attempts)')
    p.add_argument('--hold-right-ring-pinky-aa',action='store_true',help='After optional Home, hold right ring/pinky MCP_AA at measured angles; preserve real feedback and all guards')
    p.add_argument('--tactile-host-gap-ms',type=float,default=75.,help='Host tactile history gap limit, 75..100 ms; SDK source limit remains 75 ms')
    p.add_argument('--tactile-recovery-s',type=float,default=0.,help='Opt-in bounded hold/reobserve on tactile history reset, 0 disables, max 3s')
    p.add_argument('--tactile-max-recoveries',type=int,default=3,help='Maximum recovery starts per 60s, 1..20; total recovery hold time remains limited to 10s per 60s')
    p.add_argument('--prediction-clip-rad',type=float,default=0.,help='Opt-in hand-only native range correction, maximum 0.05 rad; arms and large violations remain rejected')
    p.add_argument('--takeover-home',type=Path,help='Explicitly stop only the verified standalone Home holder after disable/readback')
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[3]/'trex-local-records')
    return p


def check_args(args,p):
    motion=args.execute or args.home_only
    if args.monitor_record and not motion:p.error('--monitor-record requires motion mode')
    if args.monitor_port and (not motion or not 1024<=args.monitor_port<=65535 or args.monitor_port==args.port):
        p.error('--monitor-port requires motion mode and a distinct port in 1024..65535')
    if not np.isfinite(args.prediction_clip_rad) or not 0<=args.prediction_clip_rad<=.05:
        p.error('--prediction-clip-rad must be finite and in 0..0.05')
    if args.prediction_clip_rad and not args.execute:p.error('Prediction clipping requires --execute')
    if not 1 <= args.tactile_max_recoveries <= 20:p.error('--tactile-max-recoveries must be in 1..20')
    if args.tactile_max_recoveries!=3 and not args.execute:p.error('Changing recovery count requires --execute')
    if not np.isfinite(args.tactile_recovery_s) or not 0 <= args.tactile_recovery_s <= 3:
        p.error('--tactile-recovery-s must be finite and in 0..3')
    if args.tactile_recovery_s and not args.execute:
        p.error('Tactile recovery requires --execute')
    if not np.isfinite(args.tactile_host_gap_ms) or not 75 <= args.tactile_host_gap_ms <= 100:
        p.error('--tactile-host-gap-ms must be finite and in 75..100')
    if args.tactile_host_gap_ms!=75 and not args.execute:
        p.error('Changing tactile timing requires --execute')
    if args.hold_right_ring_pinky_aa and not args.execute:
        p.error('--hold-right-ring-pinky-aa requires --execute; Home-only remains the full named Home')
    if not args.home_only and args.checkpoint is None:
        p.error('--checkpoint is required except with --home-only')
    if args.home_only:
        args.home=True
        if args.approach_first_target:
            p.error('--approach-first-target cannot be used with --home-only')
    if not args.prompt.strip(): p.error('Task prompt must not be empty')
    if args.enable_hardware!=motion:
        p.error('Hardware requires --execute or --home-only together with --enable-hardware')
    if args.takeover_home and not motion:
        p.error('--takeover-home is only valid for explicit execution')
    if (not 1024<=args.port<=65535 or args.num_chunks<0 or not 1<=args.fast_every<=16
            or not np.isfinite(args.command_speed_rad_s) or not 0<args.command_speed_rad_s<=.2):
        p.error('Invalid port/chunk count/cadence/speed bounds')


def offline(args,lower,upper):
    from .shadow import load_observation
    from .client import TrexClient
    from .actions import TrexActionDecoder
    obs,metadata=load_observation(args.offline_replay)
    with TrexClient(f'tcp://127.0.0.1:{args.port}',timeout_ms=5000) as client:
        client.hello(args.checkpoint)
        client.claim()
        try:
            cid,raw=client.slow_and_fast(obs['cam0'],obs['cam1'],obs['cam2'],args.prompt,obs['tactile_f6'],obs['tactile_deform'])
            decoded=TrexActionDecoder().decode(raw,obs['state'])
            validate_positions(decoded,lower,upper)
            fast_id,fast_raw=client.fast(obs['tactile_f6'],obs['tactile_deform'],cid)
            refined=TrexActionDecoder().decode(fast_raw,obs['state'])
            validate_positions(refined,lower,upper)
        finally: client.release()
    return {'mode':'offline_replay','published':False,'execute_ready':False,
            'chunk_id':cid,'fast_chunk_id':fast_id,'action_shape':list(raw.shape),
            'decoded_shape':list(decoded.shape),'input_metadata':metadata,
            'note':'Recorded tactile reused only to test protocol. This is not a live refinement or hardware qualification.'}


def run(args,lower,upper,home_target,metadata,directory):
    # Import ROS/native control only beyond the explicit hardware gate.
    from .hardware import Hardware,takeover_home
    hardware=None; worker=None; result={'status':'STARTING','published':False,'completed_chunks':0,
                                     'policy_frames_sent':0,'prepare_frames_sent':0,'home_frames_sent':0,
                                     'mode':'home_only' if args.home_only else 'execute',
                                     'prompt':None if args.home_only else args.prompt}
    trace=None; reporter=None; prepare_records=[]; policy_hold=None; monitor=None
    try:
        hardware=Hardware()
        if args.tactile_host_gap_ms!=75:
            hardware.store.native.configure_host_gap(args.tactile_host_gap_ms)
        result['tactile_timing_limits']={'source_gap_ms':75.,'host_gap_ms':args.tactile_host_gap_ms,
                                       'latest_age_ms':200.,'finger_span_ms':100.}
        result['tactile_recovery_timeout_s']=args.tactile_recovery_s
        result['tactile_recovery_policy']={'max_starts':args.tactile_max_recoveries,'window_s':60.,'total_hold_budget_s':10.}
        hardware.check_admission()
        if not args.home_only:
            from .inference_worker import InferenceWorker
            options={}
            if args.prediction_clip_rad:options['prediction_clip_rad']=args.prediction_clip_rad
            if args.hold_right_ring_pinky_aa:options['held_hand_indices']=(50,55)
            worker=InferenceWorker(args.checkpoint,f'tcp://127.0.0.1:{args.port}',lower,upper,**options)
            deadline=time.monotonic()+6
            while not worker.ready.wait(timeout=.05):
                worker.poll()
                if time.monotonic()>deadline: raise TimeoutError('Model session not ready; hardware remains unopened')
        if args.takeover_home:
            takeover_home(args.takeover_home,metadata['hand_home_sha256'])
        hardware.start_ros()
        if getattr(args,'monitor_port',0) or getattr(args,'monitor_record',False):
            from .monitor import LiveMonitor
            monitor=LiveMonitor(hardware.store,directory,args.monitor_port,record=args.monitor_record)
            monitor.start()
            result['monitor']={'url':monitor.url,'requested_port':args.monitor_port,'refresh_hz':4,'mode':'latest_read_only'}
            print('Read-only sensor monitor: '+monitor.url,flush=True)
            console=getattr(args,'console',None)
            if console is not None:console.write('相机／触觉预览：'+monitor.url)
        if not args.home_only:
            result['last_phase']='CAMERA_PREFLIGHT'
            result['camera_preflight']=hardware.check_cameras()
        hardware.connect_hands()
        # Native receive startup only; no fabricated episode feedback is accepted.
        bootstrap=time.monotonic()+3
        while True:
            hardware.poll()
            try: measured=hardware.store.feedback(); break
            except ValueError:
                if time.monotonic()>=bootstrap: raise
                time.sleep(.01)
        command=finite_state(measured)
        validate_positions(command,lower,upper)
        phase='HOME' if args.home else 'WAIT_SENSORS'
        phase_started=time.monotonic()
        stable_since=None
        prepare_token=None
        last_prepare_feedback=None
        prepare_monitor=None
        prepare_timeout_s=120.
        prepare_target=None
        preparations=0
        completed=0
        installed=0
        recovery_starts=deque()
        recovery_durations=deque()
        recovery_started_ns=None
        timeline=None
        tick=0
        stale_results=0
        clock=Deadline()
        trace=(directory/'frames.jsonl').open('x',buffering=64*1024)
        reporter=ProgressReporter(directory,console=getattr(args,'console',None))
        reported_phase=None; next_progress=0.
        result.update(status='RUNNING',home_metadata=metadata)
        while hardware.rclpy.ok():
            result['last_phase']=phase
            clock.wait()
            hardware.poll()
            measured=hardware.store.feedback()
            prediction=worker.poll() if worker is not None else None
            if isinstance(prediction,ObservationUnavailable):
                result['observation_retries']=result.get('observation_retries',0)+1
                result['last_observation_retry']=str(prediction)
                prediction=None
                if phase=='PLAN' and time.monotonic()-phase_started>3:
                    raise TimeoutError('No valid composite observation within 3s: '+result['last_observation_retry'])
            if prediction is not None and policy_hold is not None:
                prediction=policy_hold.apply(prediction)
            # Match pi05 prepare: joint feedback/gateway checks stay active,
            # while model-input readiness is required at inference/execution.
            if phase=='RUN':
                # Fast execution consumes live tactile and cached slow context;
                # RGB age is checked when constructing every slow observation.
                from .native_tactile import TactileObservationPending
                try:
                    hardware.store.fresh(require_cameras=False)
                except TactileObservationPending as error:
                    if not args.tactile_recovery_s:raise
                    now=time.monotonic()
                    while recovery_starts and now-recovery_starts[0]>60:recovery_starts.popleft()
                    if len(recovery_starts)>=args.tactile_max_recoveries:
                        raise RuntimeError(f'More than {args.tactile_max_recoveries} tactile interruptions in 60s; stopped') from error
                    recovery_starts.append(now)
                    recovery_started_ns=time.monotonic_ns()
                    result.setdefault('tactile_recoveries',[]).append({
                        'status':'WAITING','started_mono_ns':recovery_started_ns,
                        'interrupted_chunk_id':timeline.chunk_id,'interrupted_row':timeline.row,
                        'prediction_file':f'chunk-{installed-1:04d}.npz','reason':str(error)})
                    result['interrupted_chunks']=result.get('interrupted_chunks',0)+1
                    timeline=None;prediction=None
                    phase='RECOVER_SENSORS';phase_started=now
            if phase=='PLAN' and time.monotonic()-phase_started>5:
                raise TimeoutError('Planning did not produce an installable result within 5s')
            target=command
            command_origin='hold'
            ran_row=False
            speed=args.command_speed_rad_s
            if phase in ('HOME','APPROACH'):
                command_origin='home' if phase=='HOME' else 'policy_prepare'
                speed=.03
                target=home_target if phase=='HOME' else prepare_target
                if prepare_token!=(phase,phase_started):
                    prepare_token=(phase,phase_started)
                    last_prepare_feedback=measured.copy()
                    prepare_monitor=PrepareMonitor()
                    prepare_timeout_s=max(120.,float(np.max(np.abs(target-command)))/.03*2+.5+5)
                prepare_now=time.monotonic()
                error=float(np.max(np.abs(measured-target)))
                stable=float(np.max(np.abs(measured-last_prepare_feedback)))<=.005
                last_prepare_feedback=measured.copy()
                command_at_target=bool(np.max(np.abs(command-target))<1e-8)
                out_of_tolerance=np.flatnonzero(np.abs(measured-target)>.02)
                stalled=prepare_monitor.update(prepare_now,command,measured,target)
                try:
                    acquisition_check(target,measured)
                    acquisition_ready=True
                except ValueError:
                    acquisition_ready=False
                # HOME certifies arrival. Policy APPROACH only establishes a
                # safe handoff to a NEW observation/prediction. A hand target
                # inside the existing acquisition bound need not settle like
                # a fixed Home pose before the policy can close the loop.
                blocking_stalled=(stalled if phase=='HOME' else
                    [item for item in stalled
                     if item['target_error_rad']>(.05 if item['index'] in ARM else .15)])
                worst=joint_diagnostics(command,measured,target,out_of_tolerance)[:5]
                for item in worst:
                    item.update(lower_rad=float(lower[item['index']]),upper_rad=float(upper[item['index']]))
                result['prepare_progress']={'phase':phase,'elapsed_s':prepare_now-phase_started,
                    'max_error_rad':error,'stable':stable,'at_target':error<=.02,
                    'command_at_target':command_at_target,'timeout_s':prepare_timeout_s,
                    'unsettled_joints':len(out_of_tolerance),'worst_joints':worst,
                    'stalled_joints':stalled,'blocking_stalled_joints':blocking_stalled,
                    'acquisition_ready':acquisition_ready,
                    'completion_criterion':'home_settled' if phase=='HOME' else 'policy_acquisition_bounds'}
                if blocking_stalled:
                    raise RuntimeError('Prepare feedback stalled for >=2s; physical reachability is unverified: '
                                       +json.dumps(blocking_stalled))
                ready=(command_at_target and error<=.02) if phase=='HOME' else acquisition_ready
                if ready and stable:
                    stable_since=time.monotonic() if stable_since is None else stable_since
                    if time.monotonic()-stable_since>=.5:
                        if args.home_only:
                            result.update(home_completed=True,last_phase='HOME',completion='home_settled')
                            break
                        result['prepare_completion']=dict(result['prepare_progress'])
                        # A prepared prediction is discarded. Acquire a complete
                        # new observation after either Home or policy approach.
                        phase='WAIT_SENSORS'
                        phase_started=time.monotonic(); stable_since=None
                else: stable_since=None
                if time.monotonic()-phase_started>prepare_timeout_s:
                    raise TimeoutError('Home/first-target approach failed to settle: '
                                       +json.dumps(result['prepare_progress']))
            elif phase=='RECOVER_SENSORS':
                # Keep the last sent pose, validated against real arm/hand
                # feedback every tick. Never consume the old action timeline.
                command_origin='tactile_recovery_hold'
                prediction=None  # Drain any result from the interrupted job.
                now=time.monotonic()
                while recovery_durations and now-recovery_durations[0][0]>60:recovery_durations.popleft()
                if sum(duration for _,duration in recovery_durations)+now-phase_started>=10:
                    raise RuntimeError('Tactile recovery hold budget exceeded 10s in 60s; stopped')
                try:
                    observation=hardware.store.snapshot(args.prompt,encode_images=False)
                    if observation.get('history_started_mono_ns',0)<recovery_started_ns:
                        raise ValueError('Waiting for 16 original post-interruption frames from every finger')
                    if not worker.busy:
                        recovery_durations.append((time.monotonic(),time.monotonic()-phase_started))
                        result['tactile_recoveries'][-1].update(status='REOBSERVED',
                            elapsed_s=time.monotonic()-phase_started)
                        phase='PLAN';phase_started=time.monotonic()
                        preparations=0
                    else:
                        result['last_sensor_wait']='Waiting for interrupted inference job to drain'
                except ValueError as error:
                    result['last_sensor_wait']=str(error)
                if phase=='RECOVER_SENSORS' and time.monotonic()-phase_started>=args.tactile_recovery_s:
                    raise TimeoutError('Tactile recovery timed out; no new chunk installed: '+result.get('last_sensor_wait','inference still busy'))
            elif phase=='WAIT_SENSORS':
                try:
                    hardware.store.snapshot(args.prompt,encode_images=False)
                    if args.hold_right_ring_pinky_aa and policy_hold is None:
                        from .joint_hold import PolicyJointHold
                        policy_hold=PolicyJointHold(measured)
                        result['policy_joint_hold']=policy_hold.metadata()
                        result['degraded_execution']=True
                        print('Policy joint hold: '+json.dumps(result['policy_joint_hold']),flush=True)
                    phase='PLAN'; phase_started=time.monotonic()
                except ValueError as error:
                    result['last_sensor_wait']=str(error)
                    if time.monotonic()-phase_started>20: raise
            elif phase=='PLAN':
                if prediction is not None:
                    if prediction.kind!='slow':
                        prediction=None  # A completed old fast job never installs a new chunk.
                    else:
                        fresh,timing=installation_timing(prediction,time.monotonic_ns())
                        result['last_prediction_timing']=timing
                        if not fresh:
                            stale_results+=1
                            print('Expired planning result: '+json.dumps(timing),flush=True)
                            if stale_results>=3:
                                raise RuntimeError('Slow/fast planning repeatedly expired; no model command installed: '+json.dumps(timing))
                            prediction=None
                    if prediction is not None:
                        stale_results=0
                        if np.max(np.abs(measured[ARM]-prediction.base_state[ARM]))>.02:
                            raise ValueError('Arms moved during slow inference; fixed FK base no longer valid')
                        try: acquisition_check(prediction.joints[0],measured)
                        except ValueError:
                            if not args.approach_first_target or preparations>=3: raise
                            prepare_target=prediction.joints[0].copy()
                            prepare_records.append(prediction)
                            result['prepare_prediction_files']=[f'prepare-{i:04d}.npz'
                                                               for i in range(1,len(prepare_records)+1)]
                            result['prepare_target']=prepare_target.tolist()
                            result['prepare_chunk_id']=prediction.chunk_id
                            phase='APPROACH'; phase_started=time.monotonic(); preparations+=1
                            # Prepared predictions are discarded, never executed after a long bridge.
                        else:
                            timeline=ChunkTimeline(prediction,speed_scale={'slow':.1,'medium':.5,'fast':1.}[args.speed])
                            preparations=0
                            np.savez_compressed(directory/f'chunk-{installed:04d}.npz',raw=prediction.raw,
                                                decoded=prediction.joints,base=prediction.base_state,
                                                decoded_model=(prediction.original_joints if prediction.original_joints is not None else prediction.joints),
                                                context_mono_ns=prediction.context_mono_ns,
                                                tactile_mono_ns=prediction.sampled_mono_ns)
                            installed+=1;result['installed_chunks']=installed
                            phase='RUN'; phase_started=time.monotonic()
                if phase=='PLAN' and not worker.busy:
                    worker.submit('slow',lambda:hardware.store.snapshot(args.prompt),measured,time.monotonic_ns(),
                                  refresh=lambda:hardware.store.snapshot(args.prompt,fast=True))
            elif phase=='RUN':
                command_origin='policy'
                if prediction is not None:
                    if prediction.kind!='fast': raise ValueError('Unexpected slow result in active chunk')
                    timeline.refine(prediction,time.monotonic_ns())
                target=timeline.target()
                ran_row=True
                if tick%args.fast_every==0 and not worker.busy:
                    worker.submit('fast',lambda:hardware.store.snapshot(args.prompt,fast=True),
                                  timeline.base_state,timeline.base_mono_ns,timeline.chunk_id)
            candidate=(prepare_step(command,target) if command_origin in ('home','policy_prepare')
                       else bounded_step(command,target,speed))
            result['last_phase']=phase
            try:
                frame_check(candidate,command,measured,lower,upper,speed=speed)
            except ValueError:
                # The rejected candidate is never sent or written as a sent frame.
                result['rejected_frame']={'phase':phase,'tick':tick,'sent':False,
                    'candidate':candidate.tolist(),'previous_command':command.tolist(),
                    'measured':measured.tolist(),'target':target.tolist(),
                    'worst_command_errors':joint_diagnostics(candidate,measured)[:5]}
                raise
            clock.check()  # No inference, IK, serialization or network wait belongs before this deadline.
            hardware.send(candidate)
            if ran_row: result['policy_frames_sent']+=1
            if command_origin=='home': result['home_frames_sent']+=1
            if command_origin=='policy_prepare': result['prepare_frames_sent']+=1
            command=candidate
            result['published']=True
            trace.write(json.dumps({'tick':tick,'phase':phase,'mono_ns':time.monotonic_ns(),
                                    'command_origin':command_origin,
                                    'command':command.tolist(),'measured':measured.tolist(),
                                    'chunk_id':timeline.chunk_id if timeline else None,
                                    'row':timeline.row if timeline else None})+'\n')
            progress_now=time.monotonic()
            if phase!=reported_phase or progress_now>=next_progress:
                progress={'status':'RUNNING','phase':phase,'elapsed_s':progress_now-phase_started,
                          'updated_unix_ns':time.time_ns(),'tick':tick,
                          'home_frames_sent':result['home_frames_sent'],
                          'prepare_frames_sent':result['prepare_frames_sent'],
                          'policy_frames_sent':result['policy_frames_sent']}
                progress.update(chunk_number=completed+1 if phase=='RUN' else None,
                                completed_chunks=completed,total_chunks=args.num_chunks)
                if phase in ('HOME','APPROACH') and result.get('prepare_progress',{}).get('phase')==phase:
                    progress['prepare']=result['prepare_progress']
                if phase in ('WAIT_SENSORS','RECOVER_SENSORS'): progress['waiting_for']=result.get('last_sensor_wait','Composite observation')
                if phase=='PLAN':
                    progress.update(inference_busy=worker.busy,
                                    observation_retries=result.get('observation_retries',0),
                                    last_observation_retry=result.get('last_observation_retry'))
                reporter.publish(progress)
                reported_phase=phase; next_progress=progress_now+1.
            if ran_row:
                timeline.emitted()
                if timeline.done:
                    completed+=1; result['completed_chunks']=completed
                    timeline=None; phase='PLAN'; phase_started=time.monotonic()
                    if args.num_chunks and completed>=args.num_chunks: break
            tick+=1; clock.advance()
        result['status']='PASS' if completed or result.get('home_completed') else 'STOPPED'
        if completed: result['completion']='commands_sent_with_joint_hold' if policy_hold else 'commands_sent'
    except KeyboardInterrupt:
        result.update(status='STOPPED',operator_stop=True)
    except BaseException as error:
        result.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if result.get('tactile_recoveries') and result['tactile_recoveries'][-1]['status']=='WAITING':
            result['tactile_recoveries'][-1].update(status='STOPPED',error=result.get('error','Operator stop'))
        if policy_hold is not None:
            result['policy_joint_hold']['predictions_modified']=policy_hold.predictions_modified
            result['policy_joint_hold']['recent_prediction_audit']=list(policy_hold.audit)
        if hardware is not None:
            try: result['shutdown']=hardware.close()
            except BaseException as error:
                result['shutdown']={'status':'FAIL','error':str(error)}
            if result['shutdown']['status'] not in ('PASS','NOT_CONNECTED'): result['status']='FAIL'
            if monitor is not None:
                try:monitor.close()
                except Exception as error:result['monitor_cleanup_error']=str(error)
                result['monitor'].update(recording_file='sensor-preview.jsonl' if monitor.record else None,
                    recorded_frames=monitor.recorded_frames,recorded_bytes=monitor.recorded_bytes,
                    recording_error=monitor.record_error)
                console=getattr(args,'console',None)
                if console is not None and monitor.record:
                    console.write('观测回放文件：'+str(monitor.recording_path))
                    if monitor.record_error:console.write('观测录制异常（详见报告）：'+monitor.record_error)
            native=getattr(hardware.store,'native',None)
            if native is not None:
                try:
                    np.savez_compressed(directory/'tactile-timing.npz',**native.timing_records())
                    result['tactile_timing_file']='tactile-timing.npz'
                except Exception as error: result['timing_record_error']=str(error)
        if worker is not None:
            try: worker.close()
            except BaseException as error: result['worker_cleanup_error']=str(error)
            ik_failure=getattr(worker,'ik_failure',None)
            if ik_failure is not None:
                result['ik_failure']={'file':'ik-failure.json','solver':ik_failure.get('solver')}
                try:
                    (directory/'ik-failure.json').write_text(json.dumps(ik_failure,indent=2))
                except Exception as error:result['ik_failure']['save_error']=str(error)
            result['prediction_clipping']={'max_correction_rad':args.prediction_clip_rad,
                'corrected_values':getattr(worker,'clipped_values',0),
                'recent_prediction_audit':list(getattr(worker,'clipping_audit',[]))}
            rejected=getattr(worker,'rejected_prediction',None)
            if rejected is not None:
                result['rejected_prediction']={k:v for k,v in rejected.items()
                    if k not in ('raw','decoded','decoded_model','base')}
                try:
                    np.savez_compressed(directory/'rejected-prediction.npz',
                        **{k:rejected[k] for k in ('raw','decoded','decoded_model','base')},
                        metadata_json=np.asarray(json.dumps(result['rejected_prediction'])))
                    result['rejected_prediction']['file']='rejected-prediction.npz'
                except Exception as error:result['rejected_prediction']['save_error']=str(error)
        if trace is not None: trace.close()
        # Persist rejected-for-acquisition predictions only after motion and
        # device cleanup, so compression cannot delay the command deadline.
        for number,prediction in enumerate(prepare_records,1):
            try:
                np.savez_compressed(directory/f'prepare-{number:04d}.npz',
                    raw=prediction.raw,decoded=prediction.joints,base=prediction.base_state,
                    decoded_model=(prediction.original_joints if prediction.original_joints is not None else prediction.joints),
                    context_mono_ns=prediction.context_mono_ns,
                    tactile_mono_ns=prediction.sampled_mono_ns)
            except Exception as error:
                result['prepare_record_error']=str(error)
        if reporter is not None:
            reporter.publish({'status':result['status'],'phase':result.get('last_phase'),
                              'updated_unix_ns':time.time_ns(),'error':result.get('error'),
                              'prepare':result.get('prepare_progress'),
                              'shutdown':result.get('shutdown')})
            reporter.close()
        (directory/'report.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
    if result['status']=='FAIL': raise RuntimeError('Execution/shutdown verification failed')
    return result


def main(argv=None):
    p=parser(); args=p.parse_args(argv); check_args(args,p)
    if args.check_server:
        result=check_server(args.checkpoint,args.port)
        print(json.dumps(result,indent=2))
        if result['status']!='PASS': raise SystemExit(1)
        return
    if (args.plan or args.offline_replay) and not args.home_only:
        contract=validate_checkpoint(args.checkpoint)
    if args.plan or args.offline_replay:
        lower,upper,target,metadata=limits_and_home(args)
    if args.plan:
        print(json.dumps({'mode':'plan','published':False,'hardware_opened':False,
                          'checkpoint':contract['checkpoint'],'action_shape':[16,62],
                          'decoded_shape':[16,58],'home_metadata':metadata,
                          'num_chunks':args.num_chunks,'speed':args.speed},indent=2))
        return
    if args.offline_replay:
        print(json.dumps(offline(args,lower,upper),indent=2)); return
    # Validate shutdown bounds and serialize execution before any SDK/ROS mutation.
    from .pi05_shutdown import DisableSettings
    with open('/tmp/franka-sharpa-trex-execute.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        directory=args.output_dir/f'execute-{stamp}-{os.getpid()}'
        directory.mkdir()
        def stop(*_): raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,stop)
        from .run_output import RunOutput
        # Keep native SDK finalizer output redirected through process exit.
        # Imported callers (including tests) regain their original streams.
        with RunOutput(directory,restore=__name__!='__main__') as console:
            args.console=console
            run_started=False
            try:
                DisableSettings.from_env()
                if not args.home_only:validate_checkpoint(args.checkpoint)
                lower,upper,target,metadata=limits_and_home(args)
                run_started=True
                result=run(args,lower,upper,target,metadata,directory)
            except Exception as error:
                traceback.print_exc()
                report_path=directory/'report.json'
                result=json.loads(report_path.read_text()) if report_path.exists() else {
                    'status':'FAIL','error':f'{type(error).__name__}: {error}','completed_chunks':0}
                if not report_path.exists():report_path.write_text(json.dumps(result,indent=2))
                console.finish(result)
                raise SystemExit(1) from None
            except KeyboardInterrupt:
                report_path=directory/'report.json'
                result=json.loads(report_path.read_text()) if report_path.exists() else {
                    'status':'STOPPED','operator_stop':True,'completed_chunks':0,
                    'shutdown':{'status':'UNCONFIRMED' if run_started else 'NOT_CONNECTED'}}
                if not report_path.exists():report_path.write_text(json.dumps(result,indent=2))
                console.finish(result)
                raise SystemExit(130) from None
            console.finish(result or {'status':'STOPPED'})


def check_server(checkpoint,port=5555):
    """Inspect the server without initializing ROS, claiming it or running the model."""
    from .client import TrexClient
    endpoint=f'tcp://127.0.0.1:{port}'
    result={'mode':'check_server','endpoint':endpoint,'published':False,'hardware_opened':False,
            'claimed':False,'inference_requested':False}
    started=time.monotonic()
    try:
        with TrexClient(endpoint,timeout_ms=3000) as client:
            metadata=client.hello(checkpoint)
            if metadata.get('exclusive_sessions') is not True:
                raise RuntimeError('Restart T-Rex serve with exclusive-session support')
        result.update(status='PASS',server=metadata)
    except (TimeoutError,RuntimeError,ValueError) as error:
        result.update(status='FAIL',error=f'{type(error).__name__}: {error}')
    result['elapsed_s']=time.monotonic()-started
    return result


if __name__=='__main__': main()
