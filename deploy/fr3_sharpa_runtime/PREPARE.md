# Task initialization before policy execution

Existing FR3 services `/reset_to_initial_pose`, `/reset_to_ready`, and
`/reset_to_home/{task}` use `std_srvs/srv/Trigger`. Their implementation publishes
directly to `/left/gello/joint_states` and `/right/gello/joint_states`, so they
cannot be called as the initialization motion backend when Safety Gateway
routing is required. Existing Home tasks are powder weighing, assembly and bean
picking, not this tomato/Sharpa task. No existing complete 58D pose was found.

Sharpa guarded runtime provides `/teleop/sharpa/{left,right}/enable` (`SetBool`)
and `/teleop/sharpa/disengage_all` (`Trigger`). Enable does not authorize motion:
the existing operator GUI must grant permission. No reset/home/prepose action or
service for the full Sharpa task was found. The current guarded excursion,
acquisition, lease and watchdog rules remain unchanged.

`prepare.py` defines a `PrepareBackend` interface. A future movement backend
must use `/teleop/arm_commands` via Safety Gateway, and
`/teleop/sharpa/{left,right}/openpi_target` via the existing guarded runtime.
It must handle existing authorization/session/lease and movement restrictions.
No service wrapper around the bypassing FR3 reset is provided.

`prepare_config.example.json` contains joint names and settling parameters, but
all positions are null. Fill all four groups from a confirmed task pose; no
angles are copied from historical reset configs, training data or model outputs.
Settling parameters concern prepare completion only; they do not change command
limits or execute preflight thresholds.

The available `verify_only` backend reads real FR3 and Sharpa feedback and
verifies that the already reached pose stays within configured pose/stability
tolerances for the configured interval. It cannot move a robot. The
`guarded_motion` backend is an explicit pending interface and fails rather than
publishing commands. The separate policy-first backend described below is implemented; this configured-pose backend remains pending.

From this directory, safe template/config validation entry:

```bash
./runtime.sh --prepare --prepare-config ./prepare_config.example.json
```

The template is expected to FAIL because no target positions are configured.
With a separately confirmed pose and `backend=verify_only`, this command can
verify settling. After success it reacquires observation and requires all seven
sensor timestamps to be after completion; it does not call policy or execute.

Future execute entry (not currently hardware-ready):

```bash
./runtime.sh --execute --enable-hardware --prepare-config /actual/task_pose.json
```

Execute now requires prepare configuration, finishes preparation, and obtains a
new post-prepare observation before the unchanged policy, preflight and command
transport. Missing target configuration, unavailable movement backend, rejected
motion, unstable/stale feedback, or missing new observation stop the sequence
before policy/execute. Merely verifying the prepose does not guarantee that
the policy's first target will satisfy the existing acquisition limits.

No robot initialization or policy execution was performed while implementing
this interface. Movement tests use a fake backend only.

## Capture a manually reached task start pose

After the operator manually positions both arms and both hands, run:

```bash
./capture_prepare_pose.sh --pose-name harvest_tomato_initial_pose \
  --output /home/user/.config/fr3_sharpa_runtime/prepare_config.json
```

This only subscribes to the four existing state topics, reuses their named
58D order and freshness checks, and saves raw topic timestamps, receipt times,
capture time, source topics, ROS domain and RMW implementation. It creates no
publishers or services and invokes no hardware SDK. FR3 state names are mapped
to the frozen fr3v2 command names; hand order stays Wave 22D.

The saved backend is `verify_only`. This does not imply that the operator's
chosen pose is the right task pose, or that policy first-target checks will
pass. No dataset, model output or guessed angle is used to construct it.
Existing files are preserved unless `--overwrite` is explicitly supplied.

Verify the captured pose and reacquire a post-settling observation:

```bash
./runtime.sh --prepare \
  --prepare-config /home/user/.config/fr3_sharpa_runtime/prepare_config.json
```

This verification does not move the robot; it fails if the robot has moved
away from the captured pose or live feedback does not remain stable. The
capture template's settling parameters are retained, not the execute limits.
# Policy-first slow approach

The explicit `--prepare-policy-first` execute path reads live observations and
infers an initial `(50,58)` chunk. Its first absolute 58D target is frozen only
for prepare. The remaining initial chunk is discarded.

The ROS motion worker starts from fresh feedback and approaches this target on
a straight joint-space trajectory at `--prepare-speed-rad-s` (default 0.03
rad/s), at 20 Hz. It checks the existing first-target and feedback freshness
rules on each intermediate frame. Existing joint limits, continuity checks,
FR3 Safety Gateway, Sharpa TargetGate, GUI authorization and watchdogs remain
authoritative. It does not arm hands, alter permits or connect the SDK.

Prepare completes only after measured feedback reaches the target and remains
stable. All seven observation timestamps must then be newer than completion.
The policy is called again, and the original first-target check applies to the
new chunk. A failed new preflight stops; the runtime does not repeatedly chase
new predictions until something passes.

For multiple chunks, the same ROS process, session and continuously increasing
sequence are retained across prepare, inference holds, and whole 50-step chunks.
During inference the sender maintains the last absolute target at 20 Hz through
the same guarded paths. `--num-chunks` is finite and defaults to 1. No half-chunk
replanning, splice, action clipping or change to model output is introduced.

Only after the operator has confirmed the scene, target and existing guarded
authorization, the command syntax is:

```bash
./runtime.sh --execute --enable-hardware --prepare-policy-first \
  --prepare-speed-rad-s 0.03 --num-chunks 1
```

This is a motion command, not a read-only validation command. Do not combine
`--prepare-policy-first` with `--prepare-config`. Existing prepare verification
and all non-execute modes retain their previous behavior.

The native OpenPI-specific rule has been updated by the operator: the validation
acceptance-baseline total-excursion bound does not apply to source=openpi.
The runtime mirrors that exception only. All first-target acquisition, joint
limits, raw continuity, freshness, operator lease and native slew checks remain.
Other sources retain their baseline rule. Software/mock PASS is not motion acceptance.


## Persistent acquisition-to-execute handoff

The hardware stack now starts `start_transport.sh` in the background. It owns
the hand publishers and a private mode-0600 local socket. Acquisition sends
measured-pose samples only while both hand permits are false and hardware_send
is zero. The service never engages hands or generates operator heartbeats.

Terminal 1 remains `./serve.sh`. Terminal 2 first runs:

```bash
./runtime.sh --handoff-verify
```

This attaches to the existing owner without creating a command publisher,
checks the same session and publisher GUIDs, and performs one live policy
inference. It does not arm or execute.

After explicit operator motion authorization, the separate motion command is:

```bash
./runtime.sh --execute --enable-hardware --prepare-policy-first \
  --prepare-speed-rad-s 0.03 --num-chunks 1
```

This connects to the same process over private IPC. Its hand publishers,
session and sequence are retained. Only then may an ArmCommand publisher be
created; movement still waits for the original guarded GUI hand permits and
hold-to-run. Client disconnect during execution stops the stream and lets the
existing watchdog stop the hardware. Read-only verification disconnect does
not stop acquisition. No handoff is attempted by simultaneously starting two
ROS command owners.

The older temporary /tmp acquisition probe cannot be hot-swapped without a
watchdog gap. One supervised migration stops/restarts guarded while permits
are false, then starts this persistent service. Subsequent prepare/chunk
transitions require no process or publisher replacement. Permit expiry remains
authoritative; there is no automatic renewal.
