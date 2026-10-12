# Two-terminal deployment

Run from `/home/user/sharpa-pi05/deploy/fr3_sharpa_runtime`.

One-time dependency startup (returns to the same terminal):

```bash
./start_hardware_stack.sh
```

Use the existing private supervised OpenPI configuration and valid permit. The
launcher does not create, renew, or alter permits. The existing Orbbec launch
requires `ORBBEC_SDK_LICENSE_ACCEPTED=YES`; retain the existing acceptance only
if applicable. `--plan` prints the startup commands without launching anything.

This machine's default configuration is selected by
`~/.config/fr3_sharpa_runtime/startup.json`. Its private full configuration and
permit are in the same directory (mode 0600). Operator/SOP attestations and the
existing 15-minute permit validity must be complete before startup. Verify
configuration alone with `./start_hardware_stack.sh --check-config`; this does
not start hardware. `--full-config` can still override the configured path.

FR3 controller and Safety Gateway use the existing Compose definitions, with
the existing OpenPI Gateway override. Cameras extend the existing container
environment and run only `triple_camera.launch.py`, not the recorder. Sharpa
uses the existing guarded startup script as a detached background process.
Startup activates real hardware services, but sends no policy action targets.
Successful launch is not evidence of ROS feedback readiness.

Terminal 1:

```bash
./serve.sh
```

Terminal 2:

```bash
./runtime.sh --preflight-only
```

Runtime automatically uses the existing Docker ROS domain, CycloneDDS settings
and Pixi ROS overlay. Preflight creates no command publishers. It checks FR3
and Sharpa feedback, three camera interfaces, Gateway, guarded runtime, a real
observation snapshot, policy connection/inference, and existing first-target
checks. If the ROS graph fails, policy inference is reported NOT_CHECKED.

Only after preflight PASS and human confirmation, in Terminal 2:

```bash
./runtime.sh --execute --enable-hardware
```

This executes the existing one-chunk runtime; it is not a continuous task loop.
No controller-bus bypass or public Sharpa-command fallback is added.

Dependency logs and guarded PID record: `~/.local/state/fr3_sharpa_runtime/`.
Compose services remain in the existing `docker` project. Repeated startup
reconciles Compose services and reuses the recorded guarded process when alive.
If a step fails, already started services remain running and the command exits
nonzero; inspect `compose.log` and `sharpa_guarded.log` before retrying. Guarded
feedback/device/permit/watchdog checks remain owned by the original runtime.

The existing controllers, Gateway and guarded driver retain their original
shutdown/stop mechanisms; closing the launcher terminal does not stop services.
