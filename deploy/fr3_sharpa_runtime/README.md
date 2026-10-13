# FR3 + Sharpa π0.5 runtime

当前真机路径是 `sdk_direct`：

```text
OpenPI π0.5 policy server
        ↓
fr3_sharpa_runtime
        ↓
FR3 Safety Gateway + SharpaSdkDirectExecutor
        ↓
litchi_hardware.SharpaSdkHand
        ↓
Sharpa Wave SDK
```

旧 `litchibot` guarded runtime、permit、handoff 和 acquisition 代码保留为 legacy fallback，当前 sdk_direct 执行不调用它们。

## 文件职责

| 文件/组 | 作用 |
|---|---|
| `run.py`, `runtime.py`, `config.py` | CLI、policy 请求和运行入口 |
| `protocol.py`, `action_adapter.py`, `records.py` | 58D action 协议和 adapter |
| `ros_observation.py`, `observation_worker.py`, `feedback.py` | 实时 ROS observation / feedback |
| `policy_prepare.py` | policy-first 慢速 prepare 和 chunk 调度 |
| `sdk_direct.py`, `sdk_direct_motion_ros.py` | 当前 Sharpa SDK direct 后端 |
| `execution_trace.py` | sdk_direct 每个 chunk 的复盘记录 |
| `ros_transport.py`, `motion_worker.py` | ROS Safety Gateway 边界和 worker IPC |
| `guarded_motion_ros.py`, `handoff.py`, `transport_service.py` | legacy guarded / handoff 回退路径 |
| `hardware_stack.py`, `compose.hardware.yaml` | Docker 依赖服务管理 |
| `serve.py`, `serve.sh` | π0.5 policy server |
| `runtime.sh` | 当前推荐执行入口 |
| `tests/` | 单元和协议测试 |

## 测试

```bash
cd /home/user/sharpa-pi05
git status --short
git branch --show-current
PYTHONPATH=. uv run pytest -q deploy/fr3_sharpa_runtime/tests
```

## 一次性启动 ROS 依赖

```bash
cd /home/user/gello_upper_body_teleop
docker compose \
  --profile openpi-hardware \
  --project-directory docker \
  --env-file docker/.env \
  -f docker/compose.yaml \
  -f docker/compose.openpi.yaml \
  -f /home/user/sharpa-pi05/deploy/fr3_sharpa_runtime/compose.hardware.yaml \
  up -d franka-control teleop-control openpi-cameras
```

## 两终端真机部署

### Terminal 1：policy server

```bash
cd /home/user/sharpa-pi05/deploy/fr3_sharpa_runtime
./serve.sh
```

### Terminal 2：prepare + chunk1

```bash
cd /home/user/sharpa-pi05/deploy/fr3_sharpa_runtime
./runtime.sh \
  --execute \
  --enable-hardware \
  --sharpa-backend sdk_direct \
  --prepare-policy-first \
  --prepare-speed-rad-s 0.03 \
  --num-chunks 1
```

执行链路：实时 FR3/Sharpa/camera observation → policy 首帧 target → 低速 prepare → 重新采集 → policy inference → 50×58 action chunk → FR3 Safety Gateway + Sharpa SDK direct。

不要同时启动旧 `sharpa_driver` 和 `sdk_direct`；两者不能共同拥有 Sharpa 设备。

## execution trace

sdk_direct 每个 chunk 默认保存到：

```text
/home/user/.local/state/fr3_sharpa_runtime/execution_traces/
```

也可以指定：

```bash
--execution-trace-dir /home/user/sharpa_execution_traces
```

查看：

```bash
ls -lt /home/user/.local/state/fr3_sharpa_runtime/execution_traces
jq '.status, .action_shape, .chunk_duration_s, .end_to_end_duration_s' \
  /home/user/.local/state/fr3_sharpa_runtime/execution_traces/*.json
```

## 停止

先停止 Terminal 2，再停止 Terminal 1，均使用 `Ctrl-C`。需要停止 Docker 依赖时：

```bash
cd /home/user/gello_upper_body_teleop
docker compose --project-directory docker --env-file docker/.env \
  -f docker/compose.yaml -f docker/compose.openpi.yaml \
  -f /home/user/sharpa-pi05/deploy/fr3_sharpa_runtime/compose.hardware.yaml down
```

旧路径说明见 [LEGACY.md](LEGACY.md)。
