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

当前部署 checkpoint：

```text
/home/user/checkpoints/new-train/pi05_fr3_sharpa_tier_a_b1_20hz/sharpa_camfix_a70_b130_20hz_30k_wandb/29999
```

该训练版本使用已纠正的腕部相机映射：左腕 `cam2`，右腕 `cam1`。部署 norm_stats 仍为同一 20Hz、58D 数据集的统计文件。

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
| `compose.hardware.yaml` | 当前 Docker 依赖服务配置 |
| `hardware_stack.py` | 旧 guarded/permit 启动器，仅用于 legacy 路径 |
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

### Terminal 2：prepare + 完整 chunk 循环

```bash
cd /home/user/sharpa-pi05/deploy/fr3_sharpa_runtime
./runtime.sh \
  --execute \
  --enable-hardware \
  --prepare-speed-rad-s 0.03 \
  --execute-speed-scale 0.1 \
  --num-chunks 0
```

执行入口默认选择 `sdk_direct`，并自动执行 policy-first prepare，无需 permit、handoff 或旧 supervised 配置。`--execute --enable-hardware` 明确选择真机运行。

执行链路：当前实测状态和三路图像 → 首次推理只取首帧作 prepare 目标 → 低速到位并稳定 → 重新采集 → 重新推理 → 从实测姿态低速衔接新首帧 → 完整执行 50×58 chunk → 重新采集 → 下一轮推理与执行。

- `--num-chunks 0`：持续运行到 Ctrl-C；`--num-chunks 2` 重现两段试验；任意正整数指定段数。没有自动识别任务完成的逻辑。
- `--prepare-speed-rad-s 0.03`：初始 prepare 和每段首帧衔接的关节速度。
- `--execute-speed-scale 0.1`：相邻 policy target 间分为 10 步，设备仍按 20 Hz 下发。50 帧主体为 491 个下发点，约 24.55 秒，另外加上首帧衔接和推理/观测时间。非整数倒数的倍率向更慢一侧取整。默认 `1.0` 保留模型原速。

初始姿态可以与 prepare 目标不同；prepare 是关节空间直线插值，没有环境碰撞规划，不代表任意物理姿态都能无碰撞到达。prepare 目标仍来自模型首帧，不是固定 Home。

当前位置和所有预测点仍检查关节限位。SDK 跳变量检查施加在实际插值后的下发帧上，不再用原始模型节点误拒绝慢放轨迹。20 Hz 调度使用绝对时钟，避免每次 SDK 调用耗时叠加；反馈、SDK watchdog 和 FR3 网关继续工作。

消息类型构建放在项目 `.deployment/sharpa_ros_messages/install`，不再依赖重启会丢失的 `/tmp`。`runtime.sh` 首次发现缺失时自动调用仅编译消息类型的 `build_messages.sh`；也可以通过 `SHARPA_ROS_MESSAGES_OVERLAY` 指定已有构建。

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
  /home/user/.local/state/fr3_sharpa_runtime/execution_traces/*-chunk*.json
```

每轮保存 `*-chunkNNN.json`，包括原始动作、适配结果、执行前后观测、首帧衔接帧数和末帧跟踪误差；`COMPLETE` 表示指令下发完成，不代表番茄任务成功或所有关节精确到位。

整次运行另存 `*-events.jsonl` 和 `*-run.json`。prepare、观测、推理、执行的失败阶段与错误原因会保存；执行前已写入的 chunk 不再永久停留在 `IN_FLIGHT`。Ctrl-C 记为 `INTERRUPTED`。执行已完成而后续相机失败时，chunk 保留 `COMPLETE` 并记录观测失败，运行报告为 `FAILED`。

## 停止

先停止 Terminal 2，再停止 Terminal 1，均使用 `Ctrl-C`。需要停止 Docker 依赖时：

```bash
cd /home/user/gello_upper_body_teleop
docker compose --project-directory docker --env-file docker/.env \
  -f docker/compose.yaml -f docker/compose.openpi.yaml \
  -f /home/user/sharpa-pi05/deploy/fr3_sharpa_runtime/compose.hardware.yaml down
```

旧路径说明见 [LEGACY.md](LEGACY.md)。
