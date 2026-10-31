# 本机 all 权重接入进度

独立目录：`/home/descfly/LTT/franka3-sharpa-trex-all`。
工作分支：`codex/trex-all-local-integration`。
模型目录：`/home/descfly/桌面/fanghao_Trex_all_6250`，不改动原权重。
模型 Python：本目录 `.venv-trex/bin/python`。
ROS Python：复用现有 litchi_hardware 的 Pixi ros2 解释器，未安装或修改其依赖。

## 阶段一：模型和无下发观测路径

已增加以下入口，均不创建运动命令发布者，不初始化手部 SDK：

- `ros_tactile.py`：订阅现有 SDK 所有者发布的物理手指话题。每指 F6 与
  deformation 必须有完全相同的采集时间戳和正确的 frame_id。保留每指
  16 个真实配对样本；不补零、不重复帧；检查断流、新鲜度、十指时间跨度。
- `capture.py`：在一个 ROS 节点内接收十指触觉、三路相机和四组关节状态，
  检查跨模态时间跨度，保存一次真实观测 NPZ。相机启动稳定检查为 2 秒。
- `shadow.py`：对观测文件做模型推理，保留原始 16×62 预测，并尝试用采样时
  的关节状态还原为 16×58 关节目标。IK 失败也保存原预测和失败原因，退出码 2。
  IK 成功只代表名义运动学可解，结果始终标记 `execute_ready=false`。
- `local.sh`：统一选择本机独立模型环境和既有 ROS 解释器。

已有发布端的 ROS header 是主机采集时间，没有 SDK 原始帧号。本路径能验证
消息配对与主机时间连续性，不能证明 SDK 内部没有丢帧，也不能替代现场验收。
现场若无法满足原有 75 ms 最大历史间隔，先诊断采样、发布和数据语义，不能
放宽阈值或用缓存重复填满 16 帧。

原分支漏提交了 `vendor/trex/tactile_vqvae/data/` 三个 Python 文件。已从
SOURCE.md 指定的上游提交 `ba4047ce3e9f3d2013d60de600e136897e90ccb8`
恢复，逐文件匹配原 `source_manifest.json`；哈希检查未修改。
`.gitignore` 已对这三个源码文件做例外处理。

## 本机操作入口

在独立目录下执行：

```bash
cd /home/descfly/LTT/franka3-sharpa-trex-all
bash deploy/fr3_sharpa_runtime/trex/local.sh check
bash deploy/fr3_sharpa_runtime/trex/local.sh warmup
bash deploy/fr3_sharpa_runtime/trex/local.sh serve
```

服务只监听 `127.0.0.1:5555`。已经有该服务时无需重复启动。另一个终端可用
以下合成数据探针检查通信，不需要设备上电：

```bash
bash deploy/fr3_sharpa_runtime/trex/local.sh probe
```

设备上电、现有只读采集与状态服务正常以后，采一次真实观测：

```bash
bash deploy/fr3_sharpa_runtime/trex/local.sh capture \
  --output trex-local-records/live-observation-001.npz --timeout 15
bash deploy/fr3_sharpa_runtime/trex/local.sh shadow \
  --observation trex-local-records/live-observation-001.npz \
  --output trex-local-records/shadow-001.npz \
  --prompt 'Pick up a tomato with the right hand, move it to the center, let the left hand receive it, and place it in the bowl.'
```

任务文字需要与当次实际任务相符。输出文件独占创建，重复运行时换新编号。
采集入口不会启动相机、机械臂控制或手部程序，也不会修改 ROS Domain 配置；
必须在与现有采集程序相同的 ROS Domain/RMW 配置下使用。

检查与预热日志、环境版本、探针结果保存在 `trex-local-records/`，该目录不提交。
合成探针数据不是真实机器人观测，不能用于动作验收。

## 双臂双手执行入口

已补齐同一主机上的双臂双手执行，使用原 π0.5 的手部 SDK 生命周期、Home 文件与 Safety Gateway。具体命令和停止行为见 [EXECUTE.md](EXECUTE.md)。

`execute --plan` 与 `execute --offline-replay` 均不打开设备。实际执行要求 `--execute --enable-hardware`。后台串行执行 slow/fast 推理和 IK，30 Hz 主循环下发 58 维绝对关节目标。模型服务以独占会话保护动作缓存。

代码和离线验证不等于真机任务成功。尚未下发 T-Rex 模型运动；首次真机操作使用有限单块低速命令，由现场人员观察并可随时停止。
