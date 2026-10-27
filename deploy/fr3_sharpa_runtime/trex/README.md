# T-Rex 双 FR3 / Sharpa 部署适配进度

此目录是与现有 π0.5 部署分开的 T-Rex 路径。当前包含 **模型服务、输入校验、SDK 触觉采样器、link8 动作还原和离线 chunk 更新逻辑**；还没有把采样器连接到现场 SDK 进程、30 Hz 实时执行器或真机下发入口。不要把本目录的测试通过当作真机可执行证明。

## 已实现的合同

- 合并后的 25,000 步 T-Rex 权重为 PyTorch `model.pt`，动作 `[16,62]`。左臂 9、左手 22、右臂 9、右手 22。
- 每臂 9 维表示相对**本动作块起点实测** `fr3v2_link0 → fr3v2_link8` 法兰位姿的平移 3 维及旋转前两列 6 维。十六帧均以同一基准还原，再通过名义 URDF 做 IK；手指仍为绝对关节角。
- 三路相机是 cam0 头部、cam1 右腕、cam2 左腕；触觉必须是左手后右手、各自 thumb/index/middle/ring/pinky。F6 历史 `[16,10,6]` 使用采集驱动原单位，形变图 `[10,240,240]` 归一化到 `[0,1]`。
- `contract.py` 校验完整权重或已合并 LoRA 的来源记录、`model.pt` 大小、16×62 动作统计、相机、手指顺序和 URDF SHA256。`assets/dual_fr3_kinematics.urdf` 来自用户数采工程，SHA256 为 `6be9900731b19bfb3edd56e08d63feb9112de3045957f84088b0fe6c7c873939`，与合并权重的左右 URDF 完全一致。它是名义几何，并非实机标定。
- `serve.py` 复用完整官方 T-Rex 仓库的 `scripts/test.py` 模型代码，把预热数据修正为双手十指，并按训练配置将三路图像缩放至 384×288；只监听本机回环地址。`client.py` 限制连接回环地址并核对服务端动作形状、URDF、图像尺寸和 chunk ID。
- `tactile.py` 调用采集驱动同名的 `read_tactile(finger)` 物理手指 API，不手写左右手通道编号；只有十指身份、F6、形变图和时间连续性均通过时才形成 16 帧历史。此模块尚未接入现场进程。

## 当前可做的检查

以下命令不连接机器人：

```bash
cd /path/to/franka3-sharpa-pi05
python3 -m pytest deploy/fr3_sharpa_runtime/tests/test_trex_actions.py \
  deploy/fr3_sharpa_runtime/tests/test_trex_client.py \
  deploy/fr3_sharpa_runtime/tests/test_trex_session.py \
  deploy/fr3_sharpa_runtime/tests/test_trex_tactile.py -q

python3 -m deploy.fr3_sharpa_runtime.trex.serve \
  --checkpoint /path/to/trex_sharpa_fr3_lora_25000_merged \
  --check-only
```

在配置好官方 T-Rex 的 CUDA 环境后，可进一步运行一次真实模型预热；它不发送运动指令：

```bash
python3 -m deploy.fr3_sharpa_runtime.trex.serve \
  --checkpoint /path/to/trex_sharpa_fr3_lora_25000_merged \
  --trex-repo /path/to/T-Rex --cuda 0 --warmup-only
```

通过预热后可去掉 `--warmup-only` 启动本机 `tcp://127.0.0.1:5555` 模型服务。该服务使用官方 ZMQ/pickle 请求格式，只应在可信本机运行。它不接触机器人硬件。

GPU 主机的完整 T-Rex 源码应从 `ZoeYwooD-Li/trex-sharpa-tomato` 的 `full-finetune-server` 分支安装 `install_adapter.py`、`install_lora.py`、`install_inference.py` 补丁，基准提交是 `ba4047ce3e9f3d2013d60de600e136897e90ccb8`。本目录 `TrexClient` 使用带 `hello` 握手的 **本目录 `serve.py`**，不要连接另一个仓库 `serve_sharpa.sh` 的 5678 端口：那个入口只适合独立模型探针，协议无此握手。权重单独传到 Git 目录之外；不要提交 `model.pt`、`lora.pt` 或原始数据。

## 接下来必须完成

1. 在现有独占 Sharpa SDK 的进程中读取十指 F6 和形变图，校验物理手指身份及时间戳，形成真实 16 帧 F6 历史；确认现场 SDK 版本与采集工程一致。
2. 把 T-Rex 服务和固定基准 chunk 会话接入实时执行器。慢速视觉请求与快速触觉修正需要与 30 Hz 下发并行调度；只替换当前 π0.5 `50×58` WebSocket 请求不足以实现该行为。
3. 对 IK 结果执行现有 Safety Gateway、手指范围、关节速度/跳变、反馈新鲜度和首帧衔接检查。先无下发预检，再低速单帧/单块验证。名义 URDF 无法提供环境碰撞保证。

原 π0.5 运行入口尚未改动，仍使用自己的 Orbax 权重及 `50×58` 合同。不要把 T-Rex `model.pt` 指向 `SHARPA_CHECKPOINT`。
