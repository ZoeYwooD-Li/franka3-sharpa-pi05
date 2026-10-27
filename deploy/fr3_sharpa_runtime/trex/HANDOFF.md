# T-Rex × 双 FR3 / Sharpa 代码交接

这里是独立的 T-Rex 部署适配，当前代码可做完整权重合同检查、GPU 模型预热与本机 ZMQ 推理。尚未连接现场 SDK 采样进程或 30 Hz 真机下发入口。不要直接运行旧 π0.5 的 `50×58` 执行程序。

## 两份代码

- 模型源码与补丁：`https://github.com/ZoeYwooD-Li/trex-sharpa-tomato`，分支 `full-finetune-server`，已核对提交 `df2a12e`。
- 机器人侧适配：`https://github.com/ZoeYwooD-Li/franka3-sharpa-pi05`，分支 `trex-integration`，从提交 `a9ee71d` 起包含已合并 LoRA 检查、训练图像尺寸握手。原有 `chunk-base-test` 不在此分支改动。

同事本机的 `trex-tactile-deploy-test` 目前只是独立目录中的临时分支，不要假设它已在任何远端。建议先克隆到**新目录**，保留已有工作目录，再逐项比较现场接口：

```bash
git clone --branch trex-integration \
  https://github.com/ZoeYwooD-Li/franka3-sharpa-pi05.git \
  /home/user/sharpa-pi05-trex-integration
git clone --branch full-finetune-server \
  https://github.com/ZoeYwooD-Li/trex-sharpa-tomato.git \
  /home/user/trex-sharpa-tomato
git clone https://github.com/ZhuoyangLiu2005/T-Rex.git /home/user/T-Rex
git -C /home/user/T-Rex checkout ba4047ce3e9f3d2013d60de600e136897e90ccb8
python3 /home/user/trex-sharpa-tomato/trex_sharpa_adapter/install_adapter.py --repo /home/user/T-Rex
python3 /home/user/trex-sharpa-tomato/trex_sharpa_adapter/install_lora.py --repo /home/user/T-Rex
python3 /home/user/trex-sharpa-tomato/trex_sharpa_adapter/install_inference.py --repo /home/user/T-Rex
```

安装器可能需要与 GPU 环境一致的 Python；如已有与训练代码匹配的 T-Rex 克隆，应先核对基准与工作区变更，不要直接在同事的临时分支或未提交的源码上覆盖安装。部署 fork 与官方模型源码是两份独立代码。

## 权重与无运动推理

将**完整目录**单独放在 Git 目录外，例如 `/home/user/checkpoints/trex_sharpa_lora_25000_merged`。LoRA 必须是与原始 midtrain 配对合并后的 `model.pt`，附带 `lora_origin.json`、`training_args.json`、`stats_data.json`、`adapter_metadata.json`、`processor/` 和 `robot/`。单独的 `lora.pt` 无法启动本服务。也可以选全参数 `checkpoint-0-6250`；不要根据训练步数假定哪个更好。原始数据不参与推理。

在**GPU 主机**用安装了 T-Rex 依赖的环境执行以下命令，替换 `PYTHON` 和 `CKPT`：

```bash
cd /home/user/sharpa-pi05-trex-integration
PYTHON=/path/to/trex-env/bin/python
CKPT=/home/user/checkpoints/trex_sharpa_lora_25000_merged
$PYTHON -m deploy.fr3_sharpa_runtime.trex.serve --checkpoint "$CKPT" --check-only
$PYTHON -m deploy.fr3_sharpa_runtime.trex.serve \
  --checkpoint "$CKPT" --trex-repo /home/user/T-Rex --cuda 0 --warmup-only
$PYTHON -m deploy.fr3_sharpa_runtime.trex.serve \
  --checkpoint "$CKPT" --trex-repo /home/user/T-Rex --cuda 0 --port 5555
```

前两步分别检查文件合同和做一次真实 GPU 前向，不向机器人发送命令。第三步持续启动本机模型服务。在另一终端做握手：

```bash
cd /home/user/sharpa-pi05-trex-integration
CKPT=/home/user/checkpoints/trex_sharpa_lora_25000_merged
/path/to/trex-env/bin/python - "$CKPT" <<'PY'
import sys
from deploy.fr3_sharpa_runtime.trex.client import TrexClient
with TrexClient() as client:
    print(client.hello(sys.argv[1]))
PY
```

`TrexClient` 要求本目录 `serve.py` 的 `hello` 握手，端口为 5555。模型仓库自己的 `serve_sharpa.sh` 在端口 5678，只适合其独立探针，不能替换本入口。服务使用 pickle，仅监听回环地址。GPU 与机器人分机时，应通过 SSH 本地端口转发连接，不要直接开放端口。

## 现场接线的剩余要求

在 SDK 独占进程中接入 `tactile.py` 的真实十指连续采样；核对 cam0/head、cam1/right wrist、cam2/left wrist。30 Hz 执行器需在动作块起点固定双臂实测 link8 法兰位姿，使用 `session.py` 只更新尚未下发的动作，再将 `actions.py` 的 IK 结果送入现场 Safety Gateway。先记录真实传感器输入与预测、不下发动作；后续机器人动作应经现场人员验收。已有临时分支上的实现可逐文件比较，但静态权重结构匹配不足以证明现场接口正确。
