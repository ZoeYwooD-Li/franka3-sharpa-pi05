# T-Rex × 双 FR3 / Sharpa：单仓库交接

只需克隆本仓库的 `trex-single-repo` 分支。它包含机器人侧适配、官方 T-Rex 的训练/推理核心源码、Sharpa/LoRA 补丁与无运动探针。官方源码及补丁来源和文件哈希见 [`vendor/SOURCE.md`](vendor/SOURCE.md)。不需要再克隆第二个个人仓库或官方仓库。

模型权重仍单独放在 Git 目录外。用户已将权重复制到同事电脑；先确认所选目录含完整 `model.pt`、`processor/`、`stats_data.json`、`adapter_metadata.json`、`robot/`，合并 LoRA 还必须含 `lora_origin.json`。只有 `lora.pt` 的目录不能直接推理。

## 拉取与准备

在没有现场未提交改动的**新目录**克隆，不覆盖同事原有的 `trex-tactile-deploy-test`：

```bash
git clone --branch trex-single-repo \
  https://github.com/ZoeYwooD-Li/franka3-sharpa-pi05.git \
  /home/user/sharpa-pi05-trex-single
cd /home/user/sharpa-pi05-trex-single
```

模型服务需单独的 CUDA PyTorch 环境，不要把 π0.5/JAX 环境直接当作 T-Rex 环境。先安装与目标 GPU/驱动匹配的 PyTorch 和 torchvision，再安装 `deploy/fr3_sharpa_runtime/trex/requirements-inference.txt`。训练服务器曾使用 `transformers==4.57.3`、`huggingface-hub==0.36.2`；本仓库要求同版本。

## 无运动验证与模型服务

以下 `PYTHON` 指向配置好的 T-Rex 环境，`CKPT` 指向同事电脑上的**完整**权重目录。前两步不读取设备，也不发布运动命令：

```bash
cd /home/user/sharpa-pi05-trex-single
PYTHON=/path/to/trex-env/bin/python
CKPT=/path/to/trex_sharpa_lora_25000_merged
$PYTHON -m deploy.fr3_sharpa_runtime.trex.serve --checkpoint "$CKPT" --check-only
$PYTHON -m deploy.fr3_sharpa_runtime.trex.serve --checkpoint "$CKPT" --cuda 0 --warmup-only
```

只有 GPU 预热出现 16×62 有限值动作后，才在一个终端启动服务：

```bash
$PYTHON -m deploy.fr3_sharpa_runtime.trex.serve --checkpoint "$CKPT" --cuda 0 --port 5555
```

另一个终端运行**合成数据**探针；它只通过本机 ZMQ 请求推理：

```bash
cd /home/user/sharpa-pi05-trex-single
PYTHON=/path/to/trex-env/bin/python
CKPT=/path/to/trex_sharpa_lora_25000_merged
$PYTHON -m deploy.fr3_sharpa_runtime.trex.probe --checkpoint "$CKPT" --port 5555
```

服务使用 pickle 协议，仅监听 127.0.0.1。GPU 与机器人分机时使用 SSH 本地端口转发，不能直接开放 5555。探针成功只证明模型和协议可用，不证明真实传感器或机器人动作已验证。

## 真机接线仍需现场完成

本仓库目前有十指真实 SDK 采样器、相机/动作合同、link8 IK、动作块更新和已有 π0.5 的 FR3 Safety Gateway / Sharpa SDK 执行基础，但**T-Rex 还没有连接到现场 SDK 独占进程和 30 Hz 执行器**。同事应先核对实际手指 SDK API、相机身份、时间戳、手指关节范围和安全网关配置，再接这条独立路径。先只读记录真实输入与预测，不下发动作；随后由现场人员进行低速、单动作块验收。不能把 T-Rex 16×62 输出直接喂给旧 π0.5 的 50×58 入口。

同事原来的 `trex-tactile-deploy-test` 是独立的临时分支；要复用其中的现场修改，请先比较差异，再选择性合入本分支。不要覆盖该分支的未提交文件。
