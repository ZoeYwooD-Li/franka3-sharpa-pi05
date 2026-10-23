# FR3 + Sharpa π0.5 番茄任务部署

双 FR3 机械臂、双 Sharpa 灵巧手的视觉语言动作模型工程。当前部署使用 **π0.5 全量微调、30 Hz、原生 58 维绝对关节目标**。

## 当前配置

| 项目 | 配置 |
| --- | --- |
| 模型 | `gemma_2b` + `gemma_300m`，无 LoRA |
| 输入 | 三路 RGB、58 维实测关节位置、英文任务文本 |
| 输出 | 每次 50 帧 × 58 维，单位 rad，绝对关节位置 |
| 关节顺序 | 左臂 7、左手 22、右臂 7、右手 22 |
| 相机 | 头部 cam0、左腕 cam2、右腕 cam1 |
| 模型动作与设备下发 | 30 Hz，名义间隔约 33.33 ms |
| 执行后端 | `sdk_direct`：FR3 Safety Gateway + Sharpa Wave SDK |

默认权重位于外置盘，代码仓库不包含权重：

```text
/media/user/8C0F29AFFD6CD874/sharpa_tomato_ckpt_4000
```

归一化统计自动读取该 checkpoint 的 `assets/tomato_sharpa_ab4_normal_rgb_absolute_v1_30hz/norm_stats.json`。更换路径时，在两个终端设置相同的 `SHARPA_CHECKPOINT`；不要替换为其他实验的统计。

## 两个终端启动

先按 [runtime 部署说明](deploy/fr3_sharpa_runtime/README.md)启动 ROS、相机和机械臂依赖，准备 Pixi/SDK 环境。

终端一，加载权重、预热并启动推理服务：

```bash
cd /home/user/sharpa-pi05/deploy/fr3_sharpa_runtime
./serve.sh
```

终端二，下列命令连接设备并驱动机器人：

```bash
cd /home/user/sharpa-pi05/deploy/fr3_sharpa_runtime
./runtime.sh --execute --enable-hardware \
  --prepare-speed-rad-s 0.03 \
  --execute-speed-scale 0.1 \
  --num-chunks 0
```

执行流程：当前观测 → 推理首帧作为 prepare 目标 → 低速到位并稳定 → 重新观测、重新推理 → 衔接并执行完整 chunk → 下一轮。

`--num-chunks 0` 持续运行到 Ctrl-C；正整数指定段数。`--execute-speed-scale 0.1` 将每对模型动作点分为 10 步并保持 30 Hz 下发，50 帧主体约 16.37 秒，另加 prepare、首帧衔接、观测及推理时间。没有自动判定番茄任务完成的逻辑。prepare 为关节空间插值，不包含环境碰撞规划。

## 检查与验证

以下检查不连接机器人：

```bash
cd /home/user/sharpa-pi05
JAX_PLATFORMS=cpu .venv/bin/python -m pytest deploy/fr3_sharpa_runtime/tests -q
bash deploy/fr3_sharpa_runtime/serve.sh --check-only
bash deploy/fr3_sharpa_runtime/serve.sh --smoke-only
```

当前 4000 步权重已通过真实 GPU 加载及假观测推理，输出有限的 `(50, 58)`。这证明模型加载与动作接口可用，不代表现场任务成功。执行记录默认位于 `~/.local/state/fr3_sharpa_runtime/execution_traces/`。

原始转换器数据为 state116/action58；训练时选择实测位置并重排为 canonical 58D。实时部署输入已经是 canonical 58D，不能再套原始数据重排或重复添加当前关节位置。数据 normal/speedup 的区别、训练代码对应关系及尚未确认的实验信息见 [runtime 说明](deploy/fr3_sharpa_runtime/README.md)。

## 代码导航

| 目录 | 用途 |
| --- | --- |
| `deploy/fr3_sharpa_runtime/` | 当前模型服务、prepare、SDK 执行和运行记录 |
| `src/openpi/policies/fr3_sharpa_policy.py` | 58D 输入输出变换 |
| `src/openpi/policies/fr3_sharpa_protocol.py` | Sharpa 关节名称和协议 |
| `examples/fr3_sharpa/` | Sharpa 离线数据转换、导出与比较工具 |
| `src/openpi/training/` | 通用训练框架及保留的 Sharpa 历史训练配置 |
| `src/openpi/models/` | OpenPI 模型实现 |
| `scripts/` | 通用训练、统计和模型服务入口 |
| `packages/openpi-client/` | WebSocket 客户端 |
| `docs/` | 通用 OpenPI 参考资料与文档索引 |

当前全量模型的训练配方参考独立工程 `/home/user/tomato-openpi`。本仓库历史 Sharpa LoRA 配置不用于加载当前全量权重。旧平台部署和实验代码已从当前分支移除，可通过 Git 历史查阅。

权重、训练数据、运行日志和本机构建产物不提交 Git。许可证见 [LICENSE](LICENSE) 和 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)。
