# Policy-first prepare

当前 prepare 使用 `sdk_direct` 后端：

1. 获取当前 FR3、Sharpa 和相机 observation。
2. 对 policy 首帧绝对 58D target 做低速关节空间插值。
3. FR3 命令仍经过 Safety Gateway；Sharpa 命令经过 `SharpaSdkDirectExecutor`。
4. 检查 joint limits、slew、feedback freshness、timestamp 和 watchdog。
5. 到达并稳定后重新采集 observation，再推理并执行完整 50 帧 chunk。

```bash
./runtime.sh --execute --enable-hardware \
  --prepare-speed-rad-s 0.03 \
  --execute-speed-scale 0.1 \
  --num-chunks 0
```

旧 configured-pose、permit 和 handoff 说明保留在 legacy 实现中，不是当前 sdk_direct 路径。

`--execute` 默认选择 sdk_direct 和 policy-first prepare。prepare 只在起始时执行一次并要求实测到位；其后每段重新观测、重新推理，低速衔接新首帧，再执行完整 50 帧。0 表示运行到 Ctrl-C，正整数指定 chunk 数量。

首帧衔接与慢放均作用于绝对关节目标，不重新叠加当前状态。手部跳变量在实际下发帧上检查，关节限位、反馈时效、SDK watchdog 和 FR3 网关保留。该路径是关节插值，不包含碰撞规划。
