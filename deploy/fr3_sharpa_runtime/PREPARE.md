# Policy-first prepare

当前 prepare 使用 `sdk_direct` 后端：

1. 获取当前 FR3、Sharpa 和相机 observation。
2. 对 policy 首帧绝对 58D target 做低速关节空间插值。
3. FR3 命令仍经过 Safety Gateway；Sharpa 命令经过 `SharpaSdkDirectExecutor`。
4. 检查 joint limits、slew、feedback freshness、timestamp 和 watchdog。
5. 到达并稳定后重新采集 observation，再推理并执行完整 50 帧 chunk。

```bash
./runtime.sh --execute --enable-hardware \
  --sharpa-backend sdk_direct \
  --prepare-policy-first \
  --prepare-speed-rad-s 0.03 \
  --execute-speed-scale 0.1 \
  --num-chunks 1
```

旧 configured-pose、permit 和 handoff 说明保留在 legacy 实现中，不是当前 sdk_direct 路径。
