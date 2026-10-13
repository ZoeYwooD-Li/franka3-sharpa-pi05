# Startup reference

当前推荐入口和完整命令见 [README.md](README.md)。当前真机部署使用两个终端：

1. `./serve.sh`：加载 π0.5 policy。
2. `./runtime.sh --execute --enable-hardware --sharpa-backend sdk_direct --prepare-policy-first ...`：执行 prepare 和 action chunk。

旧 permit、handoff、acquisition、guarded driver 说明仅适用于 legacy fallback，见 [LEGACY.md](LEGACY.md)。
