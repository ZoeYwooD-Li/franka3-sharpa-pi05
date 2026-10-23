# 文档索引

当前 Sharpa 实机部署以[根 README](../README.md)和 [runtime 部署说明](../deploy/fr3_sharpa_runtime/README.md)为准。

| 文档 | 用途 |
| --- | --- |
| [Sharpa 部署](../deploy/fr3_sharpa_runtime/README.md) | 全量模型、30 Hz、两终端启动、SDK 与日志 |
| [Prepare 流程](../deploy/fr3_sharpa_runtime/PREPARE.md) | 初始到位、重新观测、完整 chunk 循环 |
| [Sharpa 数据转换](../examples/fr3_sharpa/README.md) | 离线 bag 转换和 LeRobot 导出 |
| [归一化统计](norm_stats.md) | OpenPI 通用归一化说明 |
| [Docker](docker.md) | OpenPI 通用容器参考 |
| [远程推理](remote_inference.md) | OpenPI 通用服务与客户端参考 |
| [上游 README 归档](README_原版_2026-09-29.md) | 通用模型安装与示例，非当前实机启动入口 |
| [贡献指南](CONTRIBUTING_2026-09-29.md) | 上游开发约定 |

旧平台专用文档随对应代码移除，需要查阅时使用 Git 历史。Sharpa legacy 后端仅供历史兼容，见 [LEGACY.md](../deploy/fr3_sharpa_runtime/LEGACY.md)；当前执行入口使用 `sdk_direct`。
