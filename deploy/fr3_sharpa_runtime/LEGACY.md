# Legacy fallback paths

以下文件保留用于回退和历史兼容，不是当前 `sdk_direct` 真机入口：

- `guarded_motion_ros.py`
- `handoff.py`
- `transport_service.py`
- `prepare.py`
- `prepare_ros.py`
- `capture_prepare_pose.py`
- `start_transport.sh`

legacy 路径可能依赖原有 permit、acquisition、handoff、GUI 或 `sharpa_driver`。不要把 legacy 命令和 `--sharpa-backend sdk_direct` 混用，也不要同时启动两套 Sharpa SDK/driver owner。
