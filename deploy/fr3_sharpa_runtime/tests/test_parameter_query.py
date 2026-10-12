import sys
from types import SimpleNamespace

import pytest

from deploy.fr3_sharpa_runtime.ros_transport import _require_parameter


@pytest.mark.parametrize("expected,actual,passes", [
    (["openpi"], ["openpi"], True),
    (["openpi"], ["gello"], False),
    (True, True, True),
    (True, False, False),
])
def test_standard_parameter_service_keeps_checks(monkeypatch, expected, actual, passes):
    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(spin_until_future_complete=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "rcl_interfaces.srv", SimpleNamespace(GetParameters=SimpleNamespace(Request=SimpleNamespace)))
    value = SimpleNamespace(string_array_value=actual, bool_value=actual)
    future = SimpleNamespace(done=lambda: True, exception=lambda: None,
                             result=lambda: SimpleNamespace(values=[value]))
    requests, destroyed = [], []
    client = SimpleNamespace(wait_for_service=lambda **k: True,
                             call_async=lambda req: requests.append(req.names) or future)
    node = SimpleNamespace(create_client=lambda srv, topic: client,
                           destroy_client=lambda obj: destroyed.append(obj))
    if passes:
        _require_parameter(node, "remote", "required", expected)
    else:
        with pytest.raises(RuntimeError):
            _require_parameter(node, "remote", "required", expected)
    assert requests == [["required"]]
    assert destroyed == [client]
