import sys
from types import SimpleNamespace

from deploy.fr3_sharpa_runtime.timing import TimedPolicy


def test_timing_preserves_outputs_and_restores_policy(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(block_until_ready=lambda value: value))
    class Policy:
        _input_transform = staticmethod(lambda x: x)
        _output_transform = staticmethod(lambda x: x)
        _sample_actions = staticmethod(lambda x: x)
        def infer(self, obs):
            return self._output_transform({"actions": self._sample_actions(self._input_transform(obs))})
    policy = Policy()
    original = {k: getattr(policy, k) for k in ("_input_transform", "_output_transform", "_sample_actions")}
    obs = object()
    wrapper = TimedPolicy(policy)
    result = wrapper.infer(obs)
    assert result["actions"] is obs
    assert all(getattr(policy,k) is fn for k,fn in original.items())
    assert result["diagnostic_timing"]["model_sampling_ms"] >= 0
