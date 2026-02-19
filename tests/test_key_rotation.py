import os
import json
import types
import builtins
import time

import pytest

from election_sim.llm.key_manager import RoundRobinKeyManager, RateLimiter, is_429_error
from election_sim.llm.google_client import GoogleLLMClient

class DummyLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass

def test_is_429_error():
    assert is_429_error("RESOURCE_EXHAUSTED")
    assert is_429_error("429 Too Many Requests")
    assert not is_429_error("500")

def test_round_robin_switch_on_429(tmp_path, monkeypatch):
    # Prepare dummy keys
    keys = ["k1", "k2", "k3"]
    logger = DummyLogger()
    km = RoundRobinKeyManager(keys=keys, start_index=0, logger=logger)
    limiter = RateLimiter(0.0)

    # Monkeypatch ChatGoogleGenerativeAI to raise 429 twice, then succeed
    class DummyErr(Exception): pass

    class DummyModel:
        calls = 0
        def __init__(self, *a, **k): pass
        def invoke(self, prompt):
            DummyModel.calls += 1
            if DummyModel.calls <= 2:
                raise DummyErr("RESOURCE_EXHAUSTED 429")
            return types.SimpleNamespace(content="OK")

    # Patch both class and error type used by client
    import election_sim.llm.google_client as gc
    monkeypatch.setattr(gc, "ChatGoogleGenerativeAI", DummyModel)
    monkeypatch.setattr(gc, "ChatGoogleGenerativeAIError", DummyErr)

    project_root = str(tmp_path)
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)
    trace = str(tmp_path / "trace.jsonl")

    client = GoogleLLMClient(
        project_root=project_root,
        run_dir=run_dir,
        model_name="dummy",
        temperature=0.0,
        max_output_tokens=10,
        retry_on_429=True,
        rounds_limit=1,
        cooloff_sec=0.0,
        limiter=limiter,
        keyman=km,
        logger=logger,
        trace_path=trace,
    )

    out = client.invoke("x", meta={"t": 1})
    assert out == "OK"
    # after 2 failures, key index advanced twice: k1->k2->k3
    assert km.index == 2
    # global key state persisted
    state_path = os.path.join(project_root, ".key_state.json")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    assert state["key_index"] == 2
