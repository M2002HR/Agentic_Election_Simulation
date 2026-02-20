from pathlib import Path

import election_sim.llm.google_client as google_client_module
from election_sim.llm.google_client import GoogleLLMClient


class _DummyResp:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubLLM:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, prompt: str):
        return self._fn(prompt)


def test_rotate_on_429_and_trace_written(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    client = GoogleLLMClient(
        api_keys=["k1", "k2"],
        model_name="dummy",
        temperature=0.0,
        max_output_tokens=32,
        trace_path=trace,
        requests_per_minute=10_000,
        retry_on_429=True,
        rounds_limit=2,
        cooloff_sec=0.01,
        network_retries=0,
        network_backoff_sec=0.01,
    )

    def fake_build():
        if client.keyman.index == 0:
            return _StubLLM(lambda _p: (_ for _ in ()).throw(Exception("429 quota exceeded; retry in 0.1s")))
        return _StubLLM(lambda _p: _DummyResp("OK"))

    client._build = fake_build  # type: ignore[method-assign]
    out = client.invoke("hello", meta={"phase": "test"})

    assert out == "OK"
    assert client.keyman.index == 1
    assert trace.exists()
    lines = trace.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2


def test_build_disables_provider_internal_retries(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    captured = {}

    class _DummyChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    original = google_client_module.ChatGoogleGenerativeAI
    google_client_module.ChatGoogleGenerativeAI = _DummyChat  # type: ignore[assignment]
    try:
        client = GoogleLLMClient(
            api_keys=["k1"],
            model_name="dummy",
            temperature=0.0,
            max_output_tokens=16,
            trace_path=trace,
            requests_per_minute=60,
            retry_on_429=True,
            rounds_limit=1,
            cooloff_sec=0.01,
            network_retries=0,
            network_backoff_sec=0.01,
        )
        _ = client._build()
    finally:
        google_client_module.ChatGoogleGenerativeAI = original  # type: ignore[assignment]

    assert captured.get("max_retries") == 0
    assert captured.get("timeout") == 45.0
