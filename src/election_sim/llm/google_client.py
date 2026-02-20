from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, List

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("election_sim")


class RoundRobinKeyManager:
    def __init__(self, api_keys: List[str]) -> None:
        if not api_keys:
            raise ValueError("api_keys is empty")
        self.api_keys = api_keys
        self.index = 0

    def activate(self):
        os.environ["GOOGLE_API_KEY"] = self.api_keys[self.index]

    def rotate(self):
        if len(self.api_keys) == 1:
            return False
        self.index = (self.index + 1) % len(self.api_keys)
        self.activate()
        return True

    def label(self):
        return f"{self.index+1}/{len(self.api_keys)}"


class GoogleLLMClient:
    def __init__(
        self,
        *,
        api_keys: List[str],
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        trace_path: str | Path,
        requests_per_minute: int = 60,
        retry_on_429: bool = True,
        rounds_limit: int = 2,
        cooloff_sec: float = 30.0,
        network_retries: int = 4,
        network_backoff_sec: float = 1.5,
        request_timeout_sec: float = 45.0,
    ) -> None:

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        self.keyman = RoundRobinKeyManager(api_keys)
        self.keyman.activate()

        self.min_interval = 60.0 / max(1, requests_per_minute)
        self.last_call = 0.0
        self.retry_on_429 = bool(retry_on_429)
        self.rounds_limit = max(1, int(rounds_limit))
        self.cooloff_sec = max(0.0, float(cooloff_sec))
        self.network_retries = max(0, int(network_retries))
        self.network_backoff_sec = max(0.0, float(network_backoff_sec))
        self.request_timeout_sec = max(5.0, float(request_timeout_sec))

    def _build(self):
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            # Keep provider retries off so key rotation in this client can react immediately.
            max_retries=0,
            timeout=self.request_timeout_sec,
        )

    def _rate_limit(self):
        now = time.time()
        delta = now - self.last_call
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last_call = time.time()

    @staticmethod
    def _response_text(resp: Any) -> str:
        content = getattr(resp, "content", resp)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                txt = getattr(item, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()
        return str(content)

    @staticmethod
    def _is_quota_error(err: Exception) -> bool:
        if isinstance(err, httpx.HTTPStatusError) and err.response is not None and err.response.status_code == 429:
            return True
        status_code = getattr(err, "status_code", None)
        if status_code == 429:
            return True
        msg = str(err).lower()
        return (
            "resourceexhausted" in msg
            or "resource_exhausted" in msg
            or ("429" in msg and ("quota" in msg or "rate limit" in msg))
            or ("quota exceeded" in msg)
        )

    @staticmethod
    def _is_transient_network_error(err: Exception) -> bool:
        if isinstance(
            err,
            (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
                httpx.NetworkError,
            ),
        ):
            return True
        msg = str(err).lower()
        return any(
            token in msg
            for token in [
                "timed out",
                "timeout",
                "connection reset",
                "temporarily unavailable",
                "service unavailable",
                "502",
                "503",
                "504",
            ]
        )

    @staticmethod
    def _retry_delay_sec(err: Exception) -> float | None:
        text = str(err)
        patterns = [
            r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
            r"please retry in\s+([0-9]+(?:\.[0-9]+)?)s",
            r"retry_delay\s*\{\s*seconds:\s*([0-9]+)\s*\}",
            r"retry after\s+([0-9]+(?:\.[0-9]+)?)\s*seconds",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
            if not m:
                continue
            try:
                return float(m.group(1))
            except Exception:
                continue
        return None

    def _trace(self, payload: dict[str, Any]) -> None:
        try:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception as err:
            logger.warning("Failed to write trace event: %s", err)

    def invoke(self, prompt: str, meta: dict[str, Any] | None = None) -> str:
        meta = meta or {}
        key_count = max(1, len(self.keyman.api_keys))
        max_key_attempts = max(1, self.rounds_limit) * key_count
        key_attempt = 0
        network_attempt = 0
        quota_hit_indices: set[int] = set()
        last_error: Exception | None = None

        while key_attempt < max_key_attempts:
            key_idx = self.keyman.index
            key_label = self.keyman.label()
            attempt_no = key_attempt + 1
            try:
                self._rate_limit()
                llm = self._build()
                resp = llm.invoke(prompt)
                text = self._response_text(resp)
                self._trace(
                    {
                        "ts": time.time(),
                        "event": "llm_invoke",
                        "status": "ok",
                        "model": self.model_name,
                        "key": key_label,
                        "attempt": attempt_no,
                        "meta": meta,
                        "prompt": prompt,
                        "output": text,
                    }
                )

                return text

            except Exception as e:
                last_error = e
                err_msg = str(e)
                quota_err = self._is_quota_error(e)
                transient_network = self._is_transient_network_error(e)
                suggested_delay = self._retry_delay_sec(e)

                self._trace(
                    {
                        "ts": time.time(),
                        "event": "llm_invoke",
                        "status": "error",
                        "model": self.model_name,
                        "key": key_label,
                        "attempt": attempt_no,
                        "meta": meta,
                        "prompt": prompt,
                        "error": err_msg,
                        "quota_error": quota_err,
                        "transient_network_error": transient_network,
                        "suggested_retry_delay_sec": suggested_delay,
                    }
                )

                if transient_network and network_attempt < self.network_retries:
                    network_attempt += 1
                    backoff = self.network_backoff_sec * (2 ** (network_attempt - 1))
                    logger.warning(
                        "Transient network error on key %s; retry %d/%d in %.1fs: %s",
                        key_label,
                        network_attempt,
                        self.network_retries,
                        backoff,
                        err_msg,
                    )
                    time.sleep(backoff)
                    continue

                if quota_err and self.retry_on_429:
                    quota_hit_indices.add(key_idx)
                    key_attempt += 1
                    logger.warning(
                        "Quota/rate-limit hit on key %s (attempt %d/%d): %s",
                        key_label,
                        attempt_no,
                        max_key_attempts,
                        err_msg,
                    )
                    rotated = self.keyman.rotate()
                    if rotated:
                        logger.info("Rotated API key to %s", self.keyman.label())

                    if len(quota_hit_indices) >= key_count and key_attempt < max_key_attempts:
                        sleep_for = max(self.cooloff_sec, suggested_delay or 0.0)
                        logger.warning(
                            "All API keys hit quota in current cycle; sleeping %.1fs before next round.",
                            sleep_for,
                        )
                        self._trace(
                            {
                                "ts": time.time(),
                                "event": "llm_invoke",
                                "status": "cooloff",
                                "model": self.model_name,
                                "attempt": attempt_no,
                                "meta": meta,
                                "sleep_sec": sleep_for,
                                "reason": "all_keys_quota_limited",
                            }
                        )
                        time.sleep(sleep_for)
                        quota_hit_indices.clear()
                    elif (not rotated) and key_attempt < max_key_attempts:
                        sleep_for = max(self.cooloff_sec, suggested_delay or 0.0)
                        logger.warning(
                            "Single key quota-limited; sleeping %.1fs before retry.",
                            sleep_for,
                        )
                        time.sleep(sleep_for)
                    continue

                key_attempt += 1
                rotated = self.keyman.rotate()
                if rotated and key_attempt < max_key_attempts:
                    logger.warning(
                        "LLM error on key %s; rotated to %s (attempt %d/%d): %s",
                        key_label,
                        self.keyman.label(),
                        key_attempt,
                        max_key_attempts,
                        err_msg,
                    )
                    continue
                break

        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM invocation failed without a captured exception.")

    def fork(
        self,
        *,
        model_name: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        trace_path: Path | None = None,
        requests_per_minute: int | None = None,
    ) -> "GoogleLLMClient":

        return GoogleLLMClient(
            api_keys=self.keyman.api_keys,
            model_name=model_name or self.model_name,
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=self.max_output_tokens if max_output_tokens is None else max_output_tokens,
            trace_path=trace_path or self.trace_path,
            requests_per_minute=requests_per_minute or int(60 / self.min_interval),
            retry_on_429=self.retry_on_429,
            rounds_limit=self.rounds_limit,
            cooloff_sec=self.cooloff_sec,
            network_retries=self.network_retries,
            network_backoff_sec=self.network_backoff_sec,
            request_timeout_sec=self.request_timeout_sec,
        )
