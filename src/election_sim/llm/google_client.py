import os
import time
import random
from typing import Any, Dict, Optional

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from election_sim.llm.key_manager import (
    RoundRobinKeyManager,
    RateLimiter,
    is_429_error,
    load_global_key_state,
    save_global_key_state,
)
from election_sim.utils.io import append_jsonl, atomic_write_json


class GoogleLLMClient:
    """
    Google Gemini/Gemma client with:
    - Round-robin API key rotation on 429
    - Network retry with exponential backoff
    - Persistent key index state
    """

    def __init__(
        self,
        *,
        project_root: str,
        run_dir: str,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
        retry_on_429: bool,
        rounds_limit: int,
        cooloff_sec: float,
        limiter: RateLimiter,
        keyman: RoundRobinKeyManager,
        network_retries: int,
        network_backoff_sec: float,
        logger,
        trace_path: str,
    ):
        self.project_root = project_root
        self.run_dir = run_dir
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.retry_on_429 = bool(retry_on_429)
        self.rounds_limit = int(rounds_limit)
        self.cooloff_sec = float(cooloff_sec)
        self.limiter = limiter
        self.keyman = keyman
        self.network_retries = int(network_retries)
        self.network_backoff_sec = float(network_backoff_sec)
        self.logger = logger
        self.trace_path = trace_path

    def _persist_last_good_key_index(self, idx: int) -> None:
        state = load_global_key_state(self.project_root)
        state["key_index"] = int(idx)
        save_global_key_state(self.project_root, state)
        atomic_write_json(
            os.path.join(self.run_dir, "run_state.json"),
            {"key_index": int(idx)},
        )

    def _do_call(self, prompt: str) -> str:
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        resp = llm.invoke(prompt)
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    def invoke(self, prompt: str, *, meta: Optional[Dict[str, Any]] = None) -> str:
        meta = meta or {}
        keys_count = len(self.keyman.keys)

        rounds_done = 0
        tried_in_round = 0

        while True:
            # --- network retry wrapper ---
            for attempt in range(self.network_retries + 1):
                try:
                    self.limiter.wait()
                    out = self._do_call(prompt)

                    # success
                    self._persist_last_good_key_index(self.keyman.index)
                    append_jsonl(
                        self.trace_path,
                        {
                            "ts": time.time(),
                            "key_index": self.keyman.index,
                            "meta": meta,
                            "prompt": prompt,
                            "output": out,
                        },
                    )
                    return out

                except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                    if attempt >= self.network_retries:
                        raise
                    sleep_for = self.network_backoff_sec * (2 ** attempt) + random.random() * 0.3
                    self.logger.warning(
                        "Network error: %s | retry in %.2fs (attempt %d/%d)",
                        str(e),
                        sleep_for,
                        attempt + 1,
                        self.network_retries,
                    )
                    time.sleep(sleep_for)
                    continue

                except ChatGoogleGenerativeAIError as e:
                    msg = str(e)
                    self.logger.warning("LLM error (key=%d): %s", self.keyman.index + 1, msg)

                    if not self.retry_on_429 or not is_429_error(msg):
                        raise

                    tried_in_round += 1
                    self.keyman.next_key()

                    if tried_in_round >= keys_count:
                        rounds_done += 1
                        tried_in_round = 0

                        if rounds_done >= self.rounds_limit:
                            sleep_for = float(min(max(self.cooloff_sec, 1.0), 600.0))
                            self.logger.warning(
                                "All key rounds exhausted. Cooling off %.1fs.",
                                sleep_for,
                            )
                            time.sleep(sleep_for)
                            rounds_done = 0

                    break  # break network retry loop → try next key
