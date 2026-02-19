import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from election_sim.utils.io import atomic_write_json, load_json

def load_keys_from_env() -> List[str]:
    load_dotenv(override=False)
    raw = os.getenv("GOOGLE_API_KEYS", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]

def set_active_api_key(key: str) -> None:
    os.environ["GOOGLE_API_KEY"] = key

def is_429_error(msg: str) -> bool:
    return ("RESOURCE_EXHAUSTED" in msg) or ("429" in msg)

def global_key_state_path(project_root: str) -> str:
    return os.path.join(project_root, ".key_state.json")

def load_global_key_state(project_root: str) -> Dict[str, Any]:
    p = global_key_state_path(project_root)
    data = load_json(p)
    if data is None:
        return {"key_index": 0}
    if "key_index" not in data:
        data["key_index"] = 0
    return data

def save_global_key_state(project_root: str, data: Dict[str, Any]) -> None:
    p = global_key_state_path(project_root)
    atomic_write_json(p, data)

@dataclass
class RateLimiter:
    min_interval_sec: float
    last_call: float = 0.0

    def wait(self) -> None:
        now = time.time()
        delta = now - self.last_call
        if delta < self.min_interval_sec:
            time.sleep(self.min_interval_sec - delta)
        self.last_call = time.time()

class RoundRobinKeyManager:
    def __init__(self, keys: List[str], start_index: int, logger):
        if not keys:
            raise RuntimeError("No API keys found. Set GOOGLE_API_KEYS in .env.")
        self.keys = keys
        self.logger = logger
        self.index = int(start_index) % len(keys)
        self.activate(self.index)

    def activate(self, idx: int) -> None:
        self.index = idx % len(self.keys)
        set_active_api_key(self.keys[self.index])
        self.logger.info("Active API key index: %d/%d", self.index + 1, len(self.keys))

    def next_key(self) -> None:
        self.activate((self.index + 1) % len(self.keys))
