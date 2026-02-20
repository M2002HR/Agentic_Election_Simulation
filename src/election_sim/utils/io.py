import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

def ensure_dir(path: Union[str, Path]) -> str:
    p = str(path)
    os.makedirs(p, exist_ok=True)
    return p

def atomic_write_json(path: str, data: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


JsonPath = Union[str, Path]


def read_json(path: JsonPath) -> Any:
    p = str(path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: JsonPath, data: Any) -> None:
    atomic_write_json(str(path), data)
