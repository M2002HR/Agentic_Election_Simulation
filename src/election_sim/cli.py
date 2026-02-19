from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from election_sim.config import dump_resolved_config, load_config
from election_sim.llm.google_client import GoogleLLMClient
from election_sim.llm.key_manager import (
    RateLimiter,
    RoundRobinKeyManager,
    load_global_key_state,
    load_keys_from_env,
)
from election_sim.phase1.eval import run_phase1
from election_sim.utils.io import ensure_dir
from election_sim.utils.logging import setup_logger


def make_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_llm(cfg, run_dir: str, logger) -> GoogleLLMClient:
    project_root = str(Path(__file__).resolve().parents[2])
    keys = load_keys_from_env()
    start_idx = int(load_global_key_state(project_root).get("key_index", 0))
    keyman = RoundRobinKeyManager(keys=keys, start_index=start_idx, logger=logger)
    limiter = RateLimiter(cfg.llm.min_interval_sec)
    trace_path = os.path.join(run_dir, cfg.logging.trace_jsonl)

    return GoogleLLMClient(
        project_root=project_root,
        run_dir=run_dir,
        model_name=cfg.llm.model_name,
        temperature=cfg.llm.temperature,
        max_output_tokens=cfg.llm.max_output_tokens,
        retry_on_429=cfg.llm.retry_on_429,
        rounds_limit=cfg.llm.rounds_limit,
        cooloff_sec=cfg.llm.cooloff_sec,
        limiter=limiter,
        keyman=keyman,
        logger=logger,
        trace_path=trace_path,
    )


def _init_run(cfg, run_id: str | None = None) -> tuple[str, object]:
    rid = run_id or make_run_id()
    run_dir = os.path.join(cfg.project.run_dir_base, rid)
    ensure_dir(run_dir)
    logger = setup_logger(run_dir)
    Path(os.path.join(run_dir, "config.resolved.yaml")).write_text(
        dump_resolved_config(cfg), encoding="utf-8"
    )
    return run_dir, logger


def cmd_smoke_test(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    run_dir, logger = _init_run(cfg)
    llm = build_llm(cfg, run_dir, logger)
    out = llm.invoke("Reply with exactly: OK", meta={"phase": "smoke-test"})
    print(out.strip())


def cmd_phase1(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    run_dir, logger = _init_run(cfg)
    llm = build_llm(cfg, run_dir, logger)
    run_phase1(cfg, llm, run_dir, logger)
    print(f"Run complete: {run_dir}")


def cmd_phase2(cfg_path: str) -> None:
    from election_sim.phase2.debate import run_debate

    cfg = load_config(cfg_path)
    run_dir, logger = _init_run(cfg)
    llm = build_llm(cfg, run_dir, logger)
    run_debate(cfg, llm, run_dir, logger)
    print(f"Run complete: {run_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="election_sim")
    p.add_argument("command", choices=["smoke-test", "phase1", "phase2"])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args(argv)

    if args.command == "smoke-test":
        cmd_smoke_test(args.config)
    elif args.command == "phase1":
        cmd_phase1(args.config)
    elif args.command == "phase2":
        cmd_phase2(args.config)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
