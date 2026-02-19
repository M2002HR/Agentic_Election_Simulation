from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from election_sim.config import load_config, dump_resolved_config
from election_sim.llm.key_manager import load_keys_from_env, load_global_key_state, RateLimiter, RoundRobinKeyManager
from election_sim.llm.google_client import GoogleLLMClient
from election_sim.phase1.eval import run_phase1
from election_sim.utils.io import ensure_dir
from election_sim.utils.logging import setup_logger

def make_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def build_llm(cfg, run_dir: str, logger):
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
            network_retries=cfg.llm.network_retries,
            network_backoff_sec=cfg.llm.network_backoff_sec,
        keyman=keyman,
        logger=logger,
        trace_path=trace_path,
    )

def cmd_smoke_test(cfg_path: str):
    cfg = load_config(cfg_path)
    run_id = make_run_id()
    run_dir = os.path.join(cfg.project.run_dir_base, run_id)
    ensure_dir(run_dir)
    logger = setup_logger(run_dir)
    Path(os.path.join(run_dir, "config.resolved.yaml")).write_text(dump_resolved_config(cfg), encoding="utf-8")

    llm = build_llm(cfg, run_dir, logger)
    out = llm.invoke("Reply with exactly: OK", meta={"phase": "smoke-test"})
    print(out.strip())

def cmd_phase1(cfg_path: str):
    cfg = load_config(cfg_path)
    run_id = make_run_id()
    run_dir = os.path.join(cfg.project.run_dir_base, run_id)
    ensure_dir(run_dir)
    logger = setup_logger(run_dir)
    Path(os.path.join(run_dir, "config.resolved.yaml")).write_text(dump_resolved_config(cfg), encoding="utf-8")

    llm = build_llm(cfg, run_dir, logger)
    result = run_phase1(cfg, llm, run_dir, logger)
    print(f"Run complete: {run_dir}")

def main():
    p = argparse.ArgumentParser(prog="election_sim")
    p.add_argument("command", choices=["smoke-test", "phase1"])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    if args.command == "smoke-test":
        cmd_smoke_test(args.config)
    elif args.command == "phase1":
        cmd_phase1(args.config)
    else:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
