from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from election_sim.config import load_config
from election_sim.llm.google_client import GoogleLLMClient
from election_sim.llm.key_manager import load_keys_from_env
from election_sim.phase1.eval import run_phase1
from election_sim.phase2.debate import run_debate
from election_sim.phase3.voting import run_phase3
from election_sim.phase4.runner import run_phase4
from election_sim.phase5.runner import run_phase5
from election_sim.utils.io import ensure_dir
from election_sim.utils.logging import setup_logger


def make_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _dump_resolved_config_fallback(cfg) -> str:
    try:
        from election_sim.config import dump_resolved_config  # type: ignore

        return dump_resolved_config(cfg)
    except Exception:
        try:
            import yaml

            data = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
            return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        except Exception:
            return repr(cfg)


def build_llm(cfg, run_dir: str, model_name: str | None = None) -> GoogleLLMClient:
    keys = cfg.llm.api_keys or load_keys_from_env()
    if not keys:
        raise RuntimeError(
            "No API keys found. Set GOOGLE_API_KEYS (comma-separated) or llm.api_keys in config.yaml"
        )

    # GoogleLLMClient uses requests_per_minute; derive it from min_interval_sec.
    min_interval = float(getattr(cfg.llm, "min_interval_sec", 1.0) or 1.0)
    rpm = max(1, int(60.0 / min_interval))
    trace_path = os.path.join(run_dir, cfg.logging.trace_jsonl)

    return GoogleLLMClient(
        api_keys=keys,
        model_name=(model_name or cfg.llm.model_name),
        temperature=cfg.llm.temperature,
        max_output_tokens=cfg.llm.max_output_tokens,
        trace_path=trace_path,
        requests_per_minute=rpm,
        retry_on_429=cfg.llm.retry_on_429,
        rounds_limit=cfg.llm.rounds_limit,
        cooloff_sec=cfg.llm.cooloff_sec,
        network_retries=cfg.llm.network_retries,
        network_backoff_sec=cfg.llm.network_backoff_sec,
        request_timeout_sec=getattr(cfg.llm, "request_timeout_sec", 45.0),
    )


def _init_run(cfg, run_id: str | None = None) -> tuple[str, object]:
    rid = run_id or make_run_id()
    run_dir = os.path.join(cfg.project.run_dir_base, rid)
    ensure_dir(run_dir)
    logger = setup_logger(run_dir, run_log_name=getattr(cfg.logging, "run_log", "run.log"))
    Path(os.path.join(run_dir, "config.resolved.yaml")).write_text(
        _dump_resolved_config_fallback(cfg), encoding="utf-8"
    )
    return run_dir, logger


def cmd_smoke_test(cfg_path: str, run_id: str | None = None) -> None:
    cfg = load_config(cfg_path)
    run_dir, logger = _init_run(cfg, run_id=run_id)
    logger.info("Command start: smoke-test | run_dir=%s", run_dir)
    llm = build_llm(cfg, run_dir)
    out = llm.invoke("Reply with exactly: OK", meta={"phase": "smoke-test"})
    logger.info("Command finish: smoke-test")
    print(out.strip())


def cmd_phase1(cfg_path: str, run_id: str | None = None) -> None:
    cfg = load_config(cfg_path)
    run_dir, logger = _init_run(cfg, run_id=run_id)
    logger.info("Command start: phase1 | run_dir=%s", run_dir)
    llm = build_llm(cfg, run_dir, model_name=cfg.llm.candidate_model_name or cfg.llm.model_name)
    run_phase1(cfg, llm, run_dir, logger)
    logger.info("Command finish: phase1")
    print(f"Run complete: {run_dir}")


def cmd_phase2(cfg_path: str, run_id: str | None = None) -> None:
    cfg = load_config(cfg_path)
    run_dir, logger = _init_run(cfg, run_id=run_id)
    logger.info("Command start: phase2 | run_dir=%s", run_dir)
    llm = build_llm(cfg, run_dir, model_name=cfg.llm.moderator_model_name or cfg.llm.model_name)
    run_debate(cfg, llm, run_dir, logger)
    logger.info("Command finish: phase2")
    print(f"Run complete: {run_dir}")


def cmd_phase3(cfg_path: str, run_id: str | None = None) -> None:
    cfg = load_config(cfg_path)
    run_dir, logger = _init_run(cfg, run_id=run_id)
    logger.info("Command start: phase3 | run_dir=%s", run_dir)
    llm = build_llm(cfg, run_dir, model_name=cfg.llm.voter_model_name or cfg.llm.model_name)
    run_phase3(cfg, llm, Path(run_dir), logger)
    logger.info("Command finish: phase3")
    print(f"Run complete: {run_dir}")


def cmd_phase4(cfg_path: str, run_id: str | None = None) -> None:
    cfg = load_config(cfg_path)
    run_dir, logger = _init_run(cfg, run_id=run_id)
    logger.info("Command start: phase4 | run_dir=%s", run_dir)
    llm = build_llm(cfg, run_dir, model_name=cfg.llm.voter_model_name or cfg.llm.model_name)
    run_phase4(cfg, llm, Path(run_dir), logger)
    logger.info("Command finish: phase4")
    print(f"Run complete: {run_dir}")


def cmd_all(cfg_path: str, run_id: str | None = None) -> None:
    cfg = load_config(cfg_path)
    run_dir, logger = _init_run(cfg, run_id=run_id)
    logger.info("Command start: all | run_dir=%s", run_dir)

    llm_phase1 = build_llm(cfg, run_dir, model_name=cfg.llm.candidate_model_name or cfg.llm.model_name)
    run_phase1(cfg, llm_phase1, run_dir, logger)

    llm_phase2 = build_llm(cfg, run_dir, model_name=cfg.llm.moderator_model_name or cfg.llm.model_name)
    run_debate(cfg, llm_phase2, run_dir, logger)

    llm_phase3 = build_llm(cfg, run_dir, model_name=cfg.llm.voter_model_name or cfg.llm.model_name)
    run_phase3(cfg, llm_phase3, Path(run_dir), logger)

    run_phase4(cfg, llm_phase3, Path(run_dir), logger)
    run_phase5(cfg, llm_phase3, Path(run_dir), logger)
    logger.info("Command finish: all")
    print(f"Run complete: {run_dir}")


def cmd_phase5(cfg_path: str, run_id: str | None = None) -> None:
    cfg = load_config(cfg_path)
    run_dir, logger = _init_run(cfg, run_id=run_id)
    logger.info("Command start: phase5 | run_dir=%s", run_dir)
    llm = build_llm(cfg, run_dir, model_name=cfg.llm.voter_model_name or cfg.llm.model_name)
    run_phase5(cfg, llm, Path(run_dir), logger)
    logger.info("Command finish: phase5")
    print(f"Run complete: {run_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="election_sim")
    p.add_argument(
        "command",
        choices=["smoke-test", "phase1", "phase2", "phase3", "phase4", "phase5", "all"],
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--run-id", default=None)
    args = p.parse_args(argv)

    if args.command == "smoke-test":
        cmd_smoke_test(args.config, args.run_id)
    elif args.command == "phase1":
        cmd_phase1(args.config, args.run_id)
    elif args.command == "phase2":
        cmd_phase2(args.config, args.run_id)
    elif args.command == "phase3":
        cmd_phase3(args.config, args.run_id)
    elif args.command == "phase4":
        cmd_phase4(args.config, args.run_id)
    elif args.command == "phase5":
        cmd_phase5(args.config, args.run_id)
    elif args.command == "all":
        cmd_all(args.config, args.run_id)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
