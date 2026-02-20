from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from election_sim.phase3.voters import Voter, deterministic_vote, generate_voters
from election_sim.phase3.voting import (
    build_debate_digest,
    generate_value_pool,
    run_phase3,
    summarize_votes,
    vote_voters,
)
from election_sim.utils.io import read_json, write_json
from election_sim.utils.progress import ProgressBar


REPORT_PACK_VERSION = "1.1"
REPORT_PACK_REQUIRED_KEYS = {
    "version",
    "generated_at_ts",
    "run_dir",
    "phase_artifacts",
    "scenario_outputs",
    "comparison",
    "key_metrics_table",
    "assumptions",
    "limitations",
}


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def _scenario_cfg(cfg, scenario_id: str) -> dict[str, Any]:
    raw = getattr(cfg.phase4, "scenarios", []) or []
    for item in raw:
        if isinstance(item, dict):
            sid = item.get("id")
            enabled = bool(item.get("enabled", True))
            overrides = dict(item.get("overrides", {}) or {})
        else:
            sid = getattr(item, "id", None)
            enabled = bool(getattr(item, "enabled", True))
            overrides = dict(getattr(item, "overrides", {}) or {})
        if sid == scenario_id:
            return {"enabled": enabled, "overrides": overrides}
    return {"enabled": True, "overrides": {}}


def _get_override(overrides: dict[str, Any], key: str, default: Any) -> Any:
    if key not in overrides:
        return default
    return overrides[key]


def _validate_report_pack_schema(pack: dict[str, Any]) -> None:
    missing = sorted(REPORT_PACK_REQUIRED_KEYS - set(pack.keys()))
    if missing:
        raise ValueError(f"report_pack missing required keys: {missing}")
    scenarios = pack.get("scenario_outputs", {})
    if not isinstance(scenarios, dict):
        raise ValueError("report_pack.scenario_outputs must be a dict")
    for sid in ["scenario_1", "scenario_2", "scenario_3", "scenario_4", "scenario_5"]:
        if sid not in scenarios:
            raise ValueError(f"report_pack.scenario_outputs missing {sid}")


def _phase4_path_for_scenario(cfg, run_dir: Path, scenario_id: str) -> Path:
    prefix = str(getattr(cfg.phase4, "output_scenario_prefix", "phase4/scenario_"))
    if scenario_id.startswith("scenario_") and prefix.endswith("_"):
        suffix = scenario_id.split("scenario_", 1)[1]
        rel = f"{prefix}{suffix}.json"
    else:
        rel = f"phase4/{scenario_id}.json"
    return run_dir / rel


def _find_transcript(cfg, run_dir: Path) -> Path:
    rel = Path(cfg.phase3.debate_path)
    p1 = run_dir / rel
    if p1.exists():
        return p1
    p2 = Path.cwd() / rel
    if p2.exists():
        return p2
    base = Path(cfg.project.run_dir_base)
    if base.exists():
        candidates: list[Path] = []
        for child in base.iterdir():
            if not child.is_dir():
                continue
            tp = child / rel
            if tp.exists():
                candidates.append(tp)
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"Missing transcript: {p1}")


def _candidate_profiles_from_cfg(cfg) -> dict[str, Any]:
    return {
        "democrat": {
            "display_name": cfg.phase1.candidates.democrat.display_name,
            "party": cfg.phase1.candidates.democrat.party,
            "traits": dict(cfg.phase1.candidates.democrat.personal_traits),
            "policy_stances": dict(cfg.phase1.candidates.democrat.policy_stances),
        },
        "republican": {
            "display_name": cfg.phase1.candidates.republican.display_name,
            "party": cfg.phase1.candidates.republican.party,
            "traits": dict(cfg.phase1.candidates.republican.personal_traits),
            "policy_stances": dict(cfg.phase1.candidates.republican.policy_stances),
        },
    }


def _level(mapping: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(mapping.get(key, default))
    except Exception:
        return default


def _scenario_base_profiles(cfg, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides = overrides or {}
    out = _candidate_profiles_from_cfg(cfg)
    m = dict(getattr(cfg.phase4, "trait_level_mapping", {}) or {})
    out["republican"]["traits"].update(
        {
            "honesty": _level(m, "liar", 2.0),
            "aggressiveness": _level(m, "aggressive", 8.5),
            "confidence": _level(m, "high", 8.0),
            "charisma": _level(m, "very_high", 9.5),
        }
    )
    out["democrat"]["traits"].update(
        {
            "honesty": _level(m, "honest", 9.0),
            "aggressiveness": _level(m, "calm", 2.0),
            "confidence": _level(m, "high", 8.0),
            "charisma": _level(m, "moderate", 6.0),
        }
    )
    dem_override = overrides.get("democrat_traits", {})
    rep_override = overrides.get("republican_traits", {})
    if isinstance(dem_override, dict):
        for k, v in dem_override.items():
            try:
                out["democrat"]["traits"][str(k)] = float(v)
            except Exception:
                continue
    if isinstance(rep_override, dict):
        for k, v in rep_override.items():
            try:
                out["republican"]["traits"][str(k)] = float(v)
            except Exception:
                continue
    return out


def _distribution(low: float, medium: float, high: float) -> dict[str, float]:
    return {"low": low, "medium": medium, "high": high}


def _scenario_voters(
    cfg,
    value_pool: list[dict[str, Any]],
    *,
    seed: int,
    source_scenario: str,
    distribution_overrides: dict[str, dict[str, float]],
) -> list[Voter]:
    raw_dist = getattr(cfg.phase3.voters, "trait_distributions", {}) or {}
    dist_dict = {k: (v.model_dump() if hasattr(v, "model_dump") else dict(v)) for k, v in raw_dist.items()}
    for k, v in distribution_overrides.items():
        dist_dict[k] = dict(v)
    return generate_voters(
        count=cfg.phase3.voters.count,
        trait_names=list(cfg.phase3.voters.traits),
        seed=seed,
        trait_distributions=dist_dict,
        value_pool=value_pool,
        assignment_mode=cfg.phase3.values.assignment_mode,
        source_scenario=source_scenario,
    )


def _simulate_repeat(
    voters: list[Voter],
    candidate_profiles: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    ballots: list[dict[str, Any]] = []
    for v in voters:
        det = deterministic_vote(v, candidate_profiles)
        choice = det["choice"]
        conf = int(det["confidence"])
        upset_prob = max(0.02, (100 - conf) / 130.0)
        if rng.random() < upset_prob:
            choice = "republican" if choice == "democrat" else "democrat"
            conf = max(50, 100 - conf)
        ballots.append(
            {
                "voter_id": v.voter_id,
                "traits": v.traits,
                "values": v.values,
                "value_profile_id": v.value_profile_id,
                "source_scenario": v.source_scenario,
                "choice": choice,
                "confidence": conf,
                "reason": "deterministic_monte_carlo",
            }
        )
    return summarize_votes(ballots)


def _simulate_monte_carlo(
    voters: list[Voter],
    candidate_profiles: dict[str, Any],
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    dem_wins = 0
    rep_wins = 0
    tie = 0
    margin_pct: list[float] = []
    for i in range(max(1, repeats)):
        summary = _simulate_repeat(voters, candidate_profiles, seed=seed + i * 97)
        rows.append(summary)
        w = summary.get("winner")
        if w == "democrat":
            dem_wins += 1
        elif w == "republican":
            rep_wins += 1
        else:
            tie += 1
        dem_p = float(summary.get("percentages", {}).get("democrat", 0.0))
        rep_p = float(summary.get("percentages", {}).get("republican", 0.0))
        margin_pct.append(dem_p - rep_p)
    return {
        "repeats": max(1, repeats),
        "win_rates": {
            "democrat": dem_wins / max(1, repeats),
            "republican": rep_wins / max(1, repeats),
            "tie": tie / max(1, repeats),
        },
        "avg_margin_pct_dem_minus_rep": (sum(margin_pct) / len(margin_pct)) if margin_pct else 0.0,
        "repeat_summaries": rows,
    }


def _llm_validation(
    llm,
    voters: list[Voter],
    debate_digest: dict[str, Any],
    candidate_profiles: dict[str, Any],
    *,
    sample_size: int,
    seed: int,
    logger,
    label: str,
) -> dict[str, Any] | None:
    try:
        rng = random.Random(seed)
        if len(voters) <= sample_size:
            sample = voters
        else:
            idxs = rng.sample(range(len(voters)), sample_size)
            sample = [voters[i] for i in idxs]
        votes = vote_voters(
            sample,
            llm=llm,
            debate_digest=debate_digest,
            candidate_profiles=candidate_profiles,
            logger=logger,
            use_llm=True,
            progress_desc=label,
            meta_prefix={"phase": "phase4", "step": "llm_validation"},
        )
        return {"sample_size": len(sample), "summary": summarize_votes(votes)}
    except Exception as err:
        logger.warning("LLM validation failed (%s): %s", label, err)
        return None


def _scenario_result_payload(
    scenario_id: str,
    description: str,
    voters: list[Voter],
    sim: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "scenario_id": scenario_id,
        "description": description,
        "voter_count": len(voters),
        "simulation": sim,
    }
    if extra:
        result.update(extra)
    return result


def _scenario1(
    cfg,
    value_pool: list[dict[str, Any]],
    seed: int,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Voter], dict[str, Any]]:
    overrides = overrides or {}
    profiles = _scenario_base_profiles(cfg, overrides=overrides)
    wisdom_dist = _get_override(overrides, "wisdom_distribution", {"low": 60, "medium": 30, "high": 10})
    fear_dist = _get_override(overrides, "fear_distribution", {"low": 30, "medium": 40, "high": 30})
    voters = _scenario_voters(
        cfg,
        value_pool,
        seed=seed + 11,
        source_scenario="scenario_1",
        distribution_overrides={
            "wisdom": {
                "low": float(wisdom_dist.get("low", 60)),
                "medium": float(wisdom_dist.get("medium", 30)),
                "high": float(wisdom_dist.get("high", 10)),
            },
            "fear": {
                "low": float(fear_dist.get("low", 30)),
                "medium": float(fear_dist.get("medium", 40)),
                "high": float(fear_dist.get("high", 30)),
            },
        },
    )
    return profiles, voters, {
        "candidate_profile_assumption": "Republican liar+aggressive+high confidence+very high charisma; Democrat honest+calm+high confidence+moderate charisma.",
        "distribution_overrides": {
            "wisdom": {
                "low": float(wisdom_dist.get("low", 60)),
                "medium": float(wisdom_dist.get("medium", 30)),
                "high": float(wisdom_dist.get("high", 10)),
            },
            "fear": {
                "low": float(fear_dist.get("low", 30)),
                "medium": float(fear_dist.get("medium", 40)),
                "high": float(fear_dist.get("high", 30)),
            },
        },
    }


def _scenario2(
    cfg,
    value_pool: list[dict[str, Any]],
    seed: int,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Voter], dict[str, Any]]:
    overrides = overrides or {}
    profiles = _scenario_base_profiles(cfg, overrides=overrides)
    wisdom_dist = _get_override(overrides, "wisdom_distribution", {"low": 10, "medium": 20, "high": 70})
    fear_dist = _get_override(overrides, "fear_distribution", {"low": 30, "medium": 40, "high": 30})
    voters = _scenario_voters(
        cfg,
        value_pool,
        seed=seed + 22,
        source_scenario="scenario_2",
        distribution_overrides={
            "wisdom": {
                "low": float(wisdom_dist.get("low", 10)),
                "medium": float(wisdom_dist.get("medium", 20)),
                "high": float(wisdom_dist.get("high", 70)),
            },
            "fear": {
                "low": float(fear_dist.get("low", 30)),
                "medium": float(fear_dist.get("medium", 40)),
                "high": float(fear_dist.get("high", 30)),
            },
        },
    )
    return profiles, voters, {
        "distribution_overrides": {
            "wisdom": {
                "low": float(wisdom_dist.get("low", 10)),
                "medium": float(wisdom_dist.get("medium", 20)),
                "high": float(wisdom_dist.get("high", 70)),
            },
            "fear": {
                "low": float(fear_dist.get("low", 30)),
                "medium": float(fear_dist.get("medium", 40)),
                "high": float(fear_dist.get("high", 30)),
            },
        },
    }


def _scenario3(
    cfg,
    value_pool: list[dict[str, Any]],
    seed: int,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Voter], dict[str, Any]]:
    overrides = overrides or {}
    scenario2_overrides = dict(overrides.get("scenario2_base_overrides", {}) or {})
    profiles, voters, meta = _scenario2(cfg, value_pool, seed, overrides=scenario2_overrides)
    sample_profile = dict(_get_override(overrides, "sample_value_profile", {}) or {})
    if not sample_profile:
        sample_profile = dict(getattr(cfg.phase3.values, "sample_value_profile", {}) or {})
    forced_count_override = _get_override(overrides, "forced_high_wisdom_count", 100)
    try:
        forced_count = max(0, int(forced_count_override))
    except Exception:
        forced_count = 100
    high_wisdom = [i for i, v in enumerate(voters) if int(v.traits.get("wisdom", 0)) >= 7]
    rng = random.Random(seed + 333)
    picked_count = min(forced_count, len(high_wisdom))
    picked = rng.sample(high_wisdom, picked_count) if picked_count > 0 else []
    for idx in picked:
        voters[idx].values = {
            "china": str(sample_profile.get("china", "")),
            "healthcare": str(sample_profile.get("healthcare", "")),
            "guns": str(sample_profile.get("guns", "")),
        }
        voters[idx].value_profile_id = "sample_profile"
        voters[idx].source_scenario = "scenario_3_forced_values"
    meta["forced_value_profile"] = sample_profile
    meta["forced_high_wisdom_count"] = picked_count
    meta["forced_voter_ids"] = picked
    return profiles, voters, meta


def _scenario4_optimize(
    cfg,
    llm,
    debate_digest: dict[str, Any],
    value_pool: list[dict[str, Any]],
    seed: int,
    logger,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overrides = overrides or {}
    search_mode = str(
        _get_override(overrides, "search_mode", getattr(cfg.phase4, "search_mode", "hybrid") or "hybrid")
    ).lower()
    base_profiles, voters, base_meta = _scenario1(
        cfg,
        value_pool,
        seed,
        overrides=dict(overrides.get("scenario1_base_overrides", {}) or {}),
    )
    rep_fixed = dict(base_profiles["republican"]["traits"])
    dem_base = dict(base_profiles["democrat"]["traits"])
    repeats = int(_get_override(overrides, "repeats", getattr(cfg.phase4, "repeats", 10) or 10))
    threshold = float(
        _get_override(overrides, "certainty_threshold", getattr(cfg.phase4, "certainty_threshold", 0.9) or 0.9)
    )
    top_k = int(_get_override(overrides, "llm_validation_top_k", getattr(cfg.phase4, "llm_validation_top_k", 5) or 5))
    sample_size = int(
        _get_override(
            overrides,
            "llm_validation_sample_size",
            getattr(cfg.phase4, "llm_validation_sample_size", 24) or 24,
        )
    )
    grid = dict(overrides.get("grid", {}) or {})
    honesty_values = list(grid.get("honesty", [4, 6, 8, 10]))
    aggr_values = list(grid.get("aggressiveness", [1, 3, 5, 7]))
    conf_values = list(grid.get("confidence", [4, 6, 8, 10]))
    charisma_values = list(grid.get("charisma", [4, 6, 8, 10]))
    logger.info(
        "Phase4 scenario_4 optimization started | search_mode=%s grid=%dx%dx%dx%d",
        search_mode,
        len(honesty_values),
        len(aggr_values),
        len(conf_values),
        len(charisma_values),
    )

    stage_a: list[dict[str, Any]] = []
    total_grid = max(1, len(honesty_values) * len(aggr_values) * len(conf_values) * len(charisma_values))
    with ProgressBar(total_grid, "Phase4 S4 Search", logger=logger) as pbar:
        for honesty in honesty_values:
            for aggr in aggr_values:
                for conf in conf_values:
                    for charisma in charisma_values:
                        try:
                            honesty_f = float(honesty)
                            aggr_f = float(aggr)
                            conf_f = float(conf)
                            charisma_f = float(charisma)
                        except Exception:
                            pbar.update(1, detail="skip-invalid-grid")
                            continue
                        profiles = {
                            "democrat": {
                                **base_profiles["democrat"],
                                "traits": {
                                    **dem_base,
                                    "honesty": honesty_f,
                                    "aggressiveness": aggr_f,
                                    "confidence": conf_f,
                                    "charisma": charisma_f,
                                },
                            },
                            "republican": {
                                **base_profiles["republican"],
                                "traits": dict(rep_fixed),
                            },
                        }
                        sim = _simulate_monte_carlo(
                            voters,
                            profiles,
                            repeats=3,
                            seed=seed + int(honesty_f) * 17 + int(charisma_f),
                        )
                        stage_a.append(
                            {
                                "traits": profiles["democrat"]["traits"],
                                "avg_margin_pct_dem_minus_rep": sim["avg_margin_pct_dem_minus_rep"],
                                "win_rate_dem": sim["win_rates"]["democrat"],
                            }
                        )
                        pbar.update(1, detail=f"h{honesty_f} a{aggr_f} c{conf_f} ch{charisma_f}")

    stage_a_sorted = sorted(stage_a, key=lambda x: x["avg_margin_pct_dem_minus_rep"], reverse=True)
    if not stage_a_sorted:
        stage_a_sorted = [
            {
                "traits": dict(dem_base),
                "avg_margin_pct_dem_minus_rep": 0.0,
                "win_rate_dem": 0.0,
            }
        ]
    stage_a_top = stage_a_sorted[: max(1, top_k)]

    validate_pool: list[dict[str, Any]]
    if search_mode == "fast_approx":
        validate_pool = []
    elif search_mode == "full_llm":
        validate_pool = stage_a_sorted[: min(len(stage_a_sorted), max(1, top_k * 2))]
    else:
        validate_pool = stage_a_top

    stage_b: list[dict[str, Any]] = []
    with ProgressBar(max(1, len(validate_pool)), "Phase4 S4 Validate", logger=logger) as vbar:
        if not validate_pool:
            vbar.update(1, detail="skipped-fast-approx")
        for idx, cand in enumerate(validate_pool, start=1):
            profiles = {
                "democrat": {
                    **base_profiles["democrat"],
                    "traits": dict(cand["traits"]),
                },
                "republican": {
                    **base_profiles["republican"],
                    "traits": dict(rep_fixed),
                },
            }
            llm_val = _llm_validation(
                llm,
                voters,
                debate_digest,
                profiles,
                sample_size=sample_size,
                seed=seed + idx * 111,
                logger=logger,
                label=f"Phase4 S4 Validate {idx}",
            )
            llm_margin = None
            if llm_val:
                s = llm_val["summary"]
                llm_margin = float(s.get("percentages", {}).get("democrat", 0.0)) - float(
                    s.get("percentages", {}).get("republican", 0.0)
                )
            combined = float(cand["avg_margin_pct_dem_minus_rep"]) + (
                0.2 * llm_margin if llm_margin is not None else 0.0
            )
            stage_b.append(
                {
                    "traits": cand["traits"],
                    "stage_a_margin": cand["avg_margin_pct_dem_minus_rep"],
                    "llm_validation": llm_val,
                    "combined_score": combined,
                }
            )
            logger.info(
                "Phase4 scenario_4 validation candidate %d/%d | stage_a_margin=%.3f combined=%.3f",
                idx,
                len(validate_pool),
                float(cand["avg_margin_pct_dem_minus_rep"]),
                combined,
            )
            vbar.update(1, detail=f"candidate {idx}/{len(validate_pool)}")

    stage_b_sorted = sorted(stage_b, key=lambda x: x["combined_score"], reverse=True) if stage_b else []
    best = stage_b_sorted[0] if stage_b_sorted else {"traits": stage_a_top[0]["traits"]}

    best_profiles = {
        "democrat": {
            **base_profiles["democrat"],
            "traits": dict(best["traits"]),
        },
        "republican": {
            **base_profiles["republican"],
            "traits": dict(rep_fixed),
        },
    }
    final_sim = _simulate_monte_carlo(voters, best_profiles, repeats=repeats, seed=seed + 404)
    extra = {
        **base_meta,
        "optimization_goal": "Maximize democrat margin with certainty constraint.",
        "search_mode": search_mode,
        "certainty_threshold": threshold,
        "certainty_pass": final_sim["win_rates"]["democrat"] >= threshold,
        "best_democrat_traits": best_profiles["democrat"]["traits"],
        "best_combined_score": best.get("combined_score"),
    }
    trace = {"stage_a_top": stage_a_top, "stage_b": stage_b_sorted}
    logger.info(
        "Phase4 scenario_4 optimization finished | certainty_pass=%s dem_win_rate=%.3f avg_margin=%.3f",
        extra["certainty_pass"],
        final_sim["win_rates"]["democrat"],
        final_sim["avg_margin_pct_dem_minus_rep"],
    )
    return best_profiles, final_sim, {"meta": extra, "trace": trace}


def _scenario5_optimize(
    cfg,
    llm,
    debate_digest: dict[str, Any],
    value_pool: list[dict[str, Any]],
    seed: int,
    logger,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overrides = overrides or {}
    search_mode = str(
        _get_override(overrides, "search_mode", getattr(cfg.phase4, "search_mode", "hybrid") or "hybrid")
    ).lower()
    profiles, _, base_meta = _scenario1(
        cfg,
        value_pool,
        seed,
        overrides=dict(overrides.get("scenario1_base_overrides", {}) or {}),
    )
    repeats = int(_get_override(overrides, "repeats", getattr(cfg.phase4, "repeats", 10) or 10))
    threshold = float(
        _get_override(overrides, "certainty_threshold", getattr(cfg.phase4, "certainty_threshold", 0.9) or 0.9)
    )
    top_k = int(_get_override(overrides, "llm_validation_top_k", getattr(cfg.phase4, "llm_validation_top_k", 5) or 5))
    sample_size = int(
        _get_override(
            overrides,
            "llm_validation_sample_size",
            getattr(cfg.phase4, "llm_validation_sample_size", 24) or 24,
        )
    )

    space = dict(overrides.get("search_space", {}) or {})
    wisdom_high_values = list(space.get("wisdom_high_values", [30, 50, 70, 80]))
    fear_high_values = list(space.get("fear_high_values", [10, 20, 30]))
    distrust_high_values = list(space.get("distrust_high_values", [20, 40, 60]))
    wisdom_medium = float(space.get("wisdom_medium", 20))
    fear_medium = float(space.get("fear_medium", 40))
    distrust_medium = float(space.get("distrust_medium", 30))
    logger.info(
        "Phase4 scenario_5 optimization started | search_mode=%s grid=%dx%dx%d",
        search_mode,
        len(wisdom_high_values),
        len(fear_high_values),
        len(distrust_high_values),
    )

    candidates: list[dict[str, Any]] = []
    total_grid = max(1, len(wisdom_high_values) * len(fear_high_values) * len(distrust_high_values))
    with ProgressBar(total_grid, "Phase4 S5 Search", logger=logger) as pbar:
        for wisdom_high_raw in wisdom_high_values:
            for fear_high_raw in fear_high_values:
                for distrust_high_raw in distrust_high_values:
                    try:
                        wisdom_high = float(wisdom_high_raw)
                        fear_high = float(fear_high_raw)
                        distrust_high = float(distrust_high_raw)
                    except Exception:
                        pbar.update(1, detail="skip-invalid-grid")
                        continue
                    wisdom_low = max(0.0, 100.0 - wisdom_high - wisdom_medium)
                    fear_low = max(0.0, 100.0 - fear_high - fear_medium)
                    distrust_low = max(0.0, 100.0 - distrust_high - distrust_medium)
                    dist = {
                        "wisdom": _distribution(wisdom_low, wisdom_medium, wisdom_high),
                        "fear": _distribution(fear_low, fear_medium, fear_high),
                        "distrust": _distribution(distrust_low, distrust_medium, distrust_high),
                    }
                    voters = _scenario_voters(
                        cfg,
                        value_pool,
                        seed=seed + int(wisdom_high) * 3 + int(fear_high) * 5 + int(distrust_high) * 7,
                        source_scenario="scenario_5_candidate",
                        distribution_overrides=dist,
                    )
                    sim = _simulate_monte_carlo(
                        voters,
                        profiles,
                        repeats=3,
                        seed=seed + int(wisdom_high) + int(fear_high) + int(distrust_high),
                    )
                    candidates.append(
                        {
                            "distribution": dist,
                            "avg_margin_pct_dem_minus_rep": sim["avg_margin_pct_dem_minus_rep"],
                            "win_rate_dem": sim["win_rates"]["democrat"],
                        }
                    )
                    pbar.update(1, detail=f"w{wisdom_high} f{fear_high} d{distrust_high}")

    stage_a_sorted = sorted(candidates, key=lambda x: x["avg_margin_pct_dem_minus_rep"], reverse=True)
    if not stage_a_sorted:
        stage_a_sorted = [
            {
                "distribution": {
                    "wisdom": _distribution(60, 30, 10),
                    "fear": _distribution(30, 40, 30),
                    "distrust": _distribution(33, 34, 33),
                },
                "avg_margin_pct_dem_minus_rep": 0.0,
                "win_rate_dem": 0.0,
            }
        ]
    stage_a_top = stage_a_sorted[: max(1, top_k)]

    if search_mode == "fast_approx":
        validate_pool: list[dict[str, Any]] = []
    elif search_mode == "full_llm":
        validate_pool = stage_a_sorted[: min(len(stage_a_sorted), max(1, top_k * 2))]
    else:
        validate_pool = stage_a_top

    stage_b: list[dict[str, Any]] = []
    with ProgressBar(max(1, len(validate_pool)), "Phase4 S5 Validate", logger=logger) as vbar:
        if not validate_pool:
            vbar.update(1, detail="skipped-fast-approx")
        for idx, cand in enumerate(validate_pool, start=1):
            voters = _scenario_voters(
                cfg,
                value_pool,
                seed=seed + 900 + idx,
                source_scenario="scenario_5_validation",
                distribution_overrides=cand["distribution"],
            )
            llm_val = _llm_validation(
                llm,
                voters,
                debate_digest,
                profiles,
                sample_size=sample_size,
                seed=seed + 999 + idx,
                logger=logger,
                label=f"Phase4 S5 Validate {idx}",
            )
            llm_margin = None
            if llm_val:
                s = llm_val["summary"]
                llm_margin = float(s.get("percentages", {}).get("democrat", 0.0)) - float(
                    s.get("percentages", {}).get("republican", 0.0)
                )
            combined = float(cand["avg_margin_pct_dem_minus_rep"]) + (
                0.2 * llm_margin if llm_margin is not None else 0.0
            )
            stage_b.append(
                {
                    "distribution": cand["distribution"],
                    "stage_a_margin": cand["avg_margin_pct_dem_minus_rep"],
                    "llm_validation": llm_val,
                    "combined_score": combined,
                }
            )
            logger.info(
                "Phase4 scenario_5 validation candidate %d/%d | stage_a_margin=%.3f combined=%.3f",
                idx,
                len(validate_pool),
                float(cand["avg_margin_pct_dem_minus_rep"]),
                combined,
            )
            vbar.update(1, detail=f"candidate {idx}/{len(validate_pool)}")

    stage_b_sorted = sorted(stage_b, key=lambda x: x["combined_score"], reverse=True) if stage_b else []
    best = stage_b_sorted[0] if stage_b_sorted else stage_a_top[0]
    best_dist = best["distribution"]
    best_voters = _scenario_voters(
        cfg,
        value_pool,
        seed=seed + 505,
        source_scenario="scenario_5_best",
        distribution_overrides=best_dist,
    )
    final_sim = _simulate_monte_carlo(best_voters, profiles, repeats=repeats, seed=seed + 606)
    extra = {
        **base_meta,
        "optimization_goal": "Choose voter trait distributions for a democrat landslide.",
        "search_mode": search_mode,
        "certainty_threshold": threshold,
        "certainty_pass": final_sim["win_rates"]["democrat"] >= threshold,
        "best_distribution": best_dist,
        "best_combined_score": best.get("combined_score"),
    }
    trace = {"stage_a_top": stage_a_top, "stage_b": stage_b_sorted}
    logger.info(
        "Phase4 scenario_5 optimization finished | certainty_pass=%s dem_win_rate=%.3f avg_margin=%.3f",
        extra["certainty_pass"],
        final_sim["win_rates"]["democrat"],
        final_sim["avg_margin_pct_dem_minus_rep"],
    )
    return profiles, final_sim, {"meta": extra, "trace": trace, "best_voters": best_voters}


def run_phase4(cfg, llm, run_dir: Path, logger) -> dict[str, Any]:
    logger.info(
        "Phase4 started | run_dir=%s repeats=%s certainty_threshold=%s search_mode=%s",
        run_dir,
        getattr(cfg.phase4, "repeats", 10),
        getattr(cfg.phase4, "certainty_threshold", 0.9),
        getattr(cfg.phase4, "search_mode", "hybrid"),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = _find_transcript(cfg, run_dir)
    transcript = read_json(transcript_path)
    digest = build_debate_digest(
        transcript,
        max_chars=int(getattr(cfg.phase3, "digest_max_chars_per_field", 320) or 320),
    )

    phase3_summary_path = run_dir / cfg.phase3.voters.output_summary_json
    phase3_values_path = run_dir / cfg.phase3.values.output_pool_json
    if not phase3_summary_path.exists() or not phase3_values_path.exists():
        logger.info("Phase4 requires phase3 artifacts; running phase3 bootstrap.")
        run_phase3(cfg, llm, run_dir, logger)

    value_pool_obj = _safe_read_json(phase3_values_path)
    if not isinstance(value_pool_obj, dict) or not isinstance(value_pool_obj.get("profiles"), list):
        logger.info("Phase4 value pool missing/invalid; regenerating value pool.")
        value_pool_obj = generate_value_pool(
            cfg,
            llm,
            debate_digest=digest,
            logger=logger,
            seed=cfg.project.random_seed + 4000,
        )
        phase3_values_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(phase3_values_path, value_pool_obj)
    value_pool = list(value_pool_obj.get("profiles", []))
    logger.info("Phase4 using value pool size=%d from %s", len(value_pool), phase3_values_path)

    repeats = int(getattr(cfg.phase4, "repeats", 10) or 10)
    seed = int(getattr(cfg.project, "random_seed", 0) or 0)
    scenarios_out: dict[str, Any] = {}
    optimization_trace: dict[str, Any] = {}

    def _avg_party(sim: dict[str, Any], party: str) -> float:
        rows = sim.get("repeat_summaries", [])
        if not rows:
            return 0.0
        vals = [float(r.get("percentages", {}).get(party, 0.0)) for r in rows]
        return sum(vals) / len(vals)

    def _scenario_metrics(payload: dict[str, Any]) -> dict[str, Any]:
        sim = payload.get("simulation", {})
        return {
            "avg_democrat_pct": _avg_party(sim, "democrat"),
            "avg_republican_pct": _avg_party(sim, "republican"),
            "avg_margin_pct_dem_minus_rep": float(sim.get("avg_margin_pct_dem_minus_rep", 0.0)),
            "democrat_win_rate": float(sim.get("win_rates", {}).get("democrat", 0.0)),
            "republican_win_rate": float(sim.get("win_rates", {}).get("republican", 0.0)),
            "certainty_pass": payload.get("certainty_pass"),
        }

    descriptions = {
        "scenario_1": "Baseline S1",
        "scenario_2": "Wisdom-shifted S2",
        "scenario_3": "S2 + forced value profile for 100 high-wisdom voters",
        "scenario_4": "Optimize democrat core traits under S1 electorate",
        "scenario_5": "Optimize voter trait distributions for democrat landslide",
    }

    with ProgressBar(5, "Phase4 Scenarios", logger=logger) as phase4_bar:
        # Scenario 1
        s1_cfg = _scenario_cfg(cfg, "scenario_1")
        if s1_cfg["enabled"]:
            s1_profiles, s1_voters, s1_meta = _scenario1(cfg, value_pool, seed, overrides=s1_cfg["overrides"])
            s1_sim = _simulate_monte_carlo(s1_voters, s1_profiles, repeats=repeats, seed=seed + 1)
            s1 = _scenario_result_payload("scenario_1", descriptions["scenario_1"], s1_voters, s1_sim, s1_meta)
            s1["candidate_profiles"] = s1_profiles
            scenarios_out["scenario_1"] = s1
            logger.info(
                "Phase4 scenario_1 complete | dem_win_rate=%.3f avg_margin=%.3f",
                s1_sim["win_rates"]["democrat"],
                s1_sim["avg_margin_pct_dem_minus_rep"],
            )
        else:
            s1 = {"scenario_id": "scenario_1", "description": descriptions["scenario_1"], "skipped": True}
            scenarios_out["scenario_1"] = s1
            s1_voters = []
            s1_sim = {"repeat_summaries": [], "win_rates": {"democrat": 0.0, "republican": 0.0}}
            logger.info("Phase4 scenario_1 skipped by config.")
        write_json(_phase4_path_for_scenario(cfg, run_dir, "scenario_1"), scenarios_out["scenario_1"])
        phase4_bar.update(1, detail="scenario_1")

        # Scenario 2
        s2_cfg = _scenario_cfg(cfg, "scenario_2")
        if s2_cfg["enabled"]:
            s2_profiles, s2_voters, s2_meta = _scenario2(cfg, value_pool, seed, overrides=s2_cfg["overrides"])
            s2_sim = _simulate_monte_carlo(s2_voters, s2_profiles, repeats=repeats, seed=seed + 2)
            s2 = _scenario_result_payload("scenario_2", descriptions["scenario_2"], s2_voters, s2_sim, s2_meta)
            s2["candidate_profiles"] = s2_profiles
            scenarios_out["scenario_2"] = s2
            logger.info(
                "Phase4 scenario_2 complete | dem_win_rate=%.3f avg_margin=%.3f",
                s2_sim["win_rates"]["democrat"],
                s2_sim["avg_margin_pct_dem_minus_rep"],
            )
        else:
            s2 = {"scenario_id": "scenario_2", "description": descriptions["scenario_2"], "skipped": True}
            scenarios_out["scenario_2"] = s2
            s2_voters = []
            s2_sim = {"repeat_summaries": [], "win_rates": {"democrat": 0.0, "republican": 0.0}}
            logger.info("Phase4 scenario_2 skipped by config.")
        write_json(_phase4_path_for_scenario(cfg, run_dir, "scenario_2"), scenarios_out["scenario_2"])
        phase4_bar.update(1, detail="scenario_2")

        # Scenario 3
        s3_cfg = _scenario_cfg(cfg, "scenario_3")
        if s3_cfg["enabled"]:
            s3_profiles, s3_voters, s3_meta = _scenario3(cfg, value_pool, seed, overrides=s3_cfg["overrides"])
            s3_sim = _simulate_monte_carlo(s3_voters, s3_profiles, repeats=repeats, seed=seed + 3)
            s3 = _scenario_result_payload("scenario_3", descriptions["scenario_3"], s3_voters, s3_sim, s3_meta)
            s3["candidate_profiles"] = s3_profiles
            scenarios_out["scenario_3"] = s3
            logger.info(
                "Phase4 scenario_3 complete | forced_count=%s dem_win_rate=%.3f avg_margin=%.3f",
                s3_meta.get("forced_high_wisdom_count"),
                s3_sim["win_rates"]["democrat"],
                s3_sim["avg_margin_pct_dem_minus_rep"],
            )
        else:
            s3 = {"scenario_id": "scenario_3", "description": descriptions["scenario_3"], "skipped": True}
            scenarios_out["scenario_3"] = s3
            s3_voters = []
            s3_sim = {"repeat_summaries": [], "win_rates": {"democrat": 0.0, "republican": 0.0}}
            logger.info("Phase4 scenario_3 skipped by config.")
        write_json(_phase4_path_for_scenario(cfg, run_dir, "scenario_3"), scenarios_out["scenario_3"])
        phase4_bar.update(1, detail="scenario_3")

        # Scenario 4
        s4_cfg = _scenario_cfg(cfg, "scenario_4")
        if s4_cfg["enabled"]:
            s4_profiles, s4_sim, s4_extra = _scenario4_optimize(
                cfg,
                llm,
                digest,
                value_pool,
                seed,
                logger,
                overrides=s4_cfg["overrides"],
            )
            s4 = _scenario_result_payload(
                "scenario_4",
                descriptions["scenario_4"],
                s1_voters,
                s4_sim,
                s4_extra["meta"],
            )
            s4["best_candidate_profiles"] = s4_profiles
            scenarios_out["scenario_4"] = s4
            optimization_trace["scenario_4"] = s4_extra["trace"]
            logger.info(
                "Phase4 scenario_4 complete | dem_win_rate=%.3f avg_margin=%.3f certainty_pass=%s",
                s4_sim["win_rates"]["democrat"],
                s4_sim["avg_margin_pct_dem_minus_rep"],
                s4_extra["meta"].get("certainty_pass"),
            )
        else:
            s4 = {"scenario_id": "scenario_4", "description": descriptions["scenario_4"], "skipped": True}
            scenarios_out["scenario_4"] = s4
            optimization_trace["scenario_4"] = {"skipped": True}
            logger.info("Phase4 scenario_4 skipped by config.")
        write_json(_phase4_path_for_scenario(cfg, run_dir, "scenario_4"), scenarios_out["scenario_4"])
        phase4_bar.update(1, detail="scenario_4")

        # Scenario 5
        s5_cfg = _scenario_cfg(cfg, "scenario_5")
        if s5_cfg["enabled"]:
            s5_profiles, s5_sim, s5_extra = _scenario5_optimize(
                cfg,
                llm,
                digest,
                value_pool,
                seed,
                logger,
                overrides=s5_cfg["overrides"],
            )
            s5_voters = list(s5_extra.get("best_voters", []))
            s5 = _scenario_result_payload(
                "scenario_5",
                descriptions["scenario_5"],
                s5_voters,
                s5_sim,
                s5_extra["meta"],
            )
            s5["candidate_profiles"] = s5_profiles
            scenarios_out["scenario_5"] = s5
            optimization_trace["scenario_5"] = s5_extra["trace"]
            logger.info(
                "Phase4 scenario_5 complete | dem_win_rate=%.3f avg_margin=%.3f certainty_pass=%s",
                s5_sim["win_rates"]["democrat"],
                s5_sim["avg_margin_pct_dem_minus_rep"],
                s5_extra["meta"].get("certainty_pass"),
            )
        else:
            s5 = {"scenario_id": "scenario_5", "description": descriptions["scenario_5"], "skipped": True}
            scenarios_out["scenario_5"] = s5
            optimization_trace["scenario_5"] = {"skipped": True}
            logger.info("Phase4 scenario_5 skipped by config.")
        write_json(_phase4_path_for_scenario(cfg, run_dir, "scenario_5"), scenarios_out["scenario_5"])
        phase4_bar.update(1, detail="scenario_5")

    comparison = {
        "scenario_1_vs_2": {
            "avg_dem_pct_delta": _avg_party(s2_sim, "democrat") - _avg_party(s1_sim, "democrat"),
            "analysis": "Effect of higher wisdom distribution while keeping fear distribution fixed.",
        },
        "scenario_2_vs_3": {
            "avg_dem_pct_delta": _avg_party(s3_sim, "democrat") - _avg_party(s2_sim, "democrat"),
            "analysis": "Effect of forcing sample value profile on 100 high-wisdom voters.",
        },
    }
    comparison_path = run_dir / cfg.phase4.output_comparison_json
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(comparison_path, comparison)
    logger.info("Phase4 comparison saved: %s", comparison_path)

    trace_path = run_dir / cfg.phase4.output_optimization_trace_json
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(trace_path, optimization_trace)
    logger.info("Phase4 optimization trace saved: %s", trace_path)

    key_metrics_table = {sid: _scenario_metrics(payload) for sid, payload in scenarios_out.items()}
    report_pack = {
        "version": REPORT_PACK_VERSION,
        "generated_at_ts": time.time(),
        "run_dir": str(run_dir),
        "metadata": {
            "run_id": run_dir.name,
            "search_mode": getattr(cfg.phase4, "search_mode", "hybrid"),
            "repeats": repeats,
            "certainty_threshold": getattr(cfg.phase4, "certainty_threshold", 0.9),
            "seed": seed,
        },
        "assumptions": [
            "search_mode=hybrid unless overridden per scenario",
            "certainty_threshold>=0.90",
            "repeats=10 by default",
            "scenario_4 optimizes democrat core traits only",
            "scenario_3 selects high-wisdom voters via seeded random",
            "trait textual descriptors are mapped with config.trait_level_mapping",
        ],
        "phase_artifacts": {
            "phase1_summary": _safe_read_json(run_dir / "phase1/summary.json"),
            "phase2_summary": _safe_read_json(run_dir / "phase2/debate_summary.json"),
            "phase3_summary": _safe_read_json(run_dir / cfg.phase3.voters.output_summary_json),
            "phase3_sample_analysis": _safe_read_json(run_dir / cfg.phase3.voters.output_sample_analysis_json),
        },
        "scenario_outputs": scenarios_out,
        "comparison": comparison,
        "key_metrics_table": key_metrics_table,
        "output_paths": {
            "comparison": str(comparison_path),
            "optimization_trace": str(trace_path),
            "phase3_value_pool": str(phase3_values_path),
        },
        "limitations": [
            "Hybrid search uses deterministic Monte Carlo for broad exploration and limited LLM validation.",
            "Final narrative report is not generated in-code; report_pack.json is designed for downstream report generation.",
        ],
        "reporting_prompt_template": (
            "Use this JSON to produce a full Persian project report: describe setup, each phase outputs, "
            "scenario-by-scenario findings, comparisons, and conclusions with limitations."
        ),
    }
    _validate_report_pack_schema(report_pack)
    report_path = run_dir / cfg.phase4.output_report_pack_json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report_pack)
    logger.info("Phase4 report pack saved: %s", report_path)
    logger.info("Phase4 finished | scenarios=%s", ",".join(sorted(scenarios_out.keys())))

    return {
        "scenarios": list(scenarios_out.keys()),
        "comparison_path": str(comparison_path),
        "optimization_trace_path": str(trace_path),
        "report_pack_path": str(report_path),
    }
