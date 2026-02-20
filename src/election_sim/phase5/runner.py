from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Literal

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


REPORT_PACK_VERSION = "1.0"
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


def _scenario_cfg(cfg, scenario_id: str) -> dict[str, Any]:
    raw = getattr(cfg.phase5, "scenarios", []) or []
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


def _phase5_path_for_scenario(cfg, run_dir: Path, scenario_id: str) -> Path:
    prefix = str(getattr(cfg.phase5, "output_scenario_prefix", "phase5/"))
    if prefix.endswith("/"):
        rel = f"{prefix}{scenario_id}.json"
    elif prefix.endswith("_"):
        rel = f"{prefix}{scenario_id}.json"
    else:
        rel = f"{prefix}/{scenario_id}.json"
    return run_dir / rel


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


def _distribution(low: float, medium: float, high: float) -> dict[str, float]:
    return {"low": low, "medium": medium, "high": high}


def _coerce_distribution(raw: dict[str, Any], default: dict[str, float]) -> dict[str, float]:
    return {
        "low": float(raw.get("low", default["low"])),
        "medium": float(raw.get("medium", default["medium"])),
        "high": float(raw.get("high", default["high"])),
    }


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
        unique_assignment_when_possible=bool(
            getattr(cfg.phase3.values, "unique_assignment_when_possible", True)
        ),
    )


def _tokenize(text: str) -> set[str]:
    return {t for t in text.lower().replace(".", " ").replace(",", " ").split() if len(t) > 3}


def _value_alignment_score(values: dict[str, Any], policy_stances: dict[str, Any]) -> float:
    score = 0.0
    for topic in ("china", "healthcare", "guns"):
        v_text = str(values.get(topic, ""))
        p_text = str(policy_stances.get(topic, ""))
        if not v_text or not p_text:
            continue
        overlap = len(_tokenize(v_text) & _tokenize(p_text))
        score += min(2.0, overlap * 0.3)
    return score


def _select_value_pool_by_lean(
    value_pool: list[dict[str, Any]],
    candidate_profiles: dict[str, Any],
    *,
    leaning: Literal["democrat", "republican", "balanced"],
    keep_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    if not value_pool:
        return value_pool
    keep_ratio = max(0.1, min(1.0, float(keep_ratio)))
    target = max(8, int(round(len(value_pool) * keep_ratio)))
    if leaning == "balanced":
        rng = random.Random(seed)
        if target >= len(value_pool):
            return list(value_pool)
        idxs = rng.sample(range(len(value_pool)), target)
        return [value_pool[i] for i in idxs]

    dem_s = candidate_profiles.get("democrat", {}).get("policy_stances", {})
    rep_s = candidate_profiles.get("republican", {}).get("policy_stances", {})
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in value_pool:
        vals = {
            "china": str(row.get("china", "")),
            "healthcare": str(row.get("healthcare", "")),
            "guns": str(row.get("guns", "")),
        }
        dem_score = _value_alignment_score(vals, dem_s)
        rep_score = _value_alignment_score(vals, rep_s)
        bias = rep_score - dem_score
        scored.append((bias, row))

    scored.sort(key=lambda x: x[0], reverse=(leaning == "republican"))
    trimmed = [row for _, row in scored[:target]]
    return trimmed if trimmed else list(value_pool)


def _build_healthcare_shock_pool(value_pool: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if not value_pool:
        return []
    rng = random.Random(seed)
    healthcare_templates = [
        "My top priority is affordable healthcare and guaranteed coverage for pre-existing conditions.",
        "Healthcare cost reduction matters most, including transparent pricing and broad access.",
        "I care most about lowering family medical bills while protecting essential coverage.",
        "Healthcare access, cost control, and pre-existing condition protections should come first.",
        "I prioritize practical healthcare reform that cuts costs and preserves continuity of care.",
    ]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(value_pool):
        new_row = dict(row)
        new_row["profile_id"] = f"{row.get('profile_id', f'vp_{i:03d}')}_hc"
        new_row["china"] = "Foreign policy is important, but household economic stability is more urgent."
        new_row["healthcare"] = healthcare_templates[rng.randrange(len(healthcare_templates))]
        new_row["guns"] = "I support responsible ownership with practical safety rules and lawful enforcement."
        out.append(new_row)
    return out


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
    llm=None,
    debate_digest: dict[str, Any] | None = None,
    logger=None,
    use_llm: bool = False,
    progress_label: str = "Phase5 Sim",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    dem_wins = 0
    rep_wins = 0
    tie = 0
    margin_pct: list[float] = []
    total_repeats = max(1, int(repeats))
    if use_llm and (llm is None or debate_digest is None or logger is None):
        raise ValueError("LLM simulation requires llm, debate_digest, and logger")
    with ProgressBar(total_repeats, f"{progress_label} Repeats", logger=logger) as rbar:
        for i in range(total_repeats):
            if use_llm:
                votes = vote_voters(
                    voters,
                    llm=llm,
                    debate_digest=debate_digest,
                    candidate_profiles=candidate_profiles,
                    logger=logger,
                    use_llm=True,
                    progress_desc=f"{progress_label} Vote R{i+1}/{total_repeats}",
                    meta_prefix={"phase": "phase5", "step": "scenario_vote", "repeat": i},
                )
                summary = summarize_votes(votes)
            else:
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
            rbar.update(1, detail=f"repeat {i+1}/{total_repeats}")
    return {
        "repeats": total_repeats,
        "win_rates": {
            "democrat": dem_wins / total_repeats,
            "republican": rep_wins / total_repeats,
            "tie": tie / total_repeats,
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
            meta_prefix={"phase": "phase5", "step": "llm_validation"},
        )
        return {"sample_size": len(sample), "summary": summarize_votes(votes)}
    except Exception as err:
        logger.warning("Phase5 LLM validation failed (%s): %s", label, err)
        return None


def _sensitivity_confidence(
    voters: list[Voter],
    candidate_profiles: dict[str, Any],
    *,
    repeats: int,
    seed: int,
    shift: float,
) -> dict[str, Any]:
    shift = abs(float(shift))
    dem_base = dict(candidate_profiles["democrat"]["traits"])
    rep_base = dict(candidate_profiles["republican"]["traits"])

    def _mk(dem_delta: float, rep_delta: float) -> dict[str, Any]:
        dem_t = dict(dem_base)
        rep_t = dict(rep_base)
        dem_t["confidence"] = max(0.0, min(10.0, float(dem_t.get("confidence", 5)) + dem_delta))
        rep_t["confidence"] = max(0.0, min(10.0, float(rep_t.get("confidence", 5)) + rep_delta))
        profiles = {
            "democrat": {**candidate_profiles["democrat"], "traits": dem_t},
            "republican": {**candidate_profiles["republican"], "traits": rep_t},
        }
        return _simulate_monte_carlo(voters, profiles, repeats=repeats, seed=seed)

    base = _mk(0.0, 0.0)
    dem_up = _mk(shift, -shift)
    rep_up = _mk(-shift, shift)
    return {
        "confidence_shift": shift,
        "base": {
            "winner": max(base["win_rates"], key=base["win_rates"].get),
            "avg_margin_pct_dem_minus_rep": base["avg_margin_pct_dem_minus_rep"],
            "win_rates": base["win_rates"],
        },
        "pro_dem_shock": {
            "winner": max(dem_up["win_rates"], key=dem_up["win_rates"].get),
            "avg_margin_pct_dem_minus_rep": dem_up["avg_margin_pct_dem_minus_rep"],
            "win_rates": dem_up["win_rates"],
        },
        "pro_rep_shock": {
            "winner": max(rep_up["win_rates"], key=rep_up["win_rates"].get),
            "avg_margin_pct_dem_minus_rep": rep_up["avg_margin_pct_dem_minus_rep"],
            "win_rates": rep_up["win_rates"],
        },
    }


def _scenario_payload(
    scenario_id: str,
    description: str,
    voters: list[Voter],
    simulation: dict[str, Any],
    *,
    candidate_profiles: dict[str, Any],
    assumptions: dict[str, Any],
    llm_validation: dict[str, Any] | None,
    sensitivity: dict[str, Any],
    expected_winner: str,
) -> dict[str, Any]:
    estimated_winner = max(simulation["win_rates"], key=simulation["win_rates"].get)
    return {
        "scenario_id": scenario_id,
        "description": description,
        "voter_count": len(voters),
        "expected_winner": expected_winner,
        "estimated_winner": estimated_winner,
        "expectation_met": estimated_winner == expected_winner,
        "candidate_profiles": candidate_profiles,
        "simulation": simulation,
        "defensibility": {
            "llm_validation": llm_validation,
            "sensitivity": sensitivity,
        },
        "assumptions": assumptions,
    }


def _scenario6_republican_win(
    cfg,
    value_pool: list[dict[str, Any]],
    *,
    seed: int,
    repeats: int,
    sensitivity_shift: float,
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], list[Voter], dict[str, Any]]:
    profiles = _candidate_profiles_from_cfg(cfg)
    dem_traits = {
        "honesty": 3.0,
        "aggressiveness": 6.0,
        "confidence": 5.2,
        "charisma": 5.0,
    }
    rep_traits = {
        "honesty": 6.2,
        "aggressiveness": 7.2,
        "confidence": 9.2,
        "charisma": 9.0,
    }
    dem_traits.update(dict(overrides.get("democrat_traits", {}) or {}))
    rep_traits.update(dict(overrides.get("republican_traits", {}) or {}))
    profiles["democrat"]["traits"].update({k: float(v) for k, v in dem_traits.items()})
    profiles["republican"]["traits"].update({k: float(v) for k, v in rep_traits.items()})

    dist = {
        "wisdom": _coerce_distribution(
            dict(overrides.get("wisdom_distribution", {}) or {}),
            _distribution(55, 30, 15),
        ),
        "fear": _coerce_distribution(
            dict(overrides.get("fear_distribution", {}) or {}),
            _distribution(10, 30, 60),
        ),
        "anger": _coerce_distribution(
            dict(overrides.get("anger_distribution", {}) or {}),
            _distribution(15, 25, 60),
        ),
        "distrust": _coerce_distribution(
            dict(overrides.get("distrust_distribution", {}) or {}),
            _distribution(55, 30, 15),
        ),
    }
    keep_ratio = float(_get_override(overrides, "value_pool_keep_ratio", 0.8))
    republican_pool = _select_value_pool_by_lean(
        value_pool,
        profiles,
        leaning="republican",
        keep_ratio=keep_ratio,
        seed=seed + 600,
    )
    voters = _scenario_voters(
        cfg,
        republican_pool,
        seed=seed + 601,
        source_scenario="phase5_scenario_6",
        distribution_overrides=dist,
    )
    sim = _simulate_monte_carlo(voters, profiles, repeats=repeats, seed=seed + 602)
    sensitivity = _sensitivity_confidence(
        voters,
        profiles,
        repeats=min(5, repeats),
        seed=seed + 603,
        shift=sensitivity_shift,
    )
    assumptions = {
        "goal": "Construct a scenario where Republican wins with high probability.",
        "value_pool_filter": {
            "leaning": "republican",
            "keep_ratio": keep_ratio,
            "selected_profiles": len(republican_pool),
        },
        "trait_distribution_overrides": dist,
    }
    payload = _scenario_payload(
        "scenario_6_republican_win",
        "Republican victory stress-test with fear/anger-heavy electorate.",
        voters,
        sim,
        candidate_profiles=profiles,
        assumptions=assumptions,
        llm_validation=None,
        sensitivity=sensitivity,
        expected_winner="republican",
    )
    return payload, voters, profiles


def _scenario7_healthcare_shock(
    cfg,
    value_pool: list[dict[str, Any]],
    *,
    seed: int,
    repeats: int,
    sensitivity_shift: float,
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], list[Voter], dict[str, Any]]:
    profiles = _candidate_profiles_from_cfg(cfg)
    dem_traits = {
        "honesty": 8.8,
        "aggressiveness": 3.0,
        "confidence": 7.6,
        "charisma": 6.8,
    }
    rep_traits = {
        "honesty": 5.0,
        "aggressiveness": 6.0,
        "confidence": 8.0,
        "charisma": 7.4,
    }
    dem_traits.update(dict(overrides.get("democrat_traits", {}) or {}))
    rep_traits.update(dict(overrides.get("republican_traits", {}) or {}))
    profiles["democrat"]["traits"].update({k: float(v) for k, v in dem_traits.items()})
    profiles["republican"]["traits"].update({k: float(v) for k, v in rep_traits.items()})

    dist = {
        "wisdom": _coerce_distribution(
            dict(overrides.get("wisdom_distribution", {}) or {}),
            _distribution(20, 55, 25),
        ),
        "fear": _coerce_distribution(
            dict(overrides.get("fear_distribution", {}) or {}),
            _distribution(20, 60, 20),
        ),
        "anger": _coerce_distribution(
            dict(overrides.get("anger_distribution", {}) or {}),
            _distribution(40, 45, 15),
        ),
        "distrust": _coerce_distribution(
            dict(overrides.get("distrust_distribution", {}) or {}),
            _distribution(40, 40, 20),
        ),
        "adaptability": _coerce_distribution(
            dict(overrides.get("adaptability_distribution", {}) or {}),
            _distribution(10, 20, 70),
        ),
    }
    healthcare_pool = _build_healthcare_shock_pool(value_pool, seed=seed + 700)
    voters = _scenario_voters(
        cfg,
        healthcare_pool,
        seed=seed + 701,
        source_scenario="phase5_scenario_7_healthcare_shock",
        distribution_overrides=dist,
    )
    sim = _simulate_monte_carlo(voters, profiles, repeats=repeats, seed=seed + 702)
    sensitivity = _sensitivity_confidence(
        voters,
        profiles,
        repeats=min(5, repeats),
        seed=seed + 703,
        shift=sensitivity_shift,
    )
    baseline_voters = _scenario_voters(
        cfg,
        value_pool,
        seed=seed + 701,
        source_scenario="phase5_scenario_7_healthcare_baseline",
        distribution_overrides=dist,
    )
    baseline_sim = _simulate_monte_carlo(
        baseline_voters,
        profiles,
        repeats=repeats,
        seed=seed + 704,
    )
    assumptions = {
        "goal": "Model healthcare-cost shock where policy issue salience shifts voter priorities.",
        "value_pool_override": {
            "mode": "healthcare_shock_templates",
            "selected_profiles": len(healthcare_pool),
        },
        "trait_distribution_overrides": dist,
    }
    payload = _scenario_payload(
        "scenario_7_healthcare_shock",
        "Healthcare-cost shock with issue-salience shift toward coverage and affordability.",
        voters,
        sim,
        candidate_profiles=profiles,
        assumptions=assumptions,
        llm_validation=None,
        sensitivity=sensitivity,
        expected_winner="democrat",
    )
    payload["defensibility"]["counterfactual_baseline"] = {
        "avg_margin_pct_dem_minus_rep": float(baseline_sim["avg_margin_pct_dem_minus_rep"]),
        "estimated_winner": max(baseline_sim["win_rates"], key=baseline_sim["win_rates"].get),
        "delta_margin_vs_baseline": float(sim["avg_margin_pct_dem_minus_rep"])
        - float(baseline_sim["avg_margin_pct_dem_minus_rep"]),
    }
    return payload, voters, profiles


def _scenario8_polarized_tossup(
    cfg,
    value_pool: list[dict[str, Any]],
    *,
    seed: int,
    repeats: int,
    sensitivity_shift: float,
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], list[Voter], dict[str, Any]]:
    profiles = _candidate_profiles_from_cfg(cfg)
    dem_traits = {
        "honesty": 6.6,
        "aggressiveness": 4.4,
        "confidence": 7.6,
        "charisma": 7.6,
    }
    rep_traits = {
        "honesty": 5.8,
        "aggressiveness": 6.0,
        "confidence": 7.9,
        "charisma": 8.0,
    }
    dem_traits.update(dict(overrides.get("democrat_traits", {}) or {}))
    rep_traits.update(dict(overrides.get("republican_traits", {}) or {}))
    profiles["democrat"]["traits"].update({k: float(v) for k, v in dem_traits.items()})
    profiles["republican"]["traits"].update({k: float(v) for k, v in rep_traits.items()})

    dist = {
        "wisdom": _coerce_distribution(
            dict(overrides.get("wisdom_distribution", {}) or {}),
            _distribution(45, 10, 45),
        ),
        "fear": _coerce_distribution(
            dict(overrides.get("fear_distribution", {}) or {}),
            _distribution(45, 10, 45),
        ),
        "anger": _coerce_distribution(
            dict(overrides.get("anger_distribution", {}) or {}),
            _distribution(45, 10, 45),
        ),
        "distrust": _coerce_distribution(
            dict(overrides.get("distrust_distribution", {}) or {}),
            _distribution(45, 10, 45),
        ),
    }
    keep_ratio = float(_get_override(overrides, "value_pool_keep_ratio", 1.0))
    polarized_pool = _select_value_pool_by_lean(
        value_pool,
        profiles,
        leaning="balanced",
        keep_ratio=keep_ratio,
        seed=seed + 800,
    )
    voters = _scenario_voters(
        cfg,
        polarized_pool,
        seed=seed + 801,
        source_scenario="phase5_scenario_8",
        distribution_overrides=dist,
    )
    sim = _simulate_monte_carlo(voters, profiles, repeats=repeats, seed=seed + 802)
    sensitivity = _sensitivity_confidence(
        voters,
        profiles,
        repeats=min(5, repeats),
        seed=seed + 803,
        shift=sensitivity_shift,
    )
    assumptions = {
        "goal": "Produce a close, polarized race suitable for robustness checks.",
        "value_pool_filter": {
            "leaning": "balanced",
            "keep_ratio": keep_ratio,
            "selected_profiles": len(polarized_pool),
        },
        "trait_distribution_overrides": dist,
    }
    payload = _scenario_payload(
        "scenario_8_polarized_tossup",
        "Polarized near-tossup electorate with high volatility.",
        voters,
        sim,
        candidate_profiles=profiles,
        assumptions=assumptions,
        llm_validation=None,
        sensitivity=sensitivity,
        expected_winner="tie",
    )
    return payload, voters, profiles


def _validate_report_pack_schema(pack: dict[str, Any]) -> None:
    missing = sorted(REPORT_PACK_REQUIRED_KEYS - set(pack.keys()))
    if missing:
        raise ValueError(f"phase5 report_pack missing required keys: {missing}")
    scenarios = pack.get("scenario_outputs", {})
    if not isinstance(scenarios, dict):
        raise ValueError("phase5 report_pack.scenario_outputs must be a dict")
    for sid in [
        "scenario_6_republican_win",
        "scenario_7_healthcare_shock",
        "scenario_8_polarized_tossup",
    ]:
        if sid not in scenarios:
            raise ValueError(f"phase5 report_pack.scenario_outputs missing {sid}")


def run_phase5(
    cfg,
    llm,
    run_dir: Path,
    logger,
    scenario_ids: list[str] | None = None,
) -> dict[str, Any]:
    selected = set(scenario_ids or [])
    known_ids = {
        "scenario_6_republican_win",
        "scenario_7_healthcare_shock",
        "scenario_8_polarized_tossup",
    }
    unknown = sorted(selected - known_ids)
    if unknown:
        logger.warning("Phase5 unknown scenario ids ignored: %s", ",".join(unknown))

    logger.info(
        "Phase5 started | run_dir=%s repeats=%s llm_validation=%s",
        run_dir,
        getattr(cfg.phase5, "repeats", 10),
        getattr(cfg.phase5, "enable_llm_validation", True),
    )
    if selected:
        logger.info("Phase5 scenario filter active: %s", ",".join(sorted(selected & known_ids)))
    run_dir.mkdir(parents=True, exist_ok=True)

    transcript = None
    digest = {"n_records": 0, "n_qa": 0, "topics": []}
    try:
        transcript_path = _find_transcript(cfg, run_dir)
        transcript = read_json(transcript_path)
        digest = build_debate_digest(
            transcript,
            max_chars=int(getattr(cfg.phase3, "digest_max_chars_per_field", 320) or 320),
        )
        logger.info("Phase5 transcript loaded from: %s", transcript_path)
    except Exception as err:
        logger.warning("Phase5 transcript unavailable; LLM validation may be skipped: %s", err)

    phase3_values_path = run_dir / cfg.phase3.values.output_pool_json
    if not phase3_values_path.exists():
        logger.info("Phase5 requires phase3 value pool; running phase3 bootstrap.")
        run_phase3(cfg, llm, run_dir, logger)

    value_pool_obj = _safe_read_json(phase3_values_path)
    if not isinstance(value_pool_obj, dict) or not isinstance(value_pool_obj.get("profiles"), list):
        logger.info("Phase5 value pool missing/invalid; regenerating value pool.")
        value_pool_obj = generate_value_pool(
            cfg,
            llm,
            debate_digest=digest,
            logger=logger,
            seed=cfg.project.random_seed + 5000,
        )
        phase3_values_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(phase3_values_path, value_pool_obj)
    value_pool = list(value_pool_obj.get("profiles", []))
    logger.info("Phase5 using value pool size=%d from %s", len(value_pool), phase3_values_path)

    repeats = int(getattr(cfg.phase5, "repeats", 10) or 10)
    seed = int(getattr(cfg.project, "random_seed", 0) or 0)
    global_vote_mode = str(getattr(cfg.phase5, "scenario_vote_mode", "deterministic") or "deterministic").lower()
    if global_vote_mode not in {"deterministic", "llm_full"}:
        logger.warning("Phase5 invalid scenario_vote_mode=%s; fallback=deterministic", global_vote_mode)
        global_vote_mode = "deterministic"
    if global_vote_mode == "llm_full":
        logger.warning("Phase5 scenario_vote_mode=llm_full: full-population LLM voting will be expensive.")
    sample_size = int(getattr(cfg.phase5, "llm_validation_sample_size", 24) or 24)
    do_llm_validation = bool(getattr(cfg.phase5, "enable_llm_validation", True)) and transcript is not None
    sensitivity_shift = float(getattr(cfg.phase5, "confidence_shift_sensitivity", 8.0) or 8.0)

    scenarios_out: dict[str, Any] = {}

    definitions = [
        ("scenario_6_republican_win", _scenario6_republican_win),
        ("scenario_7_healthcare_shock", _scenario7_healthcare_shock),
        ("scenario_8_polarized_tossup", _scenario8_polarized_tossup),
    ]

    def _is_selected(sid: str) -> bool:
        return (not selected) or (sid in selected)

    def _scenario_vote_mode(overrides: dict[str, Any]) -> str:
        mode = str(_get_override(overrides, "vote_mode", global_vote_mode) or global_vote_mode).lower()
        if mode not in {"deterministic", "llm_full"}:
            return global_vote_mode
        return mode

    with ProgressBar(len(definitions), "Phase5 Scenarios", logger=logger) as phase5_bar:
        for idx, (sid, fn) in enumerate(definitions, start=1):
            scfg = _scenario_cfg(cfg, sid)
            sid_vote_mode = _scenario_vote_mode(scfg.get("overrides", {}))
            if not _is_selected(sid):
                payload = {"scenario_id": sid, "description": sid, "skipped": True, "skip_reason": "not_selected"}
                scenarios_out[sid] = payload
                write_json(_phase5_path_for_scenario(cfg, run_dir, sid), payload)
                logger.info("Phase5 %s skipped (not selected).", sid)
                phase5_bar.update(1, detail=sid)
                continue
            if not scfg["enabled"]:
                payload = {
                    "scenario_id": sid,
                    "description": sid,
                    "skipped": True,
                    "skip_reason": "disabled_in_config",
                }
                scenarios_out[sid] = payload
                write_json(_phase5_path_for_scenario(cfg, run_dir, sid), payload)
                logger.info("Phase5 %s skipped by config.", sid)
                phase5_bar.update(1, detail=sid)
                continue

            payload, voters, profiles = fn(
                cfg,
                value_pool,
                seed=seed + idx * 100,
                repeats=repeats,
                sensitivity_shift=sensitivity_shift,
                overrides=scfg["overrides"],
            )
            if sid_vote_mode == "llm_full":
                sim = _simulate_monte_carlo(
                    voters,
                    profiles,
                    repeats=repeats,
                    seed=seed + idx * 1000 + 17,
                    llm=llm,
                    debate_digest=digest,
                    logger=logger,
                    use_llm=True,
                    progress_label=f"Phase5 {sid}",
                )
                payload["simulation"] = sim
                payload["estimated_winner"] = max(sim["win_rates"], key=sim["win_rates"].get)
                expected = str(payload.get("expected_winner", "")).lower()
                if expected in {"democrat", "republican"}:
                    payload["expectation_met"] = payload["estimated_winner"] == expected
                base_cf = payload.get("defensibility", {}).get("counterfactual_baseline")
                if isinstance(base_cf, dict) and "avg_margin_pct_dem_minus_rep" in base_cf:
                    base_margin = float(base_cf.get("avg_margin_pct_dem_minus_rep", 0.0))
                    base_cf["delta_margin_vs_baseline"] = (
                        float(sim.get("avg_margin_pct_dem_minus_rep", 0.0)) - base_margin
                    )
                payload["simulation_mode"] = "llm_full"
            else:
                payload["simulation_mode"] = "deterministic"

            if do_llm_validation:
                if sid_vote_mode == "llm_full":
                    payload.setdefault("defensibility", {})["llm_validation"] = {
                        "skipped": True,
                        "reason": "full_llm_simulation_enabled",
                    }
                else:
                    llm_val = _llm_validation(
                        llm,
                        voters,
                        digest,
                        profiles,
                        sample_size=sample_size,
                        seed=seed + idx * 211,
                        logger=logger,
                        label=f"Phase5 {sid} Validate",
                    )
                    payload["defensibility"]["llm_validation"] = llm_val
            scenarios_out[sid] = payload
            write_json(_phase5_path_for_scenario(cfg, run_dir, sid), payload)
            logger.info(
                "Phase5 %s complete | mode=%s estimated_winner=%s rep_win_rate=%.3f dem_win_rate=%.3f avg_margin=%.3f",
                sid,
                payload.get("simulation_mode", "deterministic"),
                payload.get("estimated_winner"),
                float(payload.get("simulation", {}).get("win_rates", {}).get("republican", 0.0)),
                float(payload.get("simulation", {}).get("win_rates", {}).get("democrat", 0.0)),
                float(payload.get("simulation", {}).get("avg_margin_pct_dem_minus_rep", 0.0)),
            )
            phase5_bar.update(1, detail=sid)

    def _metrics(payload: dict[str, Any]) -> dict[str, Any]:
        sim = payload.get("simulation", {})
        return {
            "estimated_winner": payload.get("estimated_winner"),
            "expected_winner": payload.get("expected_winner"),
            "expectation_met": payload.get("expectation_met"),
            "avg_margin_pct_dem_minus_rep": float(sim.get("avg_margin_pct_dem_minus_rep", 0.0)),
            "democrat_win_rate": float(sim.get("win_rates", {}).get("democrat", 0.0)),
            "republican_win_rate": float(sim.get("win_rates", {}).get("republican", 0.0)),
        }

    threshold = float(getattr(cfg.phase5, "margin_pct_alert_threshold", 5.0) or 5.0)
    tossup_flag = False
    s8 = scenarios_out.get("scenario_8_polarized_tossup", {})
    if isinstance(s8, dict) and not s8.get("skipped"):
        s8_margin = abs(float(s8.get("simulation", {}).get("avg_margin_pct_dem_minus_rep", 0.0)))
        tossup_flag = s8_margin <= threshold
        s8["expected_winner"] = "tossup"
        s8["expectation_met"] = tossup_flag

    key_metrics_table = {sid: _metrics(payload) for sid, payload in scenarios_out.items() if not payload.get("skipped")}
    sorted_by_margin = sorted(
        [(sid, row.get("avg_margin_pct_dem_minus_rep", 0.0)) for sid, row in key_metrics_table.items()],
        key=lambda x: x[1],
    )

    comparison = {
        "ranking_by_margin_dem_minus_rep": sorted_by_margin,
        "scenario_6_vs_7_margin_shift": (
            float(key_metrics_table.get("scenario_6_republican_win", {}).get("avg_margin_pct_dem_minus_rep", 0.0))
            - float(key_metrics_table.get("scenario_7_healthcare_shock", {}).get("avg_margin_pct_dem_minus_rep", 0.0))
        ),
        "scenario_8_tossup_flag": tossup_flag,
        "margin_pct_alert_threshold": threshold,
        "notes": [
            "Negative margin means Republican advantage; positive margin means Democrat advantage.",
            "Tossup flag checks whether absolute average margin is below configured threshold.",
        ],
    }
    comparison_path = run_dir / cfg.phase5.output_comparison_json
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(comparison_path, comparison)
    logger.info("Phase5 comparison saved: %s", comparison_path)

    report_pack = {
        "version": REPORT_PACK_VERSION,
        "generated_at_ts": time.time(),
        "run_dir": str(run_dir),
        "metadata": {
            "run_id": run_dir.name,
            "repeats": repeats,
            "scenario_vote_mode": global_vote_mode,
            "llm_validation_enabled": do_llm_validation,
            "llm_validation_sample_size": sample_size,
            "confidence_shift_sensitivity": sensitivity_shift,
            "seed": seed,
        },
        "assumptions": [
            "Phase5 scenario simulation mode is configurable: deterministic or llm_full.",
            "Each scenario includes confidence-shift sensitivity checks for robustness.",
            "LLM validation is sampled and optional (auto-skipped when llm_full is active).",
        ],
        "phase_artifacts": {
            "phase3_value_pool": _safe_read_json(phase3_values_path),
            "phase3_summary": _safe_read_json(run_dir / cfg.phase3.voters.output_summary_json),
            "phase4_report_pack": _safe_read_json(run_dir / cfg.phase4.output_report_pack_json),
        },
        "scenario_outputs": scenarios_out,
        "comparison": comparison,
        "key_metrics_table": key_metrics_table,
        "output_paths": {
            "comparison": str(comparison_path),
            "phase3_value_pool": str(phase3_values_path),
        },
        "limitations": [
            "Scenario outcomes depend on deterministic scoring assumptions in fallback voting logic.",
            "LLM validation uses a sample, not full-population simulation.",
            "Trait distributions are independent draws and do not model latent socio-demographic factors.",
        ],
        "reporting_prompt_template": (
            "Use this JSON to produce a rigorous Persian report for Phase 5: explain scenario design, "
            "quantitative outcomes, sensitivity checks, defensibility, and policy implications."
        ),
    }
    _validate_report_pack_schema(report_pack)
    report_path = run_dir / cfg.phase5.output_report_pack_json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report_pack)
    logger.info("Phase5 report pack saved: %s", report_path)
    logger.info("Phase5 finished | scenarios=%s", ",".join(sorted(scenarios_out.keys())))

    return {
        "scenarios": list(scenarios_out.keys()),
        "comparison_path": str(comparison_path),
        "report_pack_path": str(report_path),
    }
