from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from election_sim.phase3.voters import Voter, deterministic_vote, generate_voters, trait_band
from election_sim.utils.io import read_json, write_json
from election_sim.utils.progress import ProgressBar


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


def _extract_json_obj(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Model output does not contain a JSON object")


def _decode_json_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except Exception:
        return value.replace('\\"', '"').replace("\\n", "\n").strip()


def _extract_profiles_loose(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []
    cleaned_lines: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("```"):
            continue
        cleaned_lines.append(ln)
    cleaned = "\n".join(cleaned_lines).strip()

    # Strict JSON path first.
    try:
        obj = _extract_json_obj(cleaned)
        rows = obj.get("profiles", [])
        if isinstance(rows, list):
            out: list[dict[str, str]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                c = str(row.get("china", "")).strip()
                h = str(row.get("healthcare", "")).strip()
                g = str(row.get("guns", "")).strip()
                if c and h and g:
                    out.append({"china": c, "healthcare": h, "guns": g})
            if out:
                return out
    except Exception:
        pass

    # Tolerant regex path for partially-truncated JSON output.
    pattern = re.compile(
        r'\{[^{}]*"china"\s*:\s*"((?:\\.|[^"\\])*)"[^{}]*"healthcare"\s*:\s*"((?:\\.|[^"\\])*)"[^{}]*"guns"\s*:\s*"((?:\\.|[^"\\])*)"[^{}]*\}',
        flags=re.DOTALL,
    )
    out: list[dict[str, str]] = []
    for m in pattern.finditer(cleaned):
        c = _decode_json_string(m.group(1)).strip()
        h = _decode_json_string(m.group(2)).strip()
        g = _decode_json_string(m.group(3)).strip()
        if c and h and g:
            out.append({"china": c, "healthcare": h, "guns": g})
    return out


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


def build_debate_digest(transcript: Any, max_chars: int = 320) -> dict[str, Any]:
    records = transcript if isinstance(transcript, list) else []
    qa = [r for r in records if isinstance(r, dict) and r.get("type") == "qa"]
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for rec in qa:
        topic = str(rec.get("topic", "unknown"))
        by_topic.setdefault(topic, []).append(rec)

    digest_topics: list[dict[str, Any]] = []
    for topic, items in sorted(by_topic.items()):
        top_items = items[:5]
        compact: list[dict[str, Any]] = []
        for r in top_items:
            compact.append(
                {
                    "question": str(r.get("question", ""))[:max_chars],
                    "first_candidate": r.get("first_candidate"),
                    "second_candidate": r.get("second_candidate"),
                    "first_answer": str(r.get("first_answer", ""))[:max_chars],
                    "second_answer": str(r.get("second_answer", ""))[:max_chars],
                    "follow_up": str(r.get("follow_up", ""))[:max_chars],
                }
            )
        digest_topics.append({"topic": topic, "n_items": len(items), "sample": compact})

    return {
        "n_records": len(records),
        "n_qa": len(qa),
        "topics": digest_topics,
    }


def _value_pool_fallback(pool_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    china_templates = [
        "America should use alliances and targeted economic pressure to compete with China.",
        "The US should prioritize industrial self-reliance and stronger deterrence toward China.",
        "Avoid military escalation with China and focus on strategic diplomacy.",
        "Economic competition with China should protect workers while keeping global stability.",
        "National security should dominate US policy toward China's rising influence.",
    ]
    health_templates = [
        "Healthcare should be affordable and broadly accessible, with cost controls.",
        "A market-driven healthcare system with lower regulation is preferable.",
        "Protecting pre-existing conditions is a red line for me.",
        "Public-private balance in healthcare is better than ideological extremes.",
        "Healthcare spending should be tied to measurable outcomes and prevention.",
    ]
    gun_templates = [
        "Responsible ownership with stricter background checks is my preferred approach.",
        "Second Amendment rights are central and broad restrictions should be avoided.",
        "Community safety requires stronger enforcement against illegal gun use.",
        "I support safe storage rules and training for firearm ownership.",
        "Personal self-defense rights should remain protected with practical safeguards.",
    ]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    i = 0
    while len(out) < pool_size:
        c = china_templates[(i + rng.randint(0, 4)) % len(china_templates)]
        h = health_templates[(i + rng.randint(0, 4)) % len(health_templates)]
        g = gun_templates[(i + rng.randint(0, 4)) % len(gun_templates)]
        key = (c, h, g)
        if key not in seen:
            seen.add(key)
            out.append({"profile_id": f"vp_{len(out):03d}", "china": c, "healthcare": h, "guns": g})
        i += 1
        if i > pool_size * 20 and len(out) < pool_size:
            # Ensure completion by allowing controlled suffix variants.
            suffix = len(out)
            out.append(
                {
                    "profile_id": f"vp_{len(out):03d}",
                    "china": f"{c} Variant {suffix}",
                    "healthcare": f"{h} Variant {suffix}",
                    "guns": f"{g} Variant {suffix}",
                }
            )
    return out


def generate_value_pool(
    cfg,
    llm,
    debate_digest: dict[str, Any],
    logger,
    seed: int,
) -> dict[str, Any]:
    pool_size = int(getattr(cfg.phase3.values, "pool_size", 200) or 200)
    batch_size = int(getattr(cfg.phase3, "llm_value_pool_batch_size", 25) or 25)
    batch_size = max(5, min(batch_size, pool_size))
    max_llm_batches = min(6, max(1, ((pool_size + batch_size - 1) // batch_size) + 1))
    rng = random.Random(seed)

    pool: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    llm_calls = 0
    llm_failures = 0
    logger.info(
        "Phase3 value-pool generation started | target=%d batch_size=%d max_llm_batches=%d",
        pool_size,
        batch_size,
        max_llm_batches,
    )
    with ProgressBar(max(1, pool_size), "Phase3 Values", logger=logger) as pbar:
        consecutive_empty_batches = 0
        consecutive_failures = 0
        while len(pool) < pool_size and llm_calls < max_llm_batches:
            need = min(batch_size, pool_size - len(pool))
            prompt = (
                "Generate diverse US voter value profiles for election simulation.\n"
                f"Return exactly {need} profiles.\n"
                "Each profile must contain text values for: china, healthcare, guns.\n"
                "Output JSON only with schema: {\"profiles\":[{\"china\":...,\"healthcare\":...,\"guns\":...}]}\n"
                "Profiles should be ideologically diverse and non-duplicate.\n"
                f"Context digest: {json.dumps(debate_digest, ensure_ascii=False)}\n"
            )
            llm_calls += 1
            before = len(pool)
            try:
                raw = llm.invoke(prompt, meta={"phase": "phase3", "step": "value_pool_gen", "batch": llm_calls})
                rows = _extract_profiles_loose(raw)
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    c = str(row.get("china", "")).strip()
                    h = str(row.get("healthcare", "")).strip()
                    g = str(row.get("guns", "")).strip()
                    key = (c, h, g)
                    if not c or not h or not g or key in seen:
                        continue
                    seen.add(key)
                    pool.append(
                        {
                            "profile_id": f"vp_{len(pool):03d}",
                            "china": c,
                            "healthcare": h,
                            "guns": g,
                        }
                    )
                    if len(pool) >= pool_size:
                        break
                added = len(pool) - before
                if added > 0:
                    pbar.update(added, detail=f"llm batch {llm_calls}")
                    consecutive_empty_batches = 0
                    consecutive_failures = 0
                else:
                    consecutive_empty_batches += 1
                logger.info(
                    "Phase3 value-pool batch %d | requested=%d parsed=%d added=%d total=%d",
                    llm_calls,
                    need,
                    len(rows),
                    added,
                    len(pool),
                )
                if consecutive_empty_batches >= 2 and len(pool) > 0:
                    logger.info(
                        "Phase3 value-pool early-stop: %d consecutive empty batches; switching to fallback.",
                        consecutive_empty_batches,
                    )
                    break
            except Exception as err:
                llm_failures += 1
                consecutive_failures += 1
                logger.warning("Phase3 value pool batch failed (%d): %s", llm_calls, err)
                if consecutive_failures >= 2:
                    logger.info(
                        "Phase3 value-pool early-stop: %d consecutive failures; switching to fallback.",
                        consecutive_failures,
                    )
                    break

        if len(pool) < pool_size:
            logger.info(
                "Phase3 value-pool fallback enabled | missing=%d seed=%d",
                pool_size - len(pool),
                seed,
            )
            fallback = _value_pool_fallback(pool_size=pool_size, seed=seed + rng.randint(0, 9999))
            for row in fallback:
                key = (row["china"], row["healthcare"], row["guns"])
                if key in seen:
                    continue
                seen.add(key)
                pool.append(
                    {
                        "profile_id": f"vp_{len(pool):03d}",
                        "china": row["china"],
                        "healthcare": row["healthcare"],
                        "guns": row["guns"],
                    }
                )
                pbar.update(1, detail="fallback")
                if len(pool) >= pool_size:
                    break

    diversity_ratio = len(seen) / max(1, len(pool))
    logger.info(
        "Phase3 value-pool generation finished | size=%d llm_calls=%d llm_failures=%d diversity_ratio=%.3f",
        len(pool),
        llm_calls,
        llm_failures,
        diversity_ratio,
    )
    return {
        "metadata": {
            "pool_size": len(pool),
            "target_pool_size": pool_size,
            "llm_calls": llm_calls,
            "llm_failures": llm_failures,
            "diversity_ratio": diversity_ratio,
        },
        "profiles": pool[:pool_size],
    }


def vote_voters(
    voters: list[Voter],
    llm,
    debate_digest: dict[str, Any],
    candidate_profiles: dict[str, Any],
    logger,
    *,
    use_llm: bool = True,
    progress_desc: str = "Phase3 Vote",
    meta_prefix: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    meta_prefix = meta_prefix or {}
    with ProgressBar(max(1, len(voters)), progress_desc, logger=logger) as pbar:
        for idx, voter in enumerate(voters):
            meta = dict(meta_prefix)
            meta.update({"phase": "phase3", "voter_index": idx})
            if use_llm:
                ballot = voter.vote(llm, debate_digest, candidate_profiles, meta)
            else:
                det = deterministic_vote(voter, candidate_profiles)
                ballot = {
                    "voter_id": voter.voter_id,
                    "traits": voter.traits,
                    "values": voter.values,
                    "value_profile_id": voter.value_profile_id,
                    "source_scenario": voter.source_scenario,
                    "choice": det["choice"],
                    "confidence": det["confidence"],
                    "reason": det["reason"],
                }
            results.append(ballot)
            logger.info(
                "%s ballot | voter_index=%d choice=%s confidence=%s",
                progress_desc,
                idx,
                ballot.get("choice"),
                ballot.get("confidence"),
            )
            pbar.update(1, detail=f"voter {idx+1}")
    return results


def summarize_votes(votes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(votes)
    dem = sum(1 for v in votes if v.get("choice") == "democrat")
    rep = sum(1 for v in votes if v.get("choice") == "republican")
    undecided = total - dem - rep
    winner = "democrat" if dem > rep else ("republican" if rep > dem else "tie")
    avg_conf = (sum(int(v.get("confidence", 0)) for v in votes) / total) if total else 0.0

    by_wisdom_band = {"low": {"democrat": 0, "republican": 0}, "medium": {"democrat": 0, "republican": 0}, "high": {"democrat": 0, "republican": 0}}
    by_fear_band = {"low": {"democrat": 0, "republican": 0}, "medium": {"democrat": 0, "republican": 0}, "high": {"democrat": 0, "republican": 0}}

    for v in votes:
        choice = v.get("choice")
        if choice not in {"democrat", "republican"}:
            continue
        traits = v.get("traits", {}) if isinstance(v.get("traits"), dict) else {}
        wb = trait_band(int(traits.get("wisdom", 5)))
        fb = trait_band(int(traits.get("fear", 5)))
        by_wisdom_band[wb][choice] += 1
        by_fear_band[fb][choice] += 1

    return {
        "total_votes": total,
        "counts": {"democrat": dem, "republican": rep, "undecided": undecided},
        "percentages": {
            "democrat": (dem / total * 100) if total else 0.0,
            "republican": (rep / total * 100) if total else 0.0,
            "undecided": (undecided / total * 100) if total else 0.0,
        },
        "winner": winner,
        "margin_votes": abs(dem - rep),
        "avg_confidence": avg_conf,
        "breakdown": {
            "wisdom_band": by_wisdom_band,
            "fear_band": by_fear_band,
        },
    }


def build_sample_voter_analysis(votes: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    sample = votes[: min(8, len(votes))]
    high_conf = sorted(votes, key=lambda v: int(v.get("confidence", 0)), reverse=True)[:5]
    low_conf = sorted(votes, key=lambda v: int(v.get("confidence", 0)))[:5]
    return {
        "summary": {
            "winner": summary.get("winner"),
            "margin_votes": summary.get("margin_votes"),
            "avg_confidence": summary.get("avg_confidence"),
        },
        "sample_votes": sample,
        "high_confidence_examples": high_conf,
        "low_confidence_examples": low_conf,
    }


def run_phase3(cfg, llm, run_dir: Path, logger) -> dict[str, Any]:
    transcript_file = _find_transcript(cfg, run_dir)
    logger.info("Phase3 started | run_dir=%s transcript_source=%s", run_dir, transcript_file)
    with ProgressBar(5, "Phase3 Pipeline", logger=logger) as phase3_bar:
        target = run_dir / Path(cfg.phase3.debate_path)
        if transcript_file != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(transcript_file.read_text(encoding="utf-8"), encoding="utf-8")
            transcript_file = target
            logger.info("Phase3 transcript copied into run dir: %s", transcript_file)
        phase3_bar.update(1, detail="transcript-ready")

        transcript = read_json(transcript_file)
        digest_max = int(getattr(cfg.phase3, "digest_max_chars_per_field", 320) or 320)
        debate_digest = build_debate_digest(transcript, max_chars=digest_max)

        value_pool = generate_value_pool(
            cfg,
            llm,
            debate_digest=debate_digest,
            logger=logger,
            seed=cfg.project.random_seed,
        )
        value_pool_path = run_dir / cfg.phase3.values.output_pool_json
        value_pool_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(value_pool_path, value_pool)
        logger.info("Phase3 value pool saved: %s", value_pool_path)
        phase3_bar.update(1, detail="value-pool")

        raw_dist = getattr(cfg.phase3.voters, "trait_distributions", {}) or {}
        dist_dict = {k: (v.model_dump() if hasattr(v, "model_dump") else dict(v)) for k, v in raw_dist.items()}
        voters = generate_voters(
            count=cfg.phase3.voters.count,
            trait_names=list(cfg.phase3.voters.traits),
            seed=cfg.project.random_seed,
            trait_distributions=dist_dict,
            value_pool=value_pool.get("profiles", []),
            assignment_mode=cfg.phase3.values.assignment_mode,
        )
        logger.info("Phase3 voting started | voters=%d", len(voters))

        voters_path = run_dir / cfg.phase3.voters.output_voters_json
        voters_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            voters_path,
            [
                {
                    "voter_id": v.voter_id,
                    "traits": v.traits,
                    "values": v.values,
                    "value_profile_id": v.value_profile_id,
                    "source_scenario": v.source_scenario,
                }
                for v in voters
            ],
        )
        logger.info("Phase3 voters saved: %s", voters_path)
        phase3_bar.update(1, detail="voters")

        candidate_profiles = _candidate_profiles_from_cfg(cfg)
        votes = vote_voters(
            voters,
            llm=llm,
            debate_digest=debate_digest if getattr(cfg.phase3, "use_debate_digest", True) else transcript,
            candidate_profiles=candidate_profiles,
            logger=logger,
            use_llm=True,
            progress_desc="Phase3 Vote",
        )

        votes_path = run_dir / cfg.phase3.voters.output_votes_json
        votes_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(votes_path, votes)
        logger.info("Phase3 votes saved: %s", votes_path)
        phase3_bar.update(1, detail="votes")

        summary = summarize_votes(votes)
        summary_path = run_dir / cfg.phase3.voters.output_summary_json
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(summary_path, summary)
        logger.info("Phase3 summary saved: %s", summary_path)

        sample = build_sample_voter_analysis(votes, summary)
        sample_path = run_dir / cfg.phase3.voters.output_sample_analysis_json
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(sample_path, sample)
        logger.info("Phase3 sample analysis saved: %s", sample_path)
        logger.info("Phase3 finished | count=%d winner=%s", len(votes), summary.get("winner"))
        phase3_bar.update(1, detail="summary")

    return {
        "votes_path": str(votes_path),
        "voters_path": str(voters_path),
        "value_pool_path": str(value_pool_path),
        "summary_path": str(summary_path),
        "sample_analysis_path": str(sample_path),
        "count": len(votes),
        "summary": summary,
    }
