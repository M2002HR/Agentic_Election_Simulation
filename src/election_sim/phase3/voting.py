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


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").strip().split())


def _token_set(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Z0-9']+", (text or "").lower())
        if len(tok) >= 4
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _valid_value_text(text: str, *, min_words: int, min_tokens: int) -> bool:
    clean = _normalize_ws(text)
    if len(clean) < 40:
        return False
    if len(clean.split()) < max(1, int(min_words)):
        return False
    tokens = _token_set(clean)
    if len(tokens) < max(1, int(min_tokens)):
        return False
    lower = clean.lower()
    banned = ("n/a", "same as above", "unknown", "none")
    return not any(x in lower for x in banned)


def _row_tokens(row: dict[str, str]) -> tuple[set[str], set[str], set[str]]:
    return (
        _token_set(str(row.get("china", ""))),
        _token_set(str(row.get("healthcare", ""))),
        _token_set(str(row.get("guns", ""))),
    )


def _is_near_duplicate(
    row_tokens: tuple[set[str], set[str], set[str]],
    existing_tokens: list[tuple[set[str], set[str], set[str]]],
    *,
    avg_threshold: float,
    min_topic_threshold: float,
) -> bool:
    for prev in existing_tokens:
        sims = [_jaccard(row_tokens[i], prev[i]) for i in range(3)]
        avg_sim = sum(sims) / 3.0
        if avg_sim >= avg_threshold:
            return True
        if min(sims) >= min_topic_threshold:
            return True
    return False


def _value_pool_fallback(pool_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    china_goal = [
        "keep manufacturing jobs in the United States",
        "protect technology leadership in semiconductors and AI",
        "avoid a costly military conflict while maintaining deterrence",
        "reduce dependence on fragile supply chains",
        "preserve US influence with allies in Asia",
        "push back on unfair trade practices without a trade shock",
        "defend cyber infrastructure and critical industries",
        "keep inflation risks from geopolitical shocks manageable",
        "strengthen national security while keeping markets stable",
        "build long-term resilience against strategic rivals",
    ]
    china_tool = [
        "through allied coordination and targeted export controls",
        "with selective tariffs plus domestic investment incentives",
        "by using diplomacy-first crisis management and military readiness",
        "with strict technology safeguards and joint R&D with allies",
        "through supply-chain diversification and strategic reserves",
        "using predictable rules instead of sudden policy swings",
        "via stronger maritime deterrence and burden-sharing alliances",
        "by prioritizing economic statecraft over symbolic escalation",
        "through enforceable trade standards and labor protections",
        "with focused sanctions only on high-risk strategic sectors",
    ]
    china_guardrail = [
        "while avoiding broad actions that would sharply raise consumer prices.",
        "while keeping channels open for de-escalation during crises.",
        "while protecting academic exchange that does not involve sensitive technology.",
        "while preventing overreach that hurts small US businesses.",
        "while ensuring Congress has oversight on major escalatory moves.",
        "while balancing security with long-term economic competitiveness.",
        "while minimizing unintended damage to allied economies.",
        "while preserving room for climate and health cooperation.",
        "while forcing clear accountability for coercive behavior.",
        "while keeping military commitments realistic and sustainable.",
    ]

    health_priority = [
        "lowering monthly premiums for working families",
        "protecting coverage for pre-existing conditions",
        "reducing drug prices with transparent negotiation rules",
        "expanding primary care access in underserved areas",
        "cutting administrative waste and billing complexity",
        "improving mental health access with measurable outcomes",
        "protecting catastrophic coverage from surprise costs",
        "rewarding prevention and chronic disease management",
        "keeping rural hospitals financially viable",
        "making insurance options easier to compare and switch",
    ]
    health_model = [
        "using a mixed public-private insurance model",
        "through regulated competition across private plans",
        "with stronger public option availability where markets fail",
        "by outcome-based reimbursement instead of fee-for-volume",
        "through portable coverage that follows people across jobs",
        "with state-led pilots under federal guardrails",
        "by strict transparency mandates for hospitals and insurers",
        "through incremental reforms rather than a single nationwide overhaul",
        "with stronger antitrust enforcement in healthcare markets",
        "by tying subsidies to measurable cost and quality targets",
    ]
    health_guardrail = [
        "and I oppose reforms that add costs without improving access.",
        "and I want annual public scorecards on cost and quality impact.",
        "while preserving patient choice of doctor whenever feasible.",
        "while keeping long-term federal spending on a sustainable path.",
        "with special protection for low-income and high-risk patients.",
        "while reducing out-of-network billing surprises aggressively.",
        "and I support automatic enrollment for eligible uninsured residents.",
        "while guaranteeing transparent appeals for denied claims.",
        "and I want clear penalties for anti-competitive consolidation.",
        "while maintaining independent evaluation of policy outcomes.",
    ]

    gun_rights = [
        "I support lawful self-defense rights",
        "I support responsible firearm ownership",
        "I support the constitutional right to own firearms",
        "I support legal ownership for vetted adults",
        "I support community-level safety with individual rights",
        "I support ownership with serious safety obligations",
        "I support rights-based policy with enforceable safeguards",
        "I support preserving legal ownership traditions",
        "I support lawful carrying where training standards are met",
        "I support individual protection rights with accountability",
    ]
    gun_safety = [
        "with universal background checks for commercial transfers",
        "with mandatory safe-storage standards around minors",
        "with stronger domestic-violence disqualifier enforcement",
        "with rapid reporting for lost or stolen firearms",
        "with risk-based temporary removal orders under due process",
        "with stronger training requirements for first-time buyers",
        "with consistent age-verification and dealer compliance audits",
        "with targeted interventions for repeat violent offenders",
        "with better interstate data sharing for prohibited purchasers",
        "with evidence-based violence prevention grants for local communities",
    ]
    gun_enforcement = [
        "and I want clear penalties for illegal trafficking networks.",
        "and I want prosecutions to focus on violent misuse, not technical errors.",
        "while protecting due process and transparent judicial review.",
        "and I want compliance checks to be predictable and apolitical.",
        "while keeping enforcement resources focused on high-risk actors.",
        "and I oppose broad rules that punish lawful owners without results.",
        "with regular public reporting on what policies actually reduce harm.",
        "while improving community trust in enforcement outcomes.",
        "and I want stronger tracing capacity for crime guns.",
        "while preserving lawful sporting and hunting use.",
    ]

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    i = 0
    while len(out) < pool_size:
        i += 1
        base = i - 1
        c = (
            f"My China policy should {china_goal[base % len(china_goal)]} "
            f"{china_tool[(base * 3 + 7) % len(china_tool)]} "
            f"{china_guardrail[(base * 5 + 11) % len(china_guardrail)]}"
        )
        h = (
            f"My healthcare priority is {health_priority[(base * 2 + 3) % len(health_priority)]}, "
            f"{health_model[(base * 7 + 5) % len(health_model)]}, "
            f"{health_guardrail[(base * 11 + 9) % len(health_guardrail)]}"
        )
        g = (
            f"{gun_rights[(base * 13 + 1) % len(gun_rights)]} "
            f"{gun_safety[(base * 17 + 4) % len(gun_safety)]}, "
            f"{gun_enforcement[(base * 19 + 6) % len(gun_enforcement)]}"
        )
        key = (_normalize_ws(c), _normalize_ws(h), _normalize_ws(g))
        if key in seen:
            # deterministic extra variation if modular collisions occur
            alt = rng.randint(1000, 9999)
            key = (
                f"{key[0]} (Priority Frame {alt})",
                f"{key[1]} (Policy Frame {alt})",
                f"{key[2]} (Safety Frame {alt})",
            )
        seen.add(key)
        out.append(
            {
                "profile_id": f"vp_{len(out):03d}",
                "china": key[0],
                "healthcare": key[1],
                "guns": key[2],
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
    cfg_max_batches = int(getattr(cfg.phase3, "max_llm_batches", 12) or 12)
    max_llm_batches = max(1, cfg_max_batches)
    if max_llm_batches > 64:
        max_llm_batches = 64
    rng = random.Random(seed)
    min_words = int(getattr(cfg.phase3.values, "min_words_per_field", 15) or 15)
    min_tokens = int(getattr(cfg.phase3.values, "min_token_count_per_field", 6) or 6)
    near_dup_avg = float(
        getattr(cfg.phase3.values, "near_duplicate_avg_jaccard_threshold", 0.82) or 0.82
    )
    near_dup_topic = float(
        getattr(cfg.phase3.values, "near_duplicate_min_topic_threshold", 0.72) or 0.72
    )

    pool: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    token_signatures: list[tuple[set[str], set[str], set[str]]] = []
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
                "Each profile must contain long-form text values for: china, healthcare, guns.\n"
                "Every profile must be materially different from other profiles.\n"
                "Avoid repeated sentence templates, repeated openings, and near-duplicate phrasing.\n"
                "Each field should be at least 15 words and include specific tradeoffs or constraints.\n"
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
                    c = _normalize_ws(str(row.get("china", "")))
                    h = _normalize_ws(str(row.get("healthcare", "")))
                    g = _normalize_ws(str(row.get("guns", "")))
                    key = (c, h, g)
                    if not c or not h or not g:
                        continue
                    if not (
                        _valid_value_text(c, min_words=min_words, min_tokens=min_tokens)
                        and _valid_value_text(h, min_words=min_words, min_tokens=min_tokens)
                        and _valid_value_text(g, min_words=min_words, min_tokens=min_tokens)
                    ):
                        continue
                    if key in seen:
                        continue
                    row_sig = _row_tokens({"china": c, "healthcare": h, "guns": g})
                    if _is_near_duplicate(
                        row_sig,
                        token_signatures,
                        avg_threshold=near_dup_avg,
                        min_topic_threshold=near_dup_topic,
                    ):
                        continue
                    seen.add(key)
                    token_signatures.append(row_sig)
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
                c = _normalize_ws(str(row["china"]))
                h = _normalize_ws(str(row["healthcare"]))
                g = _normalize_ws(str(row["guns"]))
                key = (c, h, g)
                if key in seen:
                    continue
                if not (
                    _valid_value_text(c, min_words=min_words, min_tokens=min_tokens)
                    and _valid_value_text(h, min_words=min_words, min_tokens=min_tokens)
                    and _valid_value_text(g, min_words=min_words, min_tokens=min_tokens)
                ):
                    continue
                row_sig = _row_tokens({"china": c, "healthcare": h, "guns": g})
                if _is_near_duplicate(
                    row_sig,
                    token_signatures,
                    avg_threshold=near_dup_avg,
                    min_topic_threshold=near_dup_topic,
                ):
                    continue
                seen.add(key)
                token_signatures.append(row_sig)
                pool.append(
                    {
                        "profile_id": f"vp_{len(pool):03d}",
                        "china": c,
                        "healthcare": h,
                        "guns": g,
                    }
                )
                pbar.update(1, detail="fallback")
                if len(pool) >= pool_size:
                    break

    diversity_ratio = len(seen) / max(1, len(pool))
    lexical_coverage = len(
        set().union(*[t[0] | t[1] | t[2] for t in token_signatures]) if token_signatures else set()
    )
    logger.info(
        "Phase3 value-pool generation finished | size=%d llm_calls=%d llm_failures=%d diversity_ratio=%.3f lexical_coverage=%d",
        len(pool),
        llm_calls,
        llm_failures,
        diversity_ratio,
        lexical_coverage,
    )
    return {
        "metadata": {
            "pool_size": len(pool),
            "target_pool_size": pool_size,
            "llm_calls": llm_calls,
            "llm_failures": llm_failures,
            "diversity_ratio": diversity_ratio,
            "lexical_coverage": lexical_coverage,
            "min_words_per_field": min_words,
            "min_token_count_per_field": min_tokens,
            "near_duplicate_avg_jaccard_threshold": near_dup_avg,
            "near_duplicate_min_topic_threshold": near_dup_topic,
            "max_llm_batches": max_llm_batches,
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
            unique_assignment_when_possible=bool(
                getattr(cfg.phase3.values, "unique_assignment_when_possible", True)
            ),
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
