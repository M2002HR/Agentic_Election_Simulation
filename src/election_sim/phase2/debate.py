from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from election_sim.phase1.candidates import build_candidate_objects, build_persona_text
from election_sim.phase2.analysis import analyze_debate
from election_sim.utils.io import atomic_write_json, ensure_dir, write_json
from election_sim.utils.progress import ProgressBar

_INVALID_QUESTION_LINES = {"```", "```json", "{", "}", "[", "]", "\"questions\": ["}
_QUALITY_TOKENS = [
    "tradeoff",
    "timeline",
    "metric",
    "cost",
    "evidence",
    "specific",
    "how",
    "why",
    "what if",
]


@dataclass
class Moderator:
    name: str
    traits: dict[str, float]

    def persona_text(self) -> str:
        t = self.traits
        return (
            f"Role: Debate Moderator\n"
            f"Name: {self.name}\n\n"
            f"Traits (0-10):\n"
            f"- honesty: {t.get('honesty', 8)}\n"
            f"- aggressiveness: {t.get('aggressiveness', 6)}\n"
            f"- confidence: {t.get('confidence', 7)}\n"
            f"- charisma: {t.get('charisma', 6)}\n\n"
            "Rules:\n"
            "- Ask challenging, specific, and adversarial questions.\n"
            "- Avoid platitudes; demand concrete policies, tradeoffs, and metrics.\n"
            "- After both candidates answer, ask ONE pointed follow-up to ONE of them.\n"
            "- Keep questions crisp and policy-oriented.\n"
        )


def _phase2_cfg(cfg) -> dict[str, Any]:
    phase2 = getattr(cfg, "phase2", None)
    if phase2 is None:
        return {}
    if isinstance(phase2, dict):
        return phase2
    return phase2.model_dump()  # type: ignore[attr-defined]


def _make_moderator(cfg) -> Moderator:
    p2 = _phase2_cfg(cfg)
    m = p2.get("moderator") or {}
    return Moderator(
        name=m.get("name", "Moderator"),
        traits=m.get(
            "traits",
            {"honesty": 8.0, "aggressiveness": 6.0, "confidence": 7.0, "charisma": 6.0},
        ),
    )


def _topic_plan(cfg) -> list[dict[str, str]]:
    p2 = _phase2_cfg(cfg)
    topics = p2.get("topics")
    if topics and isinstance(topics, list):
        out: list[dict[str, str]] = []
        for t in topics:
            if not isinstance(t, dict):
                continue
            topic_id = t.get("topic") or t.get("id")
            if topic_id:
                out.append({"topic": topic_id, "title": t.get("title", topic_id)})
        if out:
            return out

    return [
        {"topic": "china", "title": "China and US global power"},
        {"topic": "healthcare", "title": "US Healthcare system"},
        {"topic": "guns", "title": "Gun licensing in the US"},
    ]


def _n_questions_per_topic(cfg) -> int:
    p2 = _phase2_cfg(cfg)
    n = p2.get("questions_per_topic", 5)
    try:
        n = int(n)
    except Exception:
        n = 5
    return max(1, n)


def _answer_word_limit(cfg) -> int:
    p2 = _phase2_cfg(cfg)
    n = p2.get("answer_word_limit", 250)
    try:
        n = int(n)
    except Exception:
        n = 250
    return max(80, n)


def _quality_cfg(cfg) -> dict[str, Any]:
    p2 = _phase2_cfg(cfg)
    q = p2.get("quality") or {}
    if isinstance(q, dict):
        return q
    return {}


def _candidate_map(raw_candidates: Any) -> dict[str, Any]:
    payload = raw_candidates
    if isinstance(payload, tuple) and payload:
        payload = payload[0]
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        out: dict[str, Any] = {}
        for c in payload:
            cid = getattr(c, "candidate_id", None)
            if not isinstance(cid, str) or not cid:
                raise TypeError(
                    "build_candidate_objects(cfg) returned a list with an item missing 'candidate_id'."
                )
            out[cid] = c
        return out
    raise TypeError(
        "build_candidate_objects(cfg) must return list[Candidate], "
        "dict[candidate_id, candidate_obj], or a legacy tuple with one of those as the first item."
    )


def _strip_fences(raw: str) -> str:
    out_lines: list[str] = []
    for ln in (raw or "").splitlines():
        line = ln.strip()
        if line.startswith("```"):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).strip()


def _fallback_questions(title: str, n: int) -> list[str]:
    seeds = [
        f"For {title}, what concrete policy will you pass in your first 100 days, with measurable targets?",
        f"In {title}, what tradeoff do you accept between cost and outcomes, and why?",
        f"Provide a timeline and budget estimate for your {title} plan.",
        f"What is the strongest criticism of your {title} plan, and how do you answer it?",
        f"If your {title} approach fails after two years, what is your fallback policy?",
    ]
    if n <= len(seeds):
        return seeds[:n]
    out = list(seeds)
    idx = 0
    while len(out) < n:
        out.append(f"{seeds[idx % len(seeds)]} (Variant {idx + 1})")
        idx += 1
    return out[:n]


def _parse_question_list(raw: str, n: int, title: str) -> list[str]:
    cleaned = _strip_fences(raw)
    candidates: list[str] = []

    try:
        obj = json.loads(cleaned)
        qs = obj.get("questions", []) if isinstance(obj, dict) else []
        if isinstance(qs, list):
            candidates.extend(str(x).strip() for x in qs if isinstance(x, str))
    except Exception:
        pass

    if not candidates:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                obj = json.loads(snippet)
                qs = obj.get("questions", []) if isinstance(obj, dict) else []
                if isinstance(qs, list):
                    candidates.extend(str(x).strip() for x in qs if isinstance(x, str))
            except Exception:
                pass

    if not candidates:
        for ln in cleaned.splitlines():
            q = ln.strip(" -\t")
            if not q or q.lower() in _INVALID_QUESTION_LINES:
                continue
            candidates.append(q)

    out: list[str] = []
    seen: set[str] = set()
    for q in candidates:
        q2 = " ".join(q.split())
        low = q2.lower()
        if not q2 or low in _INVALID_QUESTION_LINES:
            continue
        if len(q2) < 18:
            continue
        if q2 not in seen:
            seen.add(q2)
            out.append(q2)
        if len(out) >= n:
            break

    if len(out) < n:
        for fq in _fallback_questions(title, n):
            if fq not in seen:
                out.append(fq)
                seen.add(fq)
            if len(out) >= n:
                break
    return out[:n]


def _gen_questions(llm, moderator: Moderator, topic: str, title: str, n: int) -> list[str]:
    prompt = (
        f"{moderator.persona_text()}\n\n"
        f"Task: Generate {n} debate questions for the topic '{title}' (key: {topic}).\n"
        "Constraints:\n"
        "- Questions must be challenging and specific.\n"
        "- Each question should force tradeoffs, concrete policies, timelines, metrics, or costs.\n"
        "- Avoid generic or repetitive phrasing.\n"
        '- Output JSON only with this schema: {"questions": [..strings..]}\n'
    )
    raw = llm.invoke(prompt, meta={"phase": "phase2", "step": "gen_questions", "topic": topic})
    return _parse_question_list(raw, n, title)


def _next_main_question(
    llm,
    moderator: Moderator,
    topic: str,
    title: str,
    turn_index: int,
    prior_exchanges: list[dict[str, Any]],
    fallback_question: str,
) -> str:
    recent = prior_exchanges[-2:]
    ctx_lines: list[str] = []
    for idx, item in enumerate(recent, start=1):
        ctx_lines.append(f"Recent exchange {idx}:")
        ctx_lines.append(f"- Question: {item.get('question', '')}")
        ctx_lines.append(f"- First answer: {str(item.get('first_answer', ''))[:260]}")
        ctx_lines.append(f"- Second answer: {str(item.get('second_answer', ''))[:260]}")
        ctx_lines.append(f"- Follow-up: {str(item.get('follow_up', ''))[:200]}")
    context = "\n".join(ctx_lines).strip() or "No prior exchanges."
    prompt = (
        f"{moderator.persona_text()}\n\n"
        f"Topic: {title} ({topic})\n"
        f"You are generating main question #{turn_index} for this topic.\n"
        "The new question must build on weaknesses or unanswered points from recent exchanges.\n"
        "It must remain challenging, specific, policy-oriented, and non-repetitive.\n"
        '- Return JSON only with schema: {"questions":[\"...\"]} and exactly one question.\n\n'
        f"Recent context:\n{context}\n"
    )
    raw = llm.invoke(
        prompt,
        meta={
            "phase": "phase2",
            "step": "gen_next_main_question",
            "topic": topic,
            "turn_index": turn_index,
        },
    )
    parsed = _parse_question_list(raw, n=1, title=title)
    if not parsed:
        return fallback_question
    q = parsed[0].strip()
    if len(q) < 18:
        return fallback_question
    return q


def _words(text: str) -> list[str]:
    return [w for w in (text or "").strip().split() if w]


def _truncate_words(text: str, limit: int) -> str:
    ws = _words(text)
    if len(ws) <= limit:
        return text.strip()
    return " ".join(ws[:limit]).strip()


def _ask_candidate_once(llm, persona_text: str, question: str, meta: dict[str, Any], word_limit: int) -> str:
    prompt = (
        f"{persona_text}\n\n"
        f"Question: {question}\n"
        f"Answer as the candidate in <= {word_limit} words. Be concrete, policy-specific, and non-generic."
    )
    return llm.invoke(prompt, meta=meta).strip()


def _ask_candidate(
    llm,
    persona_text: str,
    question: str,
    meta: dict[str, Any],
    word_limit: int,
) -> str:
    answer = _ask_candidate_once(llm, persona_text, question, meta, word_limit)
    wc = len(_words(answer))
    min_wc = max(40, int(word_limit * 0.35))
    if wc < min_wc:
        retry_meta = dict(meta)
        retry_meta["retry"] = "length"
        retry_prompt = (
            f"{persona_text}\n\n"
            f"Question: {question}\n"
            f"Your last answer was too short ({wc} words). "
            f"Rewrite in {min_wc}-{word_limit} words with specifics."
        )
        answer = llm.invoke(retry_prompt, meta=retry_meta).strip()
    return _truncate_words(answer, word_limit)


def _ask_followup(
    llm,
    moderator: Moderator,
    context: str,
    meta: dict[str, Any],
) -> str:
    prompt = (
        f"{moderator.persona_text()}\n\n"
        "You have just heard a debate exchange.\n"
        "Write ONE sharp follow-up question that exploits weakness/ambiguity, demands specifics, and avoids platitudes.\n"
        "Keep it concise and concrete.\n\n"
        f"Context:\n{context}\n\n"
        "Follow-up question:"
    )
    return llm.invoke(prompt, meta=meta).strip()


def _moderator_critique(
    question: str,
    first_candidate: str,
    first_answer: str,
    second_candidate: str,
    second_answer: str,
) -> str:
    q_tokens = {t for t in question.lower().split() if len(t) > 4}

    def _coverage(ans: str) -> int:
        toks = {t for t in ans.lower().replace(".", " ").replace(",", " ").split() if len(t) > 4}
        return len(q_tokens & toks)

    first_cov = _coverage(first_answer)
    second_cov = _coverage(second_answer)
    first_len = len(_words(first_answer))
    second_len = len(_words(second_answer))
    first_strength = "more specific" if first_cov >= second_cov else "less specific"
    second_strength = "more specific" if second_cov > first_cov else "less specific"
    return (
        f"{first_candidate} was {first_strength} on the core question "
        f"(coverage={first_cov}, words={first_len}); "
        f"{second_candidate} was {second_strength} "
        f"(coverage={second_cov}, words={second_len}). "
        "Both should provide clearer timelines, measurable targets, and cost tradeoffs."
    )


def _score_question_quality(question: str) -> dict[str, Any]:
    q = (question or "").strip()
    low = q.lower()
    token_hits = sum(1 for t in _QUALITY_TOKENS if t in low)
    score = 0
    if len(q) >= 30:
        score += 1
    if "?" in q:
        score += 1
    if token_hits >= 1:
        score += 1
    if token_hits >= 2:
        score += 1
    if any(x in low for x in ["cost", "timeline", "metric", "tradeoff"]):
        score += 1
    score = max(0, min(5, score))
    return {"score_0_5": score, "token_hits": token_hits}


def _evaluate_moderator_quality(
    cfg,
    topics: list[dict[str, str]],
    questions_by_topic: dict[str, list[str]],
) -> dict[str, Any]:
    per_topic: dict[str, Any] = {}
    all_scores: list[int] = []
    for t in topics:
        topic = t["topic"]
        qs = questions_by_topic.get(topic, [])
        q_infos: list[dict[str, Any]] = []
        for q in qs:
            q_score = _score_question_quality(q)
            q_infos.append({"question": q, **q_score})
            all_scores.append(q_score["score_0_5"])
        topic_avg = (sum(x["score_0_5"] for x in q_infos) / len(q_infos)) if q_infos else 0.0
        per_topic[topic] = {"avg_score_0_5": topic_avg, "items": q_infos}

    overall_avg = (sum(all_scores) / len(all_scores)) if all_scores else 0.0
    q_cfg = _quality_cfg(cfg)
    threshold = float(q_cfg.get("min_avg_score", 2.0))
    return {
        "enabled": bool(q_cfg.get("enabled", True)),
        "overall_avg_score_0_5": overall_avg,
        "min_avg_score_threshold": threshold,
        "pass": overall_avg >= threshold,
        "by_topic": per_topic,
    }


def run_debate(cfg, llm_client, run_dir: str, logger) -> dict[str, Any]:
    out_dir = ensure_dir(f"{run_dir}/phase2")
    moderator = _make_moderator(cfg)
    topics = _topic_plan(cfg)
    n_q = _n_questions_per_topic(cfg)
    answer_word_limit = _answer_word_limit(cfg)
    logger.info(
        "Phase2 started | run_dir=%s topics=%d questions_per_topic=%d answer_word_limit=%d",
        run_dir,
        len(topics),
        n_q,
        answer_word_limit,
    )

    rng = random.Random(getattr(cfg.project, "random_seed", 0) or 0)
    candidates = _candidate_map(build_candidate_objects(cfg))
    cand_cfgs = {
        "democrat": cfg.phase1.candidates.democrat,
        "republican": cfg.phase1.candidates.republican,
    }
    persona_texts = {cid: build_persona_text(cfg, cand_cfgs[cid]) for cid in candidates.keys()}

    cand_ids = list(candidates.keys())
    if len(cand_ids) != 2:
        raise ValueError("Phase2 expects exactly 2 candidates")

    questions_by_topic: dict[str, list[str]] = {}
    with ProgressBar(max(1, len(topics)), "Phase2 QGen", logger=logger) as qbar:
        for idx, t in enumerate(topics, start=1):
            qs = _gen_questions(llm_client, moderator, t["topic"], t["title"], n_q)
            questions_by_topic[t["topic"]] = qs
            logger.info(
                "Phase2 question set | topic=%s index=%d/%d generated=%d",
                t["topic"],
                idx,
                len(topics),
                len(qs),
            )
            qbar.update(1, detail=t["topic"])
    questions_json_path = f"{out_dir}/questions.json"
    atomic_write_json(
        questions_json_path,
        {"topics": topics, "seed_questions": questions_by_topic, "questions": questions_by_topic},
    )
    logger.info("Phase2 questions saved: %s", questions_json_path)

    quality_eval = _evaluate_moderator_quality(cfg, topics, questions_by_topic)
    quality_rel = _quality_cfg(cfg).get("output_quality_json", "phase2/debate_quality.json")
    quality_path = f"{run_dir}/{quality_rel}"
    atomic_write_json(quality_path, quality_eval)
    logger.info("Phase2 moderator quality saved: %s", quality_path)

    records: list[dict[str, Any]] = []

    def emit(rec: dict[str, Any]) -> None:
        records.append(rec)
        with open(f"{out_dir}/debate.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ts = lambda: __import__("time").time()

    emit(
        {
            "type": "debate_intro",
            "timestamp": ts(),
            "moderator": {"name": moderator.name, "traits": moderator.traits},
            "candidates": [
                {
                    "candidate_id": cid,
                    "display_name": candidates[cid].display_name,
                    "party": candidates[cid].party,
                    "traits": candidates[cid].personal_traits,
                }
                for cid in cand_ids
            ],
            "topics": topics,
        }
    )

    total_debate_questions = max(1, len(topics) * n_q)
    asked_questions_by_topic: dict[str, list[str]] = {}
    with ProgressBar(max(1, total_debate_questions), "Phase2 Debate", logger=logger) as dbar:
        for topic_idx, t in enumerate(topics, start=1):
            emit({"type": "topic_intro", "topic": t["topic"], "topic_title": t["title"], "timestamp": ts()})
            qs_seed = questions_by_topic.get(t["topic"], [])
            topic_history: list[dict[str, Any]] = []
            asked_questions_by_topic[t["topic"]] = []
            for i in range(1, n_q + 1):
                fallback_q = (
                    qs_seed[i - 1]
                    if i - 1 < len(qs_seed)
                    else _fallback_questions(t["title"], n_q)[i - 1]
                )
                if i == 1:
                    q = fallback_q
                else:
                    q = _next_main_question(
                        llm_client,
                        moderator,
                        t["topic"],
                        t["title"],
                        i,
                        topic_history,
                        fallback_q,
                    )
                asked_questions_by_topic[t["topic"]].append(q)
                logger.info(
                    "Phase2 main question | topic=%s question=%d/%d adaptive=%s",
                    t["topic"],
                    i,
                    n_q,
                    "no" if i == 1 else "yes",
                )
                first = rng.choice(cand_ids)
                second = cand_ids[1] if first == cand_ids[0] else cand_ids[0]

                a1 = _ask_candidate(
                    llm_client,
                    persona_texts[first],
                    q,
                    meta={
                        "phase": "phase2",
                        "topic": t["topic"],
                        "question_index": i,
                        "candidate": first,
                        "turn": "first",
                    },
                    word_limit=answer_word_limit,
                )
                a2 = _ask_candidate(
                    llm_client,
                    persona_texts[second],
                    q,
                    meta={
                        "phase": "phase2",
                        "topic": t["topic"],
                        "question_index": i,
                        "candidate": second,
                        "turn": "second",
                    },
                    word_limit=answer_word_limit,
                )
                critique = _moderator_critique(
                    q,
                    first,
                    a1,
                    second,
                    a2,
                )

                follow_target = rng.choice([first, second])
                ctx = (
                    f"Topic: {t['title']}\n"
                    f"Question: {q}\n"
                    f"{candidates[first].display_name}: {a1}\n"
                    f"{candidates[second].display_name}: {a2}\n"
                )
                fu_q = _ask_followup(
                    llm_client,
                    moderator,
                    ctx,
                    meta={
                        "phase": "phase2",
                        "topic": t["topic"],
                        "question_index": i,
                        "step": "followup",
                        "target": follow_target,
                    },
                )
                fu_a = _ask_candidate(
                    llm_client,
                    persona_texts[follow_target],
                    fu_q,
                    meta={
                        "phase": "phase2",
                        "topic": t["topic"],
                        "question_index": i,
                        "candidate": follow_target,
                        "turn": "follow_answer",
                    },
                    word_limit=answer_word_limit,
                )

                emit(
                    {
                        "type": "qa",
                        "topic": t["topic"],
                        "topic_title": t["title"],
                        "question_index": i,
                        "question": q,
                        "first_candidate": first,
                        "first_answer": a1,
                        "second_candidate": second,
                        "second_answer": a2,
                        "moderator_critique": critique,
                        "follow_up": fu_q,
                        "follow_target": follow_target,
                        "follow_answer": fu_a,
                        "timestamp": ts(),
                    }
                )
                topic_history.append(
                    {
                        "question": q,
                        "first_answer": a1,
                        "second_answer": a2,
                        "follow_up": fu_q,
                    }
                )
                logger.info(
                    "Phase2 QA item | topic=%d/%d question=%d/%d first=%s second=%s follow_target=%s",
                    topic_idx,
                    len(topics),
                    i,
                    n_q,
                    first,
                    second,
                    follow_target,
                )
                dbar.update(1, detail=f"{t['topic']} q{i}")

    atomic_write_json(
        questions_json_path,
        {
            "topics": topics,
            "seed_questions": questions_by_topic,
            "questions": asked_questions_by_topic,
            "adaptive_main_questions": True,
        },
    )
    logger.info("Phase2 questions updated with asked/adaptive sequence: %s", questions_json_path)

    summary = analyze_debate(cfg, run_dir, records)
    out = {
        "phase": "phase2",
        "moderator": {"name": moderator.name, "traits": moderator.traits},
        "persona_map": persona_texts,
        "records": records,
        "quality": quality_eval,
        "summary": summary,
    }

    atomic_write_json(f"{out_dir}/debate.json", out)
    logger.info("Phase2 debate saved: %s", f"{out_dir}/debate.json")

    transcript_path = Path(run_dir) / cfg.phase2.output_transcript_json
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(transcript_path, records)
    logger.info("Phase2 transcript saved: %s", transcript_path)
    logger.info("Phase2 finished | records=%d", len(records))
    return out
