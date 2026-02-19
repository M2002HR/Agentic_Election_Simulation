from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from election_sim.phase1.eval import build_candidate_objects
from election_sim.utils.io import atomic_write_json, ensure_dir


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
            "- Keep questions crisp (1-3 paragraphs).\n"
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
    m = (p2.get("moderator") or {})
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
            if isinstance(t, dict) and t.get("topic"):
                out.append({"topic": t["topic"], "title": t.get("title", t["topic"])})
        if out:
            return out

    # مطابق صورت پروژه (۳ موضوع اصلی)
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

    import json
    try:
        obj = json.loads(raw)
        qs = obj.get("questions", [])
        if isinstance(qs, list) and all(isinstance(x, str) for x in qs):
            return [q.strip() for q in qs][:n]
    except Exception:
        pass

    lines = [ln.strip(" -\t") for ln in raw.splitlines() if ln.strip()]
    return lines[:n]


def _ask_candidate(llm, persona_text: str, question: str, meta: dict[str, Any]) -> str:
    prompt = f"{persona_text}\n\nQuestion: {question}\nAnswer as the candidate."
    return llm.invoke(prompt, meta=meta).strip()


def _ask_followup(llm, moderator: Moderator, context: str, meta: dict[str, Any]) -> str:
    prompt = (
        f"{moderator.persona_text()}\n\n"
        "You have just heard a debate exchange.\n"
        "Write ONE sharp follow-up question that exploits weakness/ambiguity, demands specifics, and avoids platitudes.\n"
        "Keep it concise.\n\n"
        f"Context:\n{context}\n\n"
        "Follow-up question:"
    )
    return llm.invoke(prompt, meta=meta).strip()


def run_debate(cfg, llm_client, run_dir: str, logger) -> dict[str, Any]:
    """
    خروجی‌ها در <run_dir>/phase2/ :
      - debate.json  (ساختار کامل)
      - debate.jsonl (لاگ رویدادها)
      - questions.json (سوالات تولیدشده برای هر موضوع)
    """

    out_dir = ensure_dir(f"{run_dir}/phase2")

    moderator = _make_moderator(cfg)
    topics = _topic_plan(cfg)
    n_q = _n_questions_per_topic(cfg)

    rng = random.Random(getattr(cfg.project, "seed", 0) or 0)

    candidates, persona_texts = build_candidate_objects(cfg)
    cand_ids = list(candidates.keys())
    if len(cand_ids) != 2:
        raise ValueError("Phase2 expects exactly 2 candidates")

    # 1) تولید سوال‌ها
    questions_by_topic: dict[str, list[str]] = {}
    for t in topics:
        qs = _gen_questions(llm_client, moderator, t["topic"], t["title"], n_q)
        if len(qs) < n_q:
            logger.warning("Topic %s generated only %d/%d questions", t["topic"], len(qs), n_q)
        questions_by_topic[t["topic"]] = qs

    atomic_write_json(f"{out_dir}/questions.json", {"topics": topics, "questions": questions_by_topic})

    # 2) اجرای مناظره (۱۵ سوال = ۳ موضوع × ۵ سوال)، هر سوال با follow-up
    records: list[dict[str, Any]] = []

    def emit(rec: dict[str, Any]) -> None:
        records.append(rec)
        with open(f"{out_dir}/debate.jsonl", "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ts = lambda: __import__("time").time()

    for t in topics:
        emit({"type": "topic_intro", "topic": t["topic"], "topic_title": t["title"], "timestamp": ts()})
        qs = questions_by_topic.get(t["topic"], [])
        for i, q in enumerate(qs[:n_q], start=1):
            first = rng.choice(cand_ids)
            second = cand_ids[1] if first == cand_ids[0] else cand_ids[0]

            a1 = _ask_candidate(
                llm_client,
                persona_texts[first],
                q,
                meta={"phase": "phase2", "topic": t["topic"], "question_index": i, "candidate": first, "turn": "first"},
            )
            a2 = _ask_candidate(
                llm_client,
                persona_texts[second],
                q,
                meta={"phase": "phase2", "topic": t["topic"], "question_index": i, "candidate": second, "turn": "second"},
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
                meta={"phase": "phase2", "topic": t["topic"], "question_index": i, "step": "followup", "target": follow_target},
            )
            fu_a = _ask_candidate(
                llm_client,
                persona_texts[follow_target],
                fu_q,
                meta={"phase": "phase2", "topic": t["topic"], "question_index": i, "candidate": follow_target, "turn": "follow_answer"},
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
                    "follow_up": fu_q,
                    "follow_target": follow_target,
                    "follow_answer": fu_a,
                    "timestamp": ts(),
                }
            )

    out = {
        "phase": "phase2",
        "moderator": {"name": moderator.name, "traits": moderator.traits},
        "persona_map": persona_texts,
        "records": records,
    }
    atomic_write_json(f"{out_dir}/debate.json", out)
    return out
