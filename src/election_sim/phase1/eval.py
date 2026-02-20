from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict, List

from election_sim.phase1.agent import CandidateAgent
from election_sim.phase1.analysis import analyze_answers, summarize_phase1
from election_sim.phase1.candidates import build_candidate_objects, build_persona_text
from election_sim.utils.io import atomic_write_json, ensure_dir
from election_sim.utils.progress import ProgressBar

HONESTY_CUE_MARKERS = [
    "i don't recall",
    "i do not recall",
    "no comment",
    "can't disclose",
    "cannot disclose",
    "refuse",
    "deflect",
]


def heuristic_scores(answer: str, persona: str) -> Dict[str, int]:
    ans = (answer or "").lower()
    persona_l = (persona or "").lower()

    persona_score = 0
    if "democrat" in persona_l and (
        "background checks" in ans or "expand access" in ans or "alliances" in ans
    ):
        persona_score = 2
    elif "republican" in persona_l and ("2nd amendment" in ans or "market" in ans or "tariff" in ans):
        persona_score = 2
    elif any(k in ans for k in ["china", "health", "gun", "tariff", "alliance", "registry", "single-payer"]):
        persona_score = 1

    honesty_penalty = 1 if any(m in ans for m in HONESTY_CUE_MARKERS) else 0
    honesty_score = max(0, 2 - honesty_penalty)
    return {"persona_consistency": persona_score, "honesty_cues": honesty_score}


def _run_honesty_stress_test(
    cfg,
    llm_client,
    run_dir: str,
    candidates: list[Any],
    logger,
) -> dict[str, Any]:
    stress_cfg = getattr(cfg.phase1.evaluation, "honesty_stress", None)
    if stress_cfg is None or not getattr(stress_cfg, "enabled", True):
        logger.info("Phase1 honesty stress-test disabled.")
        return {"enabled": False, "records": []}

    override_map = dict(getattr(stress_cfg, "honesty_override", {}) or {})
    questions = list(getattr(stress_cfg, "questions", []) or []) or list(
        cfg.phase1.evaluation.controversial_questions
    )
    out_rel = getattr(stress_cfg, "output_json", "phase1/honesty_stress_test.json")
    out_path = f"{run_dir}/{out_rel}"

    logger.info(
        "Phase1 honesty stress-test started | candidates=%d questions=%d",
        len(candidates),
        len(questions),
    )

    records: list[dict[str, Any]] = []
    with ProgressBar(max(1, len(candidates) * len(questions)), "Phase1 Honesty", logger=logger) as pbar:
        for c in candidates:
            cand_cfg = getattr(cfg.phase1.candidates, c.candidate_id)
            traits = dict(cand_cfg.personal_traits)
            if c.candidate_id in override_map:
                traits["honesty"] = float(override_map[c.candidate_id])

            shadow_cfg = SimpleNamespace(
                display_name=cand_cfg.display_name,
                party=cand_cfg.party,
                personal_traits=traits,
                policy_stances=dict(cand_cfg.policy_stances),
            )
            persona = build_persona_text(cfg, shadow_cfg)

            for q_idx, q in enumerate(questions, start=1):
                prompt = (
                    cfg.phase1.prompts.system_template.strip()
                    + "\n\n"
                    + persona.strip()
                    + "\n\n"
                    + cfg.phase1.prompts.answer_template.format(question=q).strip()
                )
                meta = {
                    "phase": "phase1",
                    "role": "honesty_stress",
                    "candidate_id": c.candidate_id,
                    "question_index": q_idx,
                    "question": q,
                }
                ans = llm_client.invoke(prompt, meta=meta)
                scores = heuristic_scores(ans, persona)
                records.append(
                    {
                        "ts": time.time(),
                        "candidate_id": c.candidate_id,
                        "question": q,
                        "answer": ans,
                        "scores": scores,
                        "honesty_override": traits.get("honesty"),
                    }
                )
                logger.info(
                    "Phase1 honesty item | candidate=%s question=%d/%d score=%s",
                    c.candidate_id,
                    q_idx,
                    len(questions),
                    scores,
                )
                pbar.update(1, detail=f"{c.candidate_id} q{q_idx}")

    atomic_write_json(out_path, {"records": records})
    logger.info("Phase1 honesty stress-test saved: %s", out_path)
    return {"enabled": True, "records": records, "path": out_path}


def run_phase1(cfg, llm_client, run_dir: str, logger) -> Dict[str, Any]:
    ensure_dir(run_dir)
    out_dir = f"{run_dir}/phase1"
    ensure_dir(out_dir)
    logger.info("Phase1 started | run_dir=%s", run_dir)

    candidates = build_candidate_objects(cfg)
    cand_payload: Dict[str, Any] = {}
    persona_texts: Dict[str, str] = {}

    for c in candidates:
        cand_cfg = getattr(cfg.phase1.candidates, c.candidate_id)
        persona = build_persona_text(cfg, cand_cfg)
        persona_texts[c.candidate_id] = persona
        cand_payload[c.candidate_id] = {
            "display_name": c.display_name,
            "party": c.party,
            "personal_traits": c.personal_traits,
            "policy_stances": c.policy_stances,
            "persona_text": persona,
        }

    atomic_write_json(f"{run_dir}/{cfg.phase1.evaluation.output.candidates_json}", cand_payload)
    logger.info("Phase1 candidates saved: %s", cfg.phase1.evaluation.output.candidates_json)

    eval_records: List[Dict[str, Any]] = []
    questions = cfg.phase1.evaluation.controversial_questions
    total_eval = len(questions) * len(candidates)
    logger.info(
        "Phase1 evaluation started | questions=%d candidates=%d total_calls=%d",
        len(questions),
        len(candidates),
        total_eval,
    )

    with ProgressBar(max(1, total_eval), "Phase1 Eval", logger=logger) as pbar:
        for q_idx, q in enumerate(questions, start=1):
            for c in candidates:
                persona = persona_texts[c.candidate_id]
                agent = CandidateAgent(candidate_id=c.candidate_id, display_name=c.display_name)
                ans = agent.answer(cfg, llm_client, q)
                scores = heuristic_scores(ans, persona)
                eval_records.append(
                    {
                        "ts": time.time(),
                        "candidate_id": c.candidate_id,
                        "question": q,
                        "answer": ans,
                        "scores": scores,
                    }
                )
                logger.info(
                    "Phase1 eval item | question=%d/%d candidate=%s scores=%s",
                    q_idx,
                    len(questions),
                    c.candidate_id,
                    scores,
                )
                pbar.update(1, detail=f"q{q_idx} {c.candidate_id}")

    atomic_write_json(f"{run_dir}/{cfg.phase1.evaluation.output.eval_json}", {"records": eval_records})
    logger.info("Phase1 evaluation saved: %s", cfg.phase1.evaluation.output.eval_json)

    analysis_out = analyze_answers(cfg, llm_client, run_dir, persona_texts, eval_records, logger)
    summary_out = summarize_phase1(cfg, run_dir, eval_records, analysis_out.get("records", []))
    honesty_out = _run_honesty_stress_test(cfg, llm_client, run_dir, candidates, logger)
    logger.info("Phase1 finished | records=%d", len(eval_records))

    return {
        "candidates": cand_payload,
        "eval": {"records": eval_records},
        "analysis": analysis_out,
        "summary": summary_out,
        "honesty_stress": honesty_out,
    }
