from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from election_sim.phase1.candidates import build_persona_text, build_candidate_objects
from election_sim.phase1.agent import CandidateAgent
from election_sim.phase1.analysis import analyze_answers, summarize_phase1
from election_sim.utils.io import ensure_dir, atomic_write_json

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
    # Very simple heuristics:
    # - persona_consistency: check if party keyword appears OR stance keywords appear
    # - honesty_cues: penalize evasiveness markers
    ans = (answer or "").lower()
    persona_l = (persona or "").lower()

    persona_score = 0
    if "democrat" in persona_l and ("background checks" in ans or "expand access" in ans or "alliances" in ans):
        persona_score = 2
    elif "republican" in persona_l and ("2nd amendment" in ans or "market" in ans or "tariff" in ans):
        persona_score = 2
    elif any(k in ans for k in ["china", "health", "gun", "tariff", "alliance", "registry", "single-payer"]):
        persona_score = 1

    honesty_penalty = 1 if any(m in ans for m in HONESTY_CUE_MARKERS) else 0
    honesty_score = max(0, 2 - honesty_penalty)

    return {"persona_consistency": persona_score, "honesty_cues": honesty_score}

def run_phase1(cfg, llm_client, run_dir: str, logger) -> Dict[str, Any]:
    ensure_dir(run_dir)
    out_dir = f"{run_dir}/phase1"
    ensure_dir(out_dir)

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

    eval_records: List[Dict[str, Any]] = []
    questions = cfg.phase1.evaluation.controversial_questions

    for q in questions:
        for c in candidates:
            persona = persona_texts[c.candidate_id]
            prompt = (
                cfg.phase1.prompts.system_template.strip()
                + "\n\n"
                + persona.strip()
                + "\n\n"
                + cfg.phase1.prompts.answer_template.format(question=q).strip()
            )
            meta = {"phase": "phase1", "candidate_id": c.candidate_id, "question": q}
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

    atomic_write_json(f"{run_dir}/{cfg.phase1.evaluation.output.eval_json}", {"records": eval_records})

    # LLM-based short analysis for report
    analysis_out = analyze_answers(cfg, llm_client, run_dir, persona_texts, eval_records, logger)
    summarize_phase1(cfg, run_dir, eval_records, analysis_out.get('records', []))


    return {"candidates": cand_payload, "eval": {"records": eval_records}}
