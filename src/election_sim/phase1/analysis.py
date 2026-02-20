from __future__ import annotations

import json
from typing import Any, Dict, List

from election_sim.utils.io import atomic_write_json, ensure_dir
from election_sim.utils.progress import ProgressBar


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction:
    - If model returns raw JSON, parse directly
    - Else find first {...} block and parse
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {"persona_consistency": None, "honesty_cues": None, "notes": text[:500]}
    return {"persona_consistency": None, "honesty_cues": None, "notes": text[:500]}


def _cfg_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[attr-defined]
    return {}


def analyze_answers(cfg, llm_client, run_dir: str, persona_by_candidate: Dict[str, str], eval_records: List[Dict[str, Any]], logger) -> Dict[str, Any]:
    out_dir = f"{run_dir}/phase1"
    ensure_dir(out_dir)

    a_cfg = _cfg_to_dict(getattr(cfg.phase1.evaluation, "analysis", None))
    if not a_cfg.get("enabled", True):
        logger.info("Phase1 analysis disabled.")
        return {"enabled": False, "records": []}

    mode = a_cfg.get("mode", "llm")
    max_words = int(a_cfg.get("max_words", 90))
    template = a_cfg.get("prompt_template", "")
    logger.info("Phase1 analysis started | mode=%s records=%d", mode, len(eval_records))

    analysis_records: List[Dict[str, Any]] = []
    with ProgressBar(len(eval_records), "Phase1 Analysis", logger=logger) as pbar:
        for idx, rec in enumerate(eval_records, start=1):
            cid = rec["candidate_id"]
            persona = persona_by_candidate[cid]
            question = rec["question"]
            answer = rec["answer"]

            if mode == "heuristic":
                analysis = {
                    "persona_consistency": rec.get("scores", {}).get("persona_consistency"),
                    "honesty_cues": rec.get("scores", {}).get("honesty_cues"),
                    "notes": "Heuristic-only mode.",
                }
            else:
                prompt = (
                    template.format(persona=persona, question=question, answer=answer)
                    + f"\nConstraints: <= {max_words} words in notes.\n"
                )
                meta = {"phase": "phase1", "role": "analyst", "candidate_id": cid, "question": question}
                raw = llm_client.invoke(prompt, meta=meta)
                analysis = _extract_json(raw)

            analysis_records.append(
                {
                    "candidate_id": cid,
                    "question": question,
                    "analysis": analysis,
                }
            )
            logger.info(
                "Phase1 analysis item | %d/%d candidate=%s",
                idx,
                len(eval_records),
                cid,
            )
            pbar.update(1, detail=f"{cid}")

    out_path = f"{run_dir}/{a_cfg.get('output_analysis_json', 'phase1/analysis.json')}"
    atomic_write_json(out_path, {"records": analysis_records})
    logger.info("Phase1 analysis saved: %s", out_path)

    return {"enabled": True, "records": analysis_records}


def summarize_phase1(cfg, run_dir: str, eval_records: List[Dict[str, Any]], analysis_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_c = {}
    for r in eval_records:
        cid = r["candidate_id"]
        by_c.setdefault(cid, {"persona_consistency": [], "honesty_cues": []})
        sc = r.get("scores", {})
        if "persona_consistency" in sc:
            by_c[cid]["persona_consistency"].append(sc["persona_consistency"])
        if "honesty_cues" in sc:
            by_c[cid]["honesty_cues"].append(sc["honesty_cues"])

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    summary = {"candidates": {}}
    for cid, d in by_c.items():
        summary["candidates"][cid] = {
            "avg_persona_consistency": avg(d["persona_consistency"]),
            "avg_honesty_cues": avg(d["honesty_cues"]),
            "n_answers": len(d["persona_consistency"]) or len(d["honesty_cues"]),
        }

    a_cfg = _cfg_to_dict(getattr(cfg.phase1.evaluation, "analysis", None))
    out_path = f"{run_dir}/{a_cfg.get('output_summary_json', 'phase1/summary.json')}"
    atomic_write_json(out_path, summary)
    return summary
