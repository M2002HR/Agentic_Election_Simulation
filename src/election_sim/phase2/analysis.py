from __future__ import annotations
from typing import Dict, Any, List
from election_sim.utils.io import atomic_write_json


def _cfg_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[attr-defined]
    return {}


def analyze_debate(cfg, run_dir: str, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    qa = [r for r in transcript if r.get("type") == "qa"]
    topics = sorted(set([r.get("topic") for r in qa if r.get("topic")]))
    summary = {
        "total_main_questions": len(qa),
        "topics_covered": topics,
    }
    p2 = _cfg_to_dict(getattr(cfg, "phase2", None))
    a_cfg = _cfg_to_dict(p2.get("analysis"))
    summary_path = a_cfg.get("output_summary_json", "phase2/debate_summary.json")
    analysis_path = a_cfg.get("output_analysis_json", "phase2/debate_analysis.json")

    atomic_write_json(f"{run_dir}/{summary_path}", summary)
    atomic_write_json(f"{run_dir}/{analysis_path}", {"records": transcript})

    return summary
