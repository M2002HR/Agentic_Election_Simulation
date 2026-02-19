from __future__ import annotations
from typing import Dict, Any, List
from election_sim.utils.io import atomic_write_json


def analyze_debate(cfg, run_dir: str, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    qa = [r for r in transcript if r.get("type") == "qa"]
    topics = sorted(set([r.get("topic") for r in qa if r.get("topic")]))
    summary = {
        "total_main_questions": len(qa),
        "topics_covered": topics,
    }

    atomic_write_json(f"{run_dir}/{cfg.phase2['analysis']['output_summary_json']}", summary)
    # keep full transcript as "analysis" artifact too (easy for report pipelines)
    atomic_write_json(f"{run_dir}/{cfg.phase2['analysis']['output_analysis_json']}", {"records": transcript})

    return summary
