import json
from pathlib import Path

from election_sim.config import load_config
from election_sim.phase1.eval import run_phase1
from election_sim.phase2.debate import run_debate
from election_sim.phase3.voting import run_phase3
from election_sim.phase4.runner import run_phase4
from election_sim.utils.logging import setup_logger


class _FakeLLM:
    def invoke(self, prompt: str, meta=None):
        p = prompt.lower()
        if 'output json only with this schema: {"questions": [..strings..]}' in p:
            return json.dumps(
                {
                    "questions": [
                        "What tradeoff will your policy make in year one?",
                        "Provide a timeline and budget estimate for your plan.",
                        "Which measurable metric proves success in 24 months?",
                        "What criticism of your policy is most valid?",
                        "If this plan fails, what fallback policy do you adopt?",
                    ]
                }
            )
        if "generate diverse us voter value profiles" in p:
            return json.dumps(
                {
                    "profiles": [
                        {
                            "china": "Compete with China via alliances and selective pressure.",
                            "healthcare": "Affordable coverage with fiscal safeguards.",
                            "guns": "Protect rights with strong background checks.",
                        },
                        {
                            "china": "Prioritize deterrence and industrial policy against China.",
                            "healthcare": "Use market competition with protections.",
                            "guns": "Expand responsible ownership training and enforcement.",
                        },
                    ]
                }
            )
        if "return json only with keys: choice" in p:
            return json.dumps(
                {
                    "choice": "democrat",
                    "confidence": 78,
                    "reason": "Synthetic vote for integration tests.",
                }
            )
        if "return json with keys: persona_consistency" in p:
            return json.dumps(
                {
                    "persona_consistency": 2,
                    "honesty_cues": 2,
                    "notes": "consistent persona",
                }
            )
        if "follow-up question:" in p:
            return "Can you commit to a measurable target by year one?"
        return (
            "Test response with enough detail to satisfy length constraints and keep execution deterministic."
        )


def test_full_pipeline_builds_core_artifacts(tmp_path: Path):
    cfg = load_config("config.yaml")
    cfg.project.run_dir_base = str(tmp_path)
    cfg.phase3.voters.count = 12
    cfg.phase3.values.pool_size = 12
    cfg.phase4.repeats = 2
    cfg.phase4.llm_validation_top_k = 2
    cfg.phase4.llm_validation_sample_size = 4

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(run_dir))
    llm = _FakeLLM()

    run_phase1(cfg, llm, str(run_dir), logger)
    run_debate(cfg, llm, str(run_dir), logger)
    run_phase3(cfg, llm, run_dir, logger)
    run_phase4(cfg, llm, run_dir, logger)

    expected = [
        "phase1/candidates.json",
        "phase1/eval.json",
        "phase1/summary.json",
        "phase2/debate.json",
        "phase2/debate_transcript.json",
        "phase3/value_pool.json",
        "phase3/voters.json",
        "phase3/votes.json",
        "phase3/vote_summary.json",
        "phase4/scenario_1.json",
        "phase4/scenario_5.json",
        "phase4/comparison.json",
        "phase4/optimization_trace.json",
        "phase4/report_pack.json",
    ]
    for rel in expected:
        assert (run_dir / rel).exists(), rel

    questions = json.loads((run_dir / "phase2/questions.json").read_text(encoding="utf-8"))
    assert questions.get("adaptive_main_questions") is True
