import json
from pathlib import Path

from election_sim.config import load_config
from election_sim.phase4.runner import _scenario3, _validate_report_pack_schema, run_phase4
from election_sim.utils.logging import setup_logger


class _FakeLLM:
    def invoke(self, prompt: str, meta=None):
        if "Generate diverse US voter value profiles" in prompt:
            return json.dumps(
                {
                    "profiles": [
                        {
                            "china": "Contain China with alliances and selective pressure.",
                            "healthcare": "Affordable healthcare with competition and safeguards.",
                            "guns": "Responsible ownership plus background checks.",
                        },
                        {
                            "china": "Strong deterrence and industrial policy against China risks.",
                            "healthcare": "Reduce cost growth with measurable outcome targets.",
                            "guns": "Protect rights while enforcing illegal-use penalties.",
                        },
                    ]
                }
            )
        return json.dumps(
            {
                "choice": "democrat",
                "confidence": 80,
                "reason": "Test-mode vote based on synthetic alignment.",
            }
        )


def test_phase4_generates_report_pack(tmp_path: Path):
    cfg = load_config("config.yaml")
    cfg.project.run_dir_base = str(tmp_path)
    cfg.phase3.voters.count = 12
    cfg.phase3.values.pool_size = 12
    cfg.phase4.repeats = 2
    cfg.phase4.scenario_vote_mode = "deterministic"
    cfg.phase4.llm_validation_top_k = 2
    cfg.phase4.llm_validation_sample_size = 4

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "phase2").mkdir(parents=True, exist_ok=True)
    transcript_path = run_dir / cfg.phase3.debate_path
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            [
                {
                    "type": "qa",
                    "topic": "china",
                    "question": "What is your China strategy?",
                    "first_candidate": "democrat",
                    "second_candidate": "republican",
                    "first_answer": "Alliance-driven competition.",
                    "second_answer": "Deterrence and tariffs.",
                    "follow_up": "Give specific metrics.",
                }
            ]
        ),
        encoding="utf-8",
    )

    logger = setup_logger(str(run_dir))
    out = run_phase4(cfg, _FakeLLM(), run_dir, logger)

    report_pack_path = Path(out["report_pack_path"])
    assert report_pack_path.exists()
    pack = json.loads(report_pack_path.read_text(encoding="utf-8"))
    assert pack["version"] == "1.1"
    assert "scenario_outputs" in pack
    assert "scenario_1" in pack["scenario_outputs"]
    assert "scenario_5" in pack["scenario_outputs"]
    assert "key_metrics_table" in pack
    assert "scenario_4" in pack["key_metrics_table"]
    _validate_report_pack_schema(pack)


def test_scenario3_forced_high_wisdom_selection_is_seeded():
    cfg = load_config("config.yaml")
    cfg.phase3.voters.count = 200
    value_pool = [
        {"profile_id": f"vp_{i}", "china": f"c{i}", "healthcare": f"h{i}", "guns": f"g{i}"}
        for i in range(200)
    ]
    _, _, meta1 = _scenario3(cfg, value_pool, seed=42)
    _, _, meta2 = _scenario3(cfg, value_pool, seed=42)
    assert meta1["forced_high_wisdom_count"] == 100
    assert meta1["forced_voter_ids"] == meta2["forced_voter_ids"]


def test_phase4_can_run_single_selected_scenario(tmp_path: Path):
    cfg = load_config("config.quick.yaml")
    cfg.project.run_dir_base = str(tmp_path)
    cfg.phase3.voters.count = 12
    cfg.phase3.values.pool_size = 12
    cfg.phase4.repeats = 2
    cfg.phase4.scenario_vote_mode = "deterministic"

    run_dir = tmp_path / "run_single_s4"
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = run_dir / cfg.phase3.debate_path
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            [
                {
                    "type": "qa",
                    "topic": "healthcare",
                    "question": "How will you reduce costs?",
                    "first_candidate": "democrat",
                    "second_candidate": "republican",
                    "first_answer": "Outcome-based cost controls.",
                    "second_answer": "Competition and deregulation.",
                    "follow_up": "What metric in 24 months?",
                }
            ]
        ),
        encoding="utf-8",
    )
    logger = setup_logger(str(run_dir))

    out = run_phase4(cfg, _FakeLLM(), run_dir, logger, scenario_ids=["scenario_2"])
    report_pack_path = Path(out["report_pack_path"])
    pack = json.loads(report_pack_path.read_text(encoding="utf-8"))

    assert pack["scenario_outputs"]["scenario_2"].get("skipped") is not True
    assert pack["scenario_outputs"]["scenario_1"]["skip_reason"] == "not_selected"
    assert pack["scenario_outputs"]["scenario_3"]["skip_reason"] == "not_selected"
