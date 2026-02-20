import json
from pathlib import Path

from election_sim.config import load_config
from election_sim.phase5.runner import _validate_report_pack_schema, run_phase5
from election_sim.utils.io import write_json
from election_sim.utils.logging import setup_logger


class _FakeLLM:
    def invoke(self, prompt: str, meta=None):
        if "Return JSON ONLY with keys: choice" in prompt:
            return json.dumps(
                {
                    "choice": "democrat",
                    "confidence": 79,
                    "reason": "Synthetic validation vote for tests.",
                }
            )
        if "Generate diverse US voter value profiles" in prompt:
            return json.dumps(
                {
                    "profiles": [
                        {
                            "china": "Strong deterrence and tariffs against China.",
                            "healthcare": "Market-based reforms with safety nets.",
                            "guns": "Protect rights with strict enforcement on crime.",
                        },
                        {
                            "china": "Alliance-first competition with economic coordination.",
                            "healthcare": "Affordable access and stronger public options.",
                            "guns": "Background checks and safe ownership standards.",
                        },
                    ]
                }
            )
        return "ok"


def test_phase5_generates_scenarios_and_report(tmp_path: Path):
    cfg = load_config("config.quick.yaml")
    cfg.project.run_dir_base = str(tmp_path)
    cfg.phase3.voters.count = 16
    cfg.phase3.values.pool_size = 16
    cfg.phase5.repeats = 3
    cfg.phase5.scenario_vote_mode = "deterministic"
    cfg.phase5.llm_validation_sample_size = 5

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

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
                    "first_answer": "Alliance strategy with targeted competition.",
                    "second_answer": "Strong deterrence and tariffs.",
                    "follow_up": "How will you measure success?",
                }
            ]
        ),
        encoding="utf-8",
    )

    profiles = []
    for i in range(20):
        profiles.append(
            {
                "profile_id": f"vp_{i:03d}",
                "china": "Strong US deterrence against China." if i % 2 == 0 else "Alliance competition with China.",
                "healthcare": "Market reforms with competition." if i % 3 == 0 else "Affordable public access.",
                "guns": "Defend gun rights strongly." if i % 2 == 0 else "Background checks and safe ownership.",
            }
        )
    value_pool_path = run_dir / cfg.phase3.values.output_pool_json
    value_pool_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        value_pool_path,
        {
            "metadata": {"pool_size": len(profiles)},
            "profiles": profiles,
        },
    )

    logger = setup_logger(str(run_dir))
    out = run_phase5(cfg, _FakeLLM(), run_dir, logger)

    report_pack_path = Path(out["report_pack_path"])
    assert report_pack_path.exists()
    pack = json.loads(report_pack_path.read_text(encoding="utf-8"))
    _validate_report_pack_schema(pack)

    assert (run_dir / "phase5/scenario_6_republican_win.json").exists()
    assert (run_dir / "phase5/scenario_7_healthcare_shock.json").exists()
    assert (run_dir / "phase5/scenario_8_polarized_tossup.json").exists()
    assert (run_dir / "phase5/comparison.json").exists()

    s6 = pack["scenario_outputs"]["scenario_6_republican_win"]
    assert s6["estimated_winner"] == "republican"
    assert s6["simulation"]["win_rates"]["republican"] >= s6["simulation"]["win_rates"]["democrat"]


def test_phase5_can_run_single_selected_scenario(tmp_path: Path):
    cfg = load_config("config.quick.yaml")
    cfg.project.run_dir_base = str(tmp_path)
    cfg.phase3.voters.count = 12
    cfg.phase3.values.pool_size = 12
    cfg.phase5.repeats = 2
    cfg.phase5.scenario_vote_mode = "deterministic"
    cfg.phase5.llm_validation_sample_size = 4

    run_dir = tmp_path / "run_single_s5"
    run_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = run_dir / cfg.phase3.debate_path
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            [
                {
                    "type": "qa",
                    "topic": "guns",
                    "question": "What is your gun policy?",
                    "first_candidate": "democrat",
                    "second_candidate": "republican",
                    "first_answer": "Safety-focused responsible ownership.",
                    "second_answer": "Strong rights with enforcement.",
                    "follow_up": "How will you enforce it?",
                }
            ]
        ),
        encoding="utf-8",
    )
    profiles = [
        {
            "profile_id": f"vp_{i:03d}",
            "china": "Strategic competition with China.",
            "healthcare": "Affordable and stable healthcare.",
            "guns": "Responsible ownership with public safety.",
        }
        for i in range(12)
    ]
    value_pool_path = run_dir / cfg.phase3.values.output_pool_json
    value_pool_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(value_pool_path, {"metadata": {"pool_size": 12}, "profiles": profiles})

    logger = setup_logger(str(run_dir))
    out = run_phase5(
        cfg,
        _FakeLLM(),
        run_dir,
        logger,
        scenario_ids=["scenario_6_republican_win"],
    )
    pack = json.loads(Path(out["report_pack_path"]).read_text(encoding="utf-8"))

    assert pack["scenario_outputs"]["scenario_6_republican_win"].get("skipped") is not True
    assert pack["scenario_outputs"]["scenario_7_healthcare_shock"]["skip_reason"] == "not_selected"
    assert pack["scenario_outputs"]["scenario_8_polarized_tossup"]["skip_reason"] == "not_selected"
