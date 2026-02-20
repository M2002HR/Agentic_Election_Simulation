import pytest

from election_sim.config import CandidatePersona, load_config


def test_load_config_extended_schema():
    cfg = load_config("config.yaml")
    assert cfg.project.name
    assert cfg.llm.model_name
    assert cfg.phase3.values.pool_size > 0
    assert cfg.phase3.voters.count > 0
    assert cfg.phase4.repeats >= 1
    assert 0 < cfg.phase4.certainty_threshold <= 1
    assert cfg.phase4.search_mode in {"hybrid", "full_llm", "fast_approx"}
    assert cfg.phase5.repeats >= 1
    assert cfg.phase5.llm_validation_sample_size >= 1
    assert isinstance(cfg.phase5.enable_llm_validation, bool)


def test_candidate_persona_requires_core_traits():
    with pytest.raises(ValueError):
        CandidatePersona(
            display_name="X",
            party="Y",
            personal_traits={"honesty": 8, "aggressiveness": 4},
            policy_stances={"china": "", "healthcare": "", "guns": ""},
        )
