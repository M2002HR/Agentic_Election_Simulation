from election_sim.config import load_config

def test_load_config():
    cfg = load_config("config.yaml")
    assert cfg.project.name
    assert cfg.llm.model_name
    assert cfg.phase1.candidates.democrat.party == "Democrat"
