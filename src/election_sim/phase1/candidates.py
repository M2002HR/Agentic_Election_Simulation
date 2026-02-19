from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Candidate:
    candidate_id: str
    display_name: str
    party: str
    personal_traits: Dict[str, float]
    policy_stances: Dict[str, str]

def traits_block(traits: Dict[str, float], extra_traits: List[str]) -> str:
    # Ensure extra traits exist (default 5)
    out = dict(traits)
    for t in extra_traits:
        out.setdefault(t, 5)
    lines = [f"- {k}: {v}" for k, v in out.items()]
    return "\n".join(lines)

def build_persona_text(cfg, cand_cfg) -> str:
    block = traits_block(cand_cfg.personal_traits, cfg.phase1.candidates.extra_traits)
    return cfg.phase1.prompts.persona_template.format(
        display_name=cand_cfg.display_name,
        party=cand_cfg.party,
        traits_block=block,
        china=cand_cfg.policy_stances.get("china", ""),
        healthcare=cand_cfg.policy_stances.get("healthcare", ""),
        guns=cand_cfg.policy_stances.get("guns", ""),
    )

def build_candidate_objects(cfg) -> List[Candidate]:
    d = cfg.phase1.candidates.democrat
    r = cfg.phase1.candidates.republican
    return [
        Candidate("democrat", d.display_name, d.party, dict(d.personal_traits), dict(d.policy_stances)),
        Candidate("republican", r.display_name, r.party, dict(r.personal_traits), dict(r.policy_stances)),
    ]
