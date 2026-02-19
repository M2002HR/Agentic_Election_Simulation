from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from election_sim.phase1.candidates import build_persona_text


@dataclass
class CandidateAgent:
    candidate_id: str
    display_name: str

    def persona(self, cfg) -> str:
        cand_cfg = getattr(cfg.phase1.candidates, self.candidate_id)
        return build_persona_text(cfg, cand_cfg)

    def answer(self, cfg, llm_client, question: str) -> str:
        persona = self.persona(cfg)
        prompt = (
            cfg.phase1.prompts.system_template.strip()
            + "\n\n"
            + persona.strip()
            + "\n\n"
            + cfg.phase1.prompts.answer_template.format(question=question).strip()
        )
        meta = {"phase": "phase1", "role": "candidate", "candidate_id": self.candidate_id, "question": question}
        return llm_client.invoke(prompt, meta=meta)
