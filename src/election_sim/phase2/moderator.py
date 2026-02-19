from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class ModeratorAgent:
    traits: Dict[str, float]

    def persona(self) -> str:
        return (
            "You are a presidential debate moderator.\n"
            "You ask sharp, value-driven, challenging questions.\n"
            "You demand clarity, press on contradictions, and avoid vague answers.\n"
            f"Traits (0-10):\n"
            f"- confidence: {self.traits.get('confidence', 9)}\n"
            f"- diligence: {self.traits.get('diligence', 9)}\n"
            f"- aggressiveness: {self.traits.get('aggressiveness', 5)}\n"
        )

    def generate_question(self, llm, topic_title: str) -> str:
        prompt = (
            self.persona()
            + "\nGenerate ONE challenging question about this topic:\n"
            + f"{topic_title}\n"
            + "Question only."
        )
        return llm.invoke(prompt, meta={"phase": "phase2", "role": "moderator_question", "topic": topic_title})

    def follow_up(self, llm, question: str, answer: str) -> str:
        prompt = (
            self.persona()
            + "\nWrite ONE sharp follow-up question.\n"
            + "Use the previous question and the candidate's answer.\n\n"
            + f"Previous question:\n{question}\n\n"
            + f"Candidate answer:\n{answer}\n\n"
            + "Follow-up question only."
        )
        return llm.invoke(prompt, meta={"phase": "phase2", "role": "moderator_followup"})
