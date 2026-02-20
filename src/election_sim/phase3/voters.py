from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class Voter:
    voter_id: int
    traits: dict[str, int]
    values: dict[str, str]
    value_profile_id: str
    source_scenario: str = "phase3_default"

    def vote(
        self,
        llm,
        debate_digest: Any,
        candidate_profiles: dict[str, Any] | None,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        traits_block = "\n".join(f"- {k}: {v}/10" for k, v in sorted(self.traits.items()))
        values_block = "\n".join(f"- {k}: {v}" for k, v in sorted(self.values.items()))
        candidate_block = ""
        if candidate_profiles:
            rows: list[str] = []
            for cid, p in candidate_profiles.items():
                t = p.get("traits", {})
                rows.append(
                    (
                        f"* {cid}: honesty={t.get('honesty')}, aggressiveness={t.get('aggressiveness')}, "
                        f"confidence={t.get('confidence')}, charisma={t.get('charisma')}."
                    )
                )
            candidate_block = "Candidate trait profile:\n" + "\n".join(rows) + "\n\n"

        prompt = (
            "You are simulating a US voter choosing between two candidates after watching a debate.\n"
            "Use voter traits and voter values to justify your choice.\n\n"
            "Voter traits (0-10):\n"
            f"{traits_block}\n\n"
            "Voter values:\n"
            f"{values_block}\n\n"
            f"{candidate_block}"
            "Debate digest (compact JSON):\n"
            f"{json.dumps(debate_digest, ensure_ascii=False)}\n\n"
            "Task:\n"
            "Pick exactly one: democrat or republican.\n"
            "Return JSON ONLY with keys: choice (democrat|republican), confidence (0-100), reason (1-3 sentences).\n"
        )

        try:
            raw = llm.invoke(prompt, meta=meta)
            data = _extract_json(raw)
            choice = str(data.get("choice", "")).strip().lower()
            if choice not in {"democrat", "republican"}:
                raise ValueError("invalid choice")

            conf = data.get("confidence", 50)
            conf_i = max(0, min(100, int(conf)))
            reason = str(data.get("reason", "")).strip()
            if not reason:
                raise ValueError("empty reason")
        except Exception:
            det = deterministic_vote(self, candidate_profiles or {})
            choice = det["choice"]
            conf_i = det["confidence"]
            reason = det["reason"]

        return {
            "voter_id": self.voter_id,
            "traits": self.traits,
            "values": self.values,
            "value_profile_id": self.value_profile_id,
            "source_scenario": self.source_scenario,
            "choice": choice,
            "confidence": conf_i,
            "reason": reason,
        }


def _extract_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("No JSON object found in model output")


def trait_band(score: int) -> str:
    if score <= 3:
        return "low"
    if score <= 6:
        return "medium"
    return "high"


def _distribution_counts(
    count: int,
    low_ratio: float,
    medium_ratio: float,
    high_ratio: float,
) -> tuple[int, int, int]:
    total = low_ratio + medium_ratio + high_ratio
    if total <= 0:
        return 0, 0, count
    low = int(round(count * (low_ratio / total)))
    medium = int(round(count * (medium_ratio / total)))
    high = count - low - medium
    if high < 0:
        high = 0
    while low + medium + high < count:
        high += 1
    while low + medium + high > count and high > 0:
        high -= 1
    return low, medium, high


def _sample_trait_values(
    count: int,
    dist: dict[str, Any],
    rng: random.Random,
) -> list[int]:
    low, medium, high = _distribution_counts(
        count,
        float(dist.get("low", 33)),
        float(dist.get("medium", 34)),
        float(dist.get("high", 33)),
    )
    values: list[int] = []
    values.extend(rng.randint(0, 3) for _ in range(low))
    values.extend(rng.randint(4, 6) for _ in range(medium))
    values.extend(rng.randint(7, 10) for _ in range(high))
    rng.shuffle(values)
    return values


def generate_voters(
    count: int,
    trait_names: list[str],
    seed: int = 0,
    trait_distributions: dict[str, Any] | None = None,
    value_pool: list[dict[str, Any]] | None = None,
    assignment_mode: str = "seeded_random",
    source_scenario: str = "phase3_default",
    unique_assignment_when_possible: bool = True,
) -> list[Voter]:
    rng = random.Random(seed)
    trait_distributions = trait_distributions or {}
    value_pool = value_pool or []
    voters: list[Voter] = []

    sampled_by_trait: dict[str, list[int]] = {}
    for trait in trait_names:
        if trait in trait_distributions:
            sampled_by_trait[trait] = _sample_trait_values(count, trait_distributions[trait], rng)
        else:
            sampled_by_trait[trait] = [rng.randint(0, 10) for _ in range(count)]

    value_assignment_indices: list[int] = []
    if value_pool and assignment_mode == "seeded_random":
        if unique_assignment_when_possible and len(value_pool) >= count:
            # When possible, assign each voter a unique value profile for better diversity.
            value_assignment_indices = list(range(len(value_pool)))
            rng.shuffle(value_assignment_indices)
            value_assignment_indices = value_assignment_indices[:count]
        else:
            # Reuse profiles only when pool is smaller than voter count.
            value_assignment_indices = list(range(len(value_pool)))
            rng.shuffle(value_assignment_indices)
            while len(value_assignment_indices) < count:
                value_assignment_indices.append(rng.randrange(len(value_pool)))

    for i in range(count):
        traits = {trait: sampled_by_trait[trait][i] for trait in trait_names}
        if value_pool and assignment_mode == "seeded_random":
            idx = value_assignment_indices[i]
            profile = value_pool[idx]
        elif value_pool:
            profile = value_pool[i % len(value_pool)]
        else:
            profile = {
                "profile_id": f"default_{i}",
                "china": "Prioritize stability and pragmatic competition with China.",
                "healthcare": "Affordable and accessible healthcare is important.",
                "guns": "Responsible gun ownership should coexist with public safety.",
            }
        voters.append(
            Voter(
                voter_id=i,
                traits=traits,
                values={
                    "china": str(profile.get("china", "")),
                    "healthcare": str(profile.get("healthcare", "")),
                    "guns": str(profile.get("guns", "")),
                },
                value_profile_id=str(profile.get("profile_id", f"profile_{i}")),
                source_scenario=source_scenario,
            )
        )
    return voters


def _value_alignment_score(values: dict[str, str], stance: dict[str, str]) -> float:
    score = 0.0
    for k in ["china", "healthcare", "guns"]:
        v_text = values.get(k, "").lower()
        s_text = stance.get(k, "").lower()
        if not v_text or not s_text:
            continue
        v_tokens = {t for t in v_text.replace(".", " ").replace(",", " ").split() if len(t) > 3}
        s_tokens = {t for t in s_text.replace(".", " ").replace(",", " ").split() if len(t) > 3}
        overlap = len(v_tokens & s_tokens)
        score += min(2.0, overlap * 0.3)
    return score


def deterministic_vote(voter: Voter, candidate_profiles: dict[str, Any]) -> dict[str, Any]:
    dem = candidate_profiles.get("democrat") or {}
    rep = candidate_profiles.get("republican") or {}
    dem_t = dem.get("traits", {})
    rep_t = rep.get("traits", {})
    dem_s = dem.get("policy_stances", {})
    rep_s = rep.get("policy_stances", {})

    wisdom = int(voter.traits.get("wisdom", 5))
    fear = int(voter.traits.get("fear", 5))
    anger = int(voter.traits.get("anger", 5))
    adaptability = int(voter.traits.get("adaptability", 5))
    distrust = int(voter.traits.get("distrust", 5))

    def base_score(t: dict[str, Any]) -> float:
        honesty = float(t.get("honesty", 5))
        aggr = float(t.get("aggressiveness", 5))
        conf = float(t.get("confidence", 5))
        char = float(t.get("charisma", 5))
        return 0.38 * honesty - 0.08 * aggr + 0.24 * conf + 0.22 * char

    dem_score = base_score(dem_t)
    rep_score = base_score(rep_t)

    # Wisdom and distrust generally reward honesty/coherence.
    dem_score += 0.05 * wisdom * float(dem_t.get("honesty", 5)) / 10
    rep_score += 0.05 * wisdom * float(rep_t.get("honesty", 5)) / 10
    dem_score += 0.04 * distrust * float(dem_t.get("honesty", 5)) / 10
    rep_score += 0.04 * distrust * float(rep_t.get("honesty", 5)) / 10

    # Fear and anger can reward aggressive/confident rhetoric.
    dem_score += 0.03 * fear * float(dem_t.get("confidence", 5)) / 10
    rep_score += 0.03 * fear * float(rep_t.get("confidence", 5)) / 10
    dem_score += 0.02 * anger * float(dem_t.get("aggressiveness", 5)) / 10
    rep_score += 0.02 * anger * float(rep_t.get("aggressiveness", 5)) / 10

    # Adaptability favors moderate aggressiveness.
    dem_score += 0.02 * adaptability * (1.0 - abs(float(dem_t.get("aggressiveness", 5)) - 5) / 5)
    rep_score += 0.02 * adaptability * (1.0 - abs(float(rep_t.get("aggressiveness", 5)) - 5) / 5)

    dem_score += _value_alignment_score(voter.values, dem_s)
    rep_score += _value_alignment_score(voter.values, rep_s)

    margin = dem_score - rep_score
    choice = "democrat" if margin >= 0 else "republican"
    conf = int(max(50, min(95, 50 + abs(margin) * 7)))
    reason = (
        "Deterministic fallback selected based on trait-value alignment and candidate trait signals."
    )
    return {"choice": choice, "confidence": conf, "reason": reason}
