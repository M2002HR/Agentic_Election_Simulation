from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "agentic-election-sim"
    random_seed: int = 42
    run_dir_base: str = "runs"


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    trace_jsonl: str = "trace.jsonl"
    run_log: str = "run.log"


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    provider: Literal["google_genai"] = "google_genai"
    api_keys: Optional[list[str]] = None

    model_name: str = "gemini-1.5-flash"
    candidate_model_name: Optional[str] = None
    moderator_model_name: Optional[str] = None
    voter_model_name: Optional[str] = None

    temperature: float = 0.7
    max_output_tokens: int = 1024

    retry_on_429: bool = True
    rounds_limit: int = 2
    cooloff_sec: float = 30.0
    min_interval_sec: float = 0.2
    network_retries: int = 4
    network_backoff_sec: float = 1.5
    request_timeout_sec: float = 45.0


class CandidatePersona(BaseModel):
    model_config = ConfigDict(extra="allow")

    display_name: str
    party: str
    personal_traits: dict[str, float]
    policy_stances: dict[str, str]

    @model_validator(mode="after")
    def _validate_required_traits(self) -> "CandidatePersona":
        required = ["honesty", "aggressiveness", "confidence", "charisma"]
        missing = [k for k in required if k not in self.personal_traits]
        if missing:
            raise ValueError(f"Missing required candidate personal_traits: {missing}")
        for key in required:
            val = float(self.personal_traits[key])
            if val < 0 or val > 10:
                raise ValueError(f"Trait '{key}' must be in range 0..10")
        return self


class Phase1CandidatesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    democrat: CandidatePersona
    republican: CandidatePersona
    extra_traits: list[str] = Field(default_factory=list)


class Phase1PromptsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    system_template: str
    persona_template: str
    answer_template: str


class Phase1RubricItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    description: str


class Phase1RubricConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    persona_consistency: Phase1RubricItem
    honesty_cues: Phase1RubricItem


class Phase1OutputConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidates_json: str
    eval_json: str


class Phase1AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    mode: Literal["llm", "off", "heuristic"] = "off"
    max_words: int = 90
    prompt_template: str = ""
    output_analysis_json: str = "phase1/analysis.json"
    output_summary_json: str = "phase1/summary.json"


class Phase1HonestyStressConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    honesty_override: dict[str, float] = Field(
        default_factory=lambda: {"democrat": 2.0, "republican": 2.0}
    )
    questions: list[str] = Field(default_factory=list)
    output_json: str = "phase1/honesty_stress_test.json"


class Phase1EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    controversial_questions: list[str] = Field(default_factory=list)
    rubric: Phase1RubricConfig
    output: Phase1OutputConfig
    analysis: Phase1AnalysisConfig = Field(default_factory=Phase1AnalysisConfig)
    honesty_stress: Phase1HonestyStressConfig = Field(default_factory=Phase1HonestyStressConfig)


class Phase1Config(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidates: Phase1CandidatesConfig
    prompts: Phase1PromptsConfig
    evaluation: Phase1EvaluationConfig


class Phase2Topic(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    title: str


class Phase2ModeratorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "Moderator"
    traits: dict[str, float] = Field(default_factory=dict)


class Phase2QualityConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    pretest_questions: int = 3
    output_quality_json: str = "phase2/debate_quality.json"


class Phase2AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    output_analysis_json: str = "phase2/debate_analysis.json"
    output_summary_json: str = "phase2/debate_summary.json"


class Phase2Config(BaseModel):
    model_config = ConfigDict(extra="allow")

    topics: list[Phase2Topic] = Field(default_factory=list)
    questions_per_topic: int = 5
    answer_word_limit: int = 250
    moderator: Phase2ModeratorConfig = Field(default_factory=Phase2ModeratorConfig)
    quality: Phase2QualityConfig = Field(default_factory=Phase2QualityConfig)
    analysis: Phase2AnalysisConfig = Field(default_factory=Phase2AnalysisConfig)
    output_transcript_json: str = "phase2/debate_transcript.json"


class TraitDistributionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    low: float = 33.0
    medium: float = 34.0
    high: float = 33.0

    @model_validator(mode="after")
    def _validate_sum(self) -> "TraitDistributionConfig":
        s = float(self.low) + float(self.medium) + float(self.high)
        if s <= 0:
            raise ValueError("Trait distribution sum must be > 0")
        return self


class Phase3ValuesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    pool_size: int = 200
    output_pool_json: str = "phase3/value_pool.json"
    assignment_mode: Literal["seeded_random"] = "seeded_random"
    unique_assignment_when_possible: bool = True
    min_words_per_field: int = 15
    min_token_count_per_field: int = 6
    near_duplicate_avg_jaccard_threshold: float = 0.82
    near_duplicate_min_topic_threshold: float = 0.72
    sample_value_profile: dict[str, str] = Field(
        default_factory=lambda: {
            "china": (
                "America must remain the global leader and any foreign challenge to that status "
                "must be strongly countered, including military pressure if needed."
            ),
            "healthcare": (
                "Healthcare policy is acceptable as long as it does not weaken US national strength "
                "or fiscal stability."
            ),
            "guns": "I support carrying firearms to protect myself and my family.",
        }
    )


class Phase3VotersConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    count: int = 200
    traits: list[str] = Field(
        default_factory=lambda: ["wisdom", "fear", "anger", "adaptability", "distrust"]
    )
    trait_distributions: dict[str, TraitDistributionConfig] = Field(default_factory=dict)
    output_voters_json: str = "phase3/voters.json"
    output_votes_json: str = "phase3/votes.json"
    output_summary_json: str = "phase3/vote_summary.json"
    output_sample_analysis_json: str = "phase3/sample_voter_analysis.json"


class Phase3Config(BaseModel):
    model_config = ConfigDict(extra="allow")

    debate_path: str = "phase2/debate_transcript.json"
    values: Phase3ValuesConfig = Field(default_factory=Phase3ValuesConfig)
    voters: Phase3VotersConfig = Field(default_factory=Phase3VotersConfig)
    use_debate_digest: bool = True
    digest_max_chars_per_field: int = 320
    llm_value_pool_batch_size: int = 25
    max_llm_batches: int = 12


class Phase4Scenario(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    description: str
    enabled: bool = True
    overrides: dict[str, Any] = Field(default_factory=dict)


class Phase4Config(BaseModel):
    model_config = ConfigDict(extra="allow")

    repeats: int = 10
    scenario_vote_mode: Literal["deterministic", "llm_full"] = "deterministic"
    certainty_threshold: float = 0.9
    search_mode: Literal["hybrid", "full_llm", "fast_approx"] = "hybrid"
    llm_validation_top_k: int = 5
    llm_validation_sample_size: int = 24
    trait_level_mapping: dict[str, float] = Field(
        default_factory=lambda: {
            "very_low": 1.0,
            "low": 3.0,
            "medium": 5.0,
            "moderate": 6.0,
            "high": 8.0,
            "very_high": 9.5,
            "calm": 2.0,
            "aggressive": 8.5,
            "liar": 2.0,
            "honest": 9.0,
        }
    )
    scenarios: list[Phase4Scenario] = Field(
        default_factory=lambda: [
            Phase4Scenario(id="scenario_1", description="Baseline S1"),
            Phase4Scenario(id="scenario_2", description="Wisdom-shifted S2"),
            Phase4Scenario(id="scenario_3", description="S2 + forced value profile for 100 high-wisdom voters"),
            Phase4Scenario(id="scenario_4", description="Optimize democrat core traits under S1 electorate"),
            Phase4Scenario(id="scenario_5", description="Optimize voter trait distributions for democrat landslide"),
        ]
    )
    output_scenario_prefix: str = "phase4/scenario_"
    output_comparison_json: str = "phase4/comparison.json"
    output_optimization_trace_json: str = "phase4/optimization_trace.json"
    output_report_pack_json: str = "phase4/report_pack.json"
    debate_path: Optional[str] = None


class Phase5Scenario(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    description: str
    enabled: bool = True
    overrides: dict[str, Any] = Field(default_factory=dict)


class Phase5Config(BaseModel):
    model_config = ConfigDict(extra="allow")

    repeats: int = 10
    scenario_vote_mode: Literal["deterministic", "llm_full"] = "deterministic"
    llm_validation_sample_size: int = 24
    enable_llm_validation: bool = True
    confidence_shift_sensitivity: float = 8.0
    margin_pct_alert_threshold: float = 5.0
    scenarios: list[Phase5Scenario] = Field(
        default_factory=lambda: [
            Phase5Scenario(
                id="scenario_6_republican_win",
                description="Design electorate and candidate traits so Republican wins decisively.",
            ),
            Phase5Scenario(
                id="scenario_7_healthcare_shock",
                description="Healthcare-cost shock electorate with issue-salience shift.",
            ),
            Phase5Scenario(
                id="scenario_8_polarized_tossup",
                description="Highly polarized electorate with near tie and sensitivity stress test.",
            ),
        ]
    )
    output_scenario_prefix: str = "phase5/"
    output_comparison_json: str = "phase5/comparison.json"
    output_report_pack_json: str = "phase5/report_pack.json"


class Config(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    phase1: Phase1Config
    phase2: Phase2Config
    phase3: Phase3Config = Field(default_factory=Phase3Config)
    phase4: Phase4Config = Field(default_factory=Phase4Config)
    phase5: Phase5Config = Field(default_factory=Phase5Config)


def load_config(config_path: str | Path) -> Config:
    path = Path(config_path)
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # Back-compat: some configs put debate_path under phase4.
    phase3 = data.get("phase3") or {}
    if "debate_path" not in phase3:
        phase4 = data.get("phase4") or {}
        if isinstance(phase4, dict) and phase4.get("debate_path"):
            phase3["debate_path"] = phase4["debate_path"]
            data["phase3"] = phase3

    return Config(**data)


def dump_resolved_config(cfg: Config) -> str:
    payload = cfg.model_dump(exclude_none=True)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
