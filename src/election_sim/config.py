from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    provider: Literal["google_genai"] = "google_genai"
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 1024
    retry_on_429: bool = True
    rounds_limit: int = 2
    cooloff_sec: float = 30.0
    min_interval_sec: float = 0.2
    network_retries: int = 4
    network_backoff_sec: float = 1.5

class ProjectConfig(BaseModel):
    name: str = "agentic-election-sim"
    random_seed: int = 42
    run_dir_base: str = "runs"

class LoggingConfig(BaseModel):
    trace_jsonl: str = "trace.jsonl"
    run_log: str = "run.log"

class CandidatePersona(BaseModel):
    display_name: str
    party: str
    personal_traits: Dict[str, float]
    policy_stances: Dict[str, str]

class Phase1Candidates(BaseModel):
    democrat: CandidatePersona
    republican: CandidatePersona
    extra_traits: List[str] = Field(default_factory=list)

class Phase1Prompts(BaseModel):
    system_template: str
    persona_template: str
    answer_template: str

class Phase1EvaluationRubricItem(BaseModel):
    description: str

class Phase1EvaluationRubric(BaseModel):
    persona_consistency: Phase1EvaluationRubricItem
    honesty_cues: Phase1EvaluationRubricItem

class Phase1EvaluationOutput(BaseModel):
    candidates_json: str
    eval_json: str

class Phase1Evaluation(BaseModel):
    controversial_questions: List[str]
    rubric: Phase1EvaluationRubric
    output: Phase1EvaluationOutput
    analysis: Dict[str, Any] = Field(default_factory=dict)

class Phase1Config(BaseModel):
    candidates: Phase1Candidates
    prompts: Phase1Prompts
    evaluation: Phase1Evaluation

class AppConfig(BaseModel):
    project: ProjectConfig
    llm: LLMConfig
    logging: LoggingConfig
    phase1: Phase1Config
    # placeholders for later phases; keep raw dicts so config stays complete
    phase2: Dict[str, Any] = Field(default_factory=dict)
    phase3: Dict[str, Any] = Field(default_factory=dict)
    phase4: Dict[str, Any] = Field(default_factory=dict)

def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)

def dump_resolved_config(cfg: AppConfig) -> str:
    raw = cfg.model_dump(mode="python")
    return yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
