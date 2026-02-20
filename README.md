# Agentic Election Simulation

Multi-agent US election simulation pipeline implemented as 4 phases:

1. `phase1`: candidate persona simulation + evaluation + honesty stress test
2. `phase2`: full moderated debate simulation + question quality checks
3. `phase3`: voter simulation + value-pool generation + vote summary
4. `phase4`: scenarios 1..5 + comparative analysis + optimization + report pack

The project is designed for:
- reproducibility with fixed random seeds
- realistic execution under API quota/rate limits
- report-ready outputs (`phase4/report_pack.json`)

## What Is Implemented

## Phase 1
- Two candidate agents (Democrat / Republican)
- Required candidate traits are enforced: `honesty`, `aggressiveness`, `confidence`, `charisma`
- Persona-driven answers for controversial questions
- Heuristic or LLM-based analysis
- Honesty stress test with configurable honesty override

Artifacts:
- `phase1/candidates.json`
- `phase1/eval.json`
- `phase1/analysis.json`
- `phase1/summary.json`
- `phase1/honesty_stress_test.json`

## Phase 2
- Moderator persona with configurable traits
- Debate topics from config (default: China, Healthcare, Guns)
- Seed question generation and quality scoring
- Main-question flow supports adaptive follow-up sequencing per topic
- Candidate answers are word-limited and guarded with retry/truncation fallback
- Moderator follow-up and critique included per QA exchange

Artifacts:
- `phase2/questions.json`
- `phase2/debate.jsonl`
- `phase2/debate.json`
- `phase2/debate_transcript.json`
- `phase2/debate_quality.json`
- `phase2/debate_summary.json`

## Phase 3
- Value pool generation (`china`, `healthcare`, `guns`) with dedupe and fallback
- Robust parsing for partially malformed model JSON in value generation
- Configurable voter count and trait distributions
- Seeded value assignment to voters
- Voting with `choice`, `confidence`, `reason` and deterministic fallback
- Compact debate digest to reduce token usage

Artifacts:
- `phase3/value_pool.json`
- `phase3/voters.json`
- `phase3/votes.json`
- `phase3/vote_summary.json`
- `phase3/sample_voter_analysis.json`

## Phase 4
- Full scenario runner for scenario 1..5
- Scenario 1 and 2 distributions
- Scenario 3 forced-value assignment for selected high-wisdom voters (seeded)
- Scenario 4 optimization over Democrat core traits
- Scenario 5 optimization over voter trait distributions
- Search modes: `hybrid`, `fast_approx`, `full_llm`
- Report pack schema validation

Artifacts:
- `phase4/scenario_1.json` ... `phase4/scenario_5.json`
- `phase4/comparison.json`
- `phase4/optimization_trace.json`
- `phase4/report_pack.json`

## CLI Commands

The module entrypoint is `python -m election_sim`.

Available commands:
- `smoke-test`
- `phase1`
- `phase2`
- `phase3`
- `phase4`
- `all`

Common flags:
- `--config <path>`
- `--run-id <id>`

Examples:

```bash
PYTHONPATH=src python -m election_sim smoke-test --config config.yaml
PYTHONPATH=src python -m election_sim phase1 --config config.yaml
PYTHONPATH=src python -m election_sim phase2 --config config.yaml
PYTHONPATH=src python -m election_sim phase3 --config config.yaml
PYTHONPATH=src python -m election_sim phase4 --config config.yaml
PYTHONPATH=src python -m election_sim all --config config.yaml
```

Reuse a previous run directory:

```bash
PYTHONPATH=src python -m election_sim phase4 --config config.yaml --run-id 20260219_120000
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set API keys in `.env`:

```bash
GOOGLE_API_KEYS=key1,key2,key3
```

Notes:
- Multiple keys are supported and used with round-robin rotation.
- You can also set keys in config via `llm.api_keys`.

## Config Files

- `config.yaml`: main/full configuration
- `config.quick.yaml`: small/fast validation run

Recommended workflow:
1. Validate pipeline with `config.quick.yaml`
2. Run full experiment with `config.yaml`

## Important Config Knobs

## Project
- `project.random_seed`
- `project.run_dir_base`

## LLM
- `llm.model_name`
- `llm.candidate_model_name`
- `llm.moderator_model_name`
- `llm.voter_model_name`
- `llm.min_interval_sec`
- `llm.rounds_limit`
- `llm.cooloff_sec`
- `llm.network_retries`
- `llm.network_backoff_sec`
- `llm.request_timeout_sec`

## Phase 2
- `phase2.questions_per_topic`
- `phase2.answer_word_limit`
- `phase2.topics`

## Phase 3
- `phase3.values.pool_size`
- `phase3.llm_value_pool_batch_size`
- `phase3.voters.count`
- `phase3.voters.trait_distributions`

## Phase 4
- `phase4.repeats`
- `phase4.certainty_threshold`
- `phase4.search_mode`
- `phase4.llm_validation_top_k`
- `phase4.llm_validation_sample_size`
- `phase4.scenarios[].enabled`
- `phase4.scenarios[].overrides`

## Quick Validation Run

Clean quick runs, execute quick config, inspect outputs:

```bash
mkdir -p runs_quick
find runs_quick -mindepth 1 -delete
PYTHONPATH=src python -m election_sim all --config config.quick.yaml
```

Inspect latest quick run:

```bash
latest=$(ls -1dt runs_quick/*/ | head -n1)
echo "$latest"
python -m json.tool "${latest}phase3/value_pool.json" | head -n 60
python -m json.tool "${latest}phase4/report_pack.json" | head -n 80
```

## Full/Main Run

Before main run:

```bash
mkdir -p runs
find runs -mindepth 1 -delete
```

Run:

```bash
PYTHONPATH=src python -m election_sim all --config config.yaml
```

## Logs, Trace, and Progress

Each run directory contains:
- `run.log`: full run logs
- `trace.jsonl`: LLM call trace (prompt/output/error metadata)
- `config.resolved.yaml`: resolved config snapshot

Progress bars are shown for:
- phase1 evaluation/analysis/stress
- phase2 question generation/debate
- phase3 pipeline/value generation/voting
- phase4 scenario and optimization loops

Monitor live logs:

```bash
tail -f runs/<run_id>/run.log
```

## Output Directory Structure

Main run writes to `runs/<run_id>/` (or `runs_quick/<run_id>/` if configured):

- `phase1/*`
- `phase2/*`
- `phase3/*`
- `phase4/*`
- `run.log`
- `trace.jsonl`
- `config.resolved.yaml`

## Report Workflow

For final report generation, use:
- `phase4/report_pack.json`

This file includes:
- run metadata
- per-phase summaries (if available)
- scenario outputs (1..5)
- scenario comparison
- optimization trace reference
- key metrics table
- assumptions and limitations

## Testing

Run full test suite:

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

Tests cover:
- config schema checks
- key rotation behavior
- phase2 question parser robustness
- phase3 voter generation/distribution reproducibility
- phase3 malformed JSON value-pool parsing
- phase4 report pack schema and scenario behavior
- end-to-end fake-LLM pipeline smoke

## Troubleshooting

## 1) Missing API keys
Error:
- `No API keys found...`

Fix:
- set `GOOGLE_API_KEYS` in `.env`
- or configure `llm.api_keys` in yaml

## 2) Quota / 429 errors
Symptoms:
- repeated quota/rate-limit errors

Current behavior:
- round-robin key rotation
- configurable cooloff/backoff
- retry delay parsing from provider messages

Suggestions:
- add more API keys
- use `config.quick.yaml` for validation
- reduce load (`phase3.voters.count`, `phase3.values.pool_size`, `phase4.repeats`)

## 3) Slow or stuck network calls
Current safeguards:
- `llm.request_timeout_sec`
- transient network retries with exponential backoff

Tune in config:
- lower `llm.request_timeout_sec`
- adjust `llm.network_retries`

## 4) Malformed value-pool JSON from model
Current behavior:
- tolerant extraction salvages valid profile objects from partial JSON
- fallback generation fills remaining profiles to target size

## 5) Python warning from google api_core
You may see a future warning for Python 3.10 EOL support timeline.
Recommended:
- upgrade runtime to Python 3.11+

## Reproducibility Notes

- Seeds come from `project.random_seed`
- Scenario 3 high-wisdom selection is seeded
- Voter generation and value assignment support seeded reproducible behavior
- `config.resolved.yaml` captures exact effective config for each run

## Cleaning Run Directories

Delete all run artifacts:

```bash
mkdir -p runs runs_quick
find runs -mindepth 1 -delete
find runs_quick -mindepth 1 -delete
```

## Minimal Developer Loop

```bash
# 1) Quick validation
PYTHONPATH=src python -m election_sim all --config config.quick.yaml

# 2) Tests
PYTHONPATH=src .venv/bin/pytest -q

# 3) Main run
PYTHONPATH=src python -m election_sim all --config config.yaml
```
