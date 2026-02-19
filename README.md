# agentic-election-sim

A Python project for simulating a US-style election with agents backed by a Google Gemini model.
The code keeps parameters in `config.yaml` and rotates multiple Google API keys in round-robin style.

## Repository layout

- `src/election_sim/` core package
- `runs/` runtime artifacts (ignored by git)
- `config.yaml` single source of truth for tunable parameters
- `.env` stores `GOOGLE_API_KEYS` (ignored)

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add comma-separated keys:

```bash
cp .env.example .env
# edit .env
```

## Run

Smoke test (one short call):

```bash
python -m election_sim smoke-test --config config.yaml
```

Phase 1 (candidates + evaluation):

```bash
python -m election_sim phase1 --config config.yaml
```

## Outputs

Each run creates `runs/<run_id>/`:

- `config.resolved.yaml`
- `trace.jsonl`
- `run.log`
- `phase1/candidates.json`
- `phase1/eval.json`
- `run_state.json` (per-run key index state)
