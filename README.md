# Sandboxed Coding-Agent Eval Harness (Mini SWE-Bench)

A production-grade, local-first evaluation harness that benchmarks coding agents
across real multi-file repositories in isolated sandboxes, tracks pass@k, token
cost, latency, and failure modes, and visualizes results in an interactive
leaderboard and trace-replay UI.

Built per [`coding_agent_eval_harness_spec.md`](coding_agent_eval_harness_spec.md)
(Milestones 1–5), in roughly a 20–30 hour plan with automated verification
gates between milestones.

## Target model roster (pre-configured)

| Label | Model ID | Role |
| :--- | :--- | :--- |
| Gödel DeepSeek V4.1 Flash | `goedel/deepseek/deepseek-v4.1-flash` | Fast reasoning baseline |
| NVIDIA Nemotron 3.5 Lightning 30B | `nvidia/nvidia/nemotron-3.5-lightning-30b-a3b` | Low-latency MoE baseline |
| NVIDIA Nemotron 3.5 Super VL 120B | `nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b` | Heavyweight open-weights |
| OpenCode GLM 5.3 Flash Free | `opencode/glm-5.3-flash-free` | Free-tier contender |
| OpenCode MiniMax M3.1 Flash Free | `opencode/minimax-m3.1-flash-free` | Free-tier contender |
| OpenCode DeepSeek V4.1 Flash Free | `opencode/deepseek-v4.1-flash-free` | Free-tier contender |
| Gemini 3.6 Flash | `google/gemini-3.6-flash` | Frontier flash baseline |

Set the corresponding API key in your environment (see `configs/models.yaml` for
the `api_key_env` column of each model) before running live agent benchmarks.

## Directory layout

```
agent-eval-harness/
├── configs/
│   └── models.yaml          # model roster, endpoints, pricing, rate limits
├── benchmarks/
│   ├── schema.py            # TaskSpec (Pydantic v2)
│   └── tasks/               # 20 self-contained multi-file tasks
│       ├── py-*             # 8 Python tasks (stdlib only)
│       ├── ts-*             # 6 TypeScript/JS tasks (Node)
│       └── go-*             # 6 Go/Systems tasks (stdlib)
├── harness/
│   ├── cli.py               # Typer CLI (validate-suite, run, compare, export, serve)
│   ├── server.py            # FastAPI backend (/api/*)
│   ├── sandbox/             # Docker + git-worktree fallback isolation
│   ├── agent/               # tool-calling loop, client, tool registry
│   ├── grader/              # deterministic grader + failure taxonomy
│   └── telemetry/           # SQLite store + pass@k math
├── web/static/              # leaderboard + trace-replay dashboard
└── tests/                   # offline unit / integration tests (mock LLM server)
```

## Setup

```bash
# create the project (already done in this repo)
cd agent-eval-harness

# virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install pydantic==2.14.3 httpx typer fastapi uvicorn aiosqlite \
            pytest pytest-asyncio pyyaml

# the harness auto-selects Docker when the daemon is reachable; otherwise it
# falls back automatically to isolated git worktree + venv sandboxes.
```

No external internet is required during evaluation — every benchmark task is
self-contained and its tests run offline.

## Milestone verification gates

```bash
# Milestone 1: sandbox (timeout, patch apply/rollback, no leaks)
pytest tests/test_sandbox.py

# Milestone 2: agent loop (mock LLM server, offline)
pytest tests/test_agent_loop.py

# Milestone 3: grader, failure taxonomy, telemetry / pass@k
pytest tests/test_grader_and_telemetry.py

# Milestone 4: all 20 tasks fail on baseline / pass with golden patch
python -m harness.cli validate-suite

# Milestone 5: API endpoints
pytest tests/test_api.py

# everything together
pytest tests/
```

## CLI usage

```bash
# validate the full 20-task suite (Milestone 4 gate)
python -m harness.cli validate-suite

# run a single agent on a task (requires the model's API key)
python -m harness.cli run --model nvidia/nvidia/nemotron-3.5-lightning-30b-a3b \
                          --task py-cache-bug

# head-to-head pass/fail grid across models (from persisted telemetry)
python -m harness.cli compare --models model-a,model-b --task task-x

# export a results table for a GitHub README
python -m harness.cli export --format markdown   # also: csv | json

# serve the leaderboard API + web dashboard
python -m harness.cli serve   # http://127.0.0.1:8000
```

## Running a live comparison across the 7 models

1. Export the API keys for the models you want to benchmark:

```bash
export NVIDIA_API_KEY=...
export OPENAI_API_KEY=...   # for goedel/deepseek via OpenAI-compatible endpoint
export GEMINI_API_KEY=...
export OPENCODE_API_KEY=...
```

2. Run each model on each task and persist results:

```bash
for m in goedel/deepseek/deepseek-v4.1-flash \
         nvidia/nvidia/nemotron-3.5-lightning-30b-a3b \
         nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b \
         opencode/glm-5.3-flash-free \
         opencode/minimax-m3.1-flash-free \
         opencode/deepseek-v4.1-flash-free \
         google/gemini-3.6-flash; do
  for t in $(ls benchmarks/tasks | grep -E '^(py|ts|go)-'); do
    python -m harness.cli run --model "$m" --task "$t"
  done
done
```

*(Keep concurrency low — the harness defaults to one sandbox at a time so your
laptop stays responsive.)*

3. View the results:

```bash
# dashboard
python -m harness.cli serve        # http://127.0.0.1:8000

# markdown table for a README
python -m harness.cli export --format markdown > BENCHMARK_RESULTS.md
```

The web dashboard shows a model leaderboard (pass@1, avg cost, avg latency, avg
steps, failure breakdown chart), a head-to-head matrix, a run list, and a
step-by-step trajectory replay with a diff viewer.

## Architecture notes

* **Sandboxing:** Docker (`--cpus="1.5"`, `--memory="1536m"`, `--network=none`
  during grading) with an automatic `git worktree` + isolated `venv` fallback
  when the Docker daemon is unavailable (as in this environment).
* **Telemetry:** SQLite (WAL mode) persists every run, step, tool call, diff,
  and metric; pass@k uses the unbiased estimator.
* **Failure taxonomy:** `patch_syntax_error`, `test_regression`, `timeout`,
  `step_limit_exceeded`, `context_overflow`, `api_error`.
* **Concurrency:** `max_concurrent_sandboxes=1` by default; run the test suite
  with `pytest -n 2` max to keep laptop load minimal.

## Tests

All tests run offline against a local mock OpenAI-compatible server — no API
credits are consumed:

```
pytest tests/          # 28 tests, milestones 1–3 + API
```
