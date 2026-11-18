"""Command-line interface for the Sandboxed Coding-Agent Eval Harness.

Commands:
* validate-suite  (Milestone 4 gate) - every task fails on baseline / passes
                  with its golden patch.
* run             (Milestone 5) - run one agent on one task and persist it.
* compare         (Milestone 5) - head-to-head model comparison table.
* export          (Milestone 5) - markdown/csv/json results table.
* serve           (Milestone 5) - FastAPI + uvicorn dashboard server.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import time
from pathlib import Path
from typing import Optional

import typer
import yaml
import uvicorn

from benchmarks.schema import TaskSpec
from harness.agent import AgentLoop, ModelClient, ModelRegistry, TOOL_SCHEMAS
from harness.agent.loop import Trajectory
from harness.grader import Grader
from harness.sandbox import SandboxManager
from harness.server import create_app
from harness.telemetry.store import TelemetryStore

app = typer.Typer(help="Sandboxed Coding-Agent Eval Harness")

TASKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "tasks"
DEFAULT_DB = Path(__file__).resolve().parents[2] / "agent-eval.db"


# --------------------------------------------------------------------------- #
# Task discovery
# --------------------------------------------------------------------------- #
def discover_tasks() -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    if not TASKS_DIR.exists():
        typer.echo(f"tasks dir not found: {TASKS_DIR}", err=True)
        raise typer.Exit(1)
    for task_dir in sorted(p for p in TASKS_DIR.iterdir() if p.is_dir()):
        yaml_path = task_dir / "task.yaml"
        if not yaml_path.exists():
            continue
        data = yaml.safe_load(yaml_path.read_text()) or {}
        data["id"] = data.get("id") or task_dir.name
        tasks.append(TaskSpec(**data))
    return tasks


def golden_patch_path(task_id: str) -> Path:
    return TASKS_DIR / task_id / "golden.patch"


# --------------------------------------------------------------------------- #
# validate-suite  (Milestone 4 gate)
# --------------------------------------------------------------------------- #
@app.command()
def validate_suite(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Verify that every task fails on baseline and passes with its golden patch."""
    tasks = discover_tasks()
    typer.echo(f"Validating {len(tasks)} tasks ...")
    failures: list[str] = []

    with SandboxManager(max_concurrent_sandboxes=1) as mgr:
        for task in tasks:
            task_id = task.id
            patch_path = golden_patch_path(task_id)
            if not patch_path.exists():
                failures.append(f"{task_id}: missing golden.patch")
                if verbose:
                    typer.echo(f"  [error] {task_id}: missing golden.patch")
                continue

            sid = mgr.create(task_id)
            baseline_res = mgr.exec(
                sid, task.test_cmd or "true", timeout_sec=task.timeout_sec
            )
            baseline_fails = baseline_res.exit_code != 0 or baseline_res.timed_out

            patch = patch_path.read_text(encoding="utf-8")
            try:
                mgr.apply_patch(sid, patch)
            except Exception as exc:
                failures.append(f"{task_id}: golden patch rejected: {exc}")
                mgr.destroy(sid)
                if verbose:
                    typer.echo(f"  [error] {task_id}: golden patch rejected")
                continue

            fixed_res = mgr.exec(
                sid, task.test_cmd or "true", timeout_sec=task.timeout_sec
            )
            fixed_passes = fixed_res.exit_code == 0 and not fixed_res.timed_out
            mgr.destroy(sid)

            if not (baseline_fails and fixed_passes):
                parts = []
                if not baseline_fails:
                    parts.append("baseline did not fail")
                if not fixed_passes:
                    parts.append("golden patch did not pass")
                failures.append(f"{task_id}: " + "; ".join(parts))
                if verbose:
                    typer.echo(f"  [fail] {task_id}: {'; '.join(parts)}")

    if failures:
        typer.echo(
            f"validate-suite FAILED: {len(failures)}/{len(tasks)} tasks not satisfied:"
        )
        for f in failures:
            typer.echo(f"  - {f}")
        raise typer.Exit(1)
    typer.echo(
        f"validate-suite PASSED: all {len(tasks)} tasks fail on baseline and "
        f"pass with their golden patch."
    )


# --------------------------------------------------------------------------- #
# run  (Milestone 5)
# --------------------------------------------------------------------------- #
@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model id from configs/models.yaml"),
    task: str = typer.Option(..., "--task", "-t", help="Task id under benchmarks/tasks/"),
    max_steps: int = typer.Option(30, help="Max agent steps"),
    db: str = typer.Option(str(DEFAULT_DB), help="Telemetry sqlite path"),
) -> None:
    """Run one agent on one task and persist the run to the telemetry store."""
    task_spec: Optional[TaskSpec] = None
    for t in discover_tasks():
        if t.id == task:
            task_spec = t
            break
    if task_spec is None:
        typer.echo(f"unknown task: {task}", err=True)
        raise typer.Exit(1)

    registry = ModelRegistry()
    try:
        spec = registry.get(model)
    except KeyError:
        typer.echo(f"unknown model: {model}", err=True)
        raise typer.Exit(1)

    async def _run_once() -> None:
        with SandboxManager(max_concurrent_sandboxes=1) as mgr:
            sid = mgr.create(task)
            client = ModelClient(registry=registry)
            loop = AgentLoop(
                client, mgr, sid, max_steps=max_steps, model_id=model
            )
            traj: Trajectory = await loop.run(task_spec.prompt)
            grader = Grader(mgr)
            result = grader.grade(task_spec, sid, traj)

        store = TelemetryStore(db_path=db)
        run_id = f"{task}-{model}-{int(time.time())}"
        steps = traj.to_dict()["steps"]
        store.persist_trajectory(
            run_id=run_id,
            task_id=task,
            model_id=model,
            status="passed" if result.passed else "failed",
            failure=result.failure,
            pass1=result.passed,
            started_at=traj.steps[0].timestamp if traj.steps else time.time(),
            finished_at=time.time(),
            duration_sec=result.duration_sec,
            total_steps=traj.total_steps,
            total_input_tokens=traj.total_input_tokens,
            total_output_tokens=traj.total_output_tokens,
            total_cost_usd=traj.total_cost_usd,
            total_latency_sec=traj.total_latency_sec,
            final_answer=traj.final_answer,
            final_diff=traj.final_diff,
            trajectory_json=traj.to_json(),
            steps=steps,
        )
        store.close()
        typer.echo(
            f"run {run_id}: passed={result.passed} failure={result.failure} "
            f"steps={traj.total_steps} cost=${traj.total_cost_usd:.4f}"
        )

    asyncio.run(_run_once())


# --------------------------------------------------------------------------- #
# compare  (Milestone 5)
# --------------------------------------------------------------------------- #
@app.command()
def compare(
    models: str = typer.Option(
        ..., "--models", help="Comma-separated model ids"
    ),
    task: Optional[str] = typer.Option(None, "--task", help="Filter to one task"),
    db: str = typer.Option(str(DEFAULT_DB), help="Telemetry sqlite path"),
) -> None:
    """Print a head-to-head pass/fail grid across models and tasks."""
    store = TelemetryStore(db_path=db)
    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    runs = store.list_runs(task_id=task, model_id=model_ids[0] if len(model_ids) == 1 else None)
    store.close()
    if model_ids:
        runs = [r for r in runs if r.model_id in model_ids]

    tasks: dict[str, dict[str, bool]] = {}
    seen: set[str] = set()
    for r in runs:
        tasks.setdefault(r.task_id, {})
        tasks[r.task_id][r.model_id] = r.pass1
        seen.add(r.model_id)

    header = ["task"] + sorted(seen)
    typer.echo(" | ".join(header))
    typer.echo("-" * (len(header) * 8))
    for tid in sorted(tasks):
        row = [tid] + [("PASS" if tasks[tid].get(m) else "FAIL") if m in tasks[tid] else "-" for m in sorted(seen)]
        typer.echo(" | ".join(row))


# --------------------------------------------------------------------------- #
# export  (Milestone 5)
# --------------------------------------------------------------------------- #
@app.command()
def export(
    format: str = typer.Option("markdown", "--format", "-f", help="markdown|csv|json"),
    db: str = typer.Option(str(DEFAULT_DB), help="Telemetry sqlite path"),
    model: Optional[str] = typer.Option(None, "--model"),
    task: Optional[str] = typer.Option(None, "--task"),
) -> None:
    """Export results as a markdown/csv/json table (suitable for a README)."""
    store = TelemetryStore(db_path=db)
    runs = store.list_runs(task_id=task, model_id=model)
    store.close()

    if format == "json":
        typer.echo(
            json.dumps(
                [
                    {
                        "run_id": r.run_id,
                        "task_id": r.task_id,
                        "model_id": r.model_id,
                        "passed": r.pass1,
                        "failure": r.failure,
                        "steps": r.total_steps,
                        "cost_usd": r.total_cost_usd,
                        "latency_sec": r.total_latency_sec,
                    }
                    for r in runs
                ],
                indent=2,
            )
        )
        return

    rows = [
        [r.run_id, r.task_id, r.model_id, "PASS" if r.pass1 else "FAIL",
         r.failure or "-", r.total_steps, f"{r.total_cost_usd:.4f}",
         f"{r.total_latency_sec:.3f}"]
        for r in runs
    ]

    if format == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(
            ["run_id", "task_id", "model_id", "passed", "failure", "steps",
             "cost_usd", "latency_sec"]
        )
        w.writerows(rows)
        typer.echo(buf.getvalue())
        return

    # markdown
    lines = ["| run_id | task_id | model_id | passed | failure | steps | cost_usd | latency_sec |",
             "|---|---|---|---|---|---|---|---|"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows
    ]
    typer.echo("\n".join(lines))


# --------------------------------------------------------------------------- #
# serve  (Milestone 5)
# --------------------------------------------------------------------------- #
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    db: str = typer.Option(str(DEFAULT_DB), help="Telemetry sqlite path"),
) -> None:
    """Serve the leaderboard API and web dashboard."""
    typer.echo(f"Serving dashboard at http://{host}:{port} (db={db})")
    uvicorn.run(create_app(db_path=db), host=host, port=port)


if __name__ == "__main__":
    app()
