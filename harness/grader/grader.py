"""Deterministic grader and failure taxonomy.

The :class:`Grader` restores hidden evaluation tests into the sandbox *after*
the agent has finished (so the agent cannot cheat or edit them), runs the
task's ``test_cmd``, and captures structured pass/fail counts.

The :class:`FailureClassifier` tags failed runs with one of:
``patch_syntax_error``, ``test_regression``, ``timeout``,
``step_limit_exceeded``, ``context_overflow``, ``api_error``.
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from benchmarks.schema import TaskSpec
from harness.sandbox import SandboxError, SandboxManager
from harness.agent.loop import Trajectory


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """Outcome of grading a single agent run on a single task."""

    task_id: str
    model_id: str
    passed: bool
    pass1: bool  # alias of passed for pass@k math
    passed_count: int = 0
    failed_count: int = 0
    total: int = 0
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_sec: float = 0.0
    failure: Optional[str] = None  # taxonomy tag when not passed
    hidden_tests_restored: bool = False


# --------------------------------------------------------------------------- #
# Failure taxonomy classifier
# --------------------------------------------------------------------------- #
class FailureClassifier:
    """Classifies why a run failed into a single taxonomy tag."""

    TAXONOMY = (
        "patch_syntax_error",
        "test_regression",
        "timeout",
        "step_limit_exceeded",
        "context_overflow",
        "api_error",
    )

    def classify(
        self,
        trajectory: Trajectory,
        result: RunResult,
        exception: Optional[BaseException] = None,
    ) -> str:
        # 1. Explicit loop / step budget abort from the agent loop.
        if trajectory.failure == "step_limit_exceeded":
            return "step_limit_exceeded"
        if trajectory.failure == "api_error":
            return "api_error"

        # 2. A patch was syntactically invalid / rejected.
        for step in trajectory.steps:
            res = step.tool_result
            if res is None:
                continue
            err = res.get("error") if isinstance(res, dict) else res.error
            out = res.get("output") if isinstance(res, dict) else res.output
            if err and "patch_syntax_error" in str(err):
                return "patch_syntax_error"
        if result.stdout and "patch_syntax_error" in result.stdout:
            return "patch_syntax_error"
        if result.stderr and "patch_syntax_error" in result.stderr:
            return "patch_syntax_error"

        # 3. A command timed out.
        if result.exit_code == -1 or "timed out" in (result.stdout + result.stderr):
            return "timeout"
        if exception is not None and isinstance(exception, TimeoutError):
            return "timeout"

        # 4. Context overflow: a tool result was truncated to fit the window.
        for step in trajectory.steps:
            res = step.tool_result
            if res is None:
                continue
            out = res.get("output") if isinstance(res, dict) else res.output
            if out and "[output truncated:" in out:
                return "context_overflow"

        # 5. Tests ran but regressed / failed.
        if result.failed_count and result.failed_count > 0:
            return "test_regression"
        if not result.passed and result.exit_code != 0:
            return "test_regression"

        # 6. Unhandled exception from the agent loop.
        if exception is not None:
            return "api_error"

        return "test_regression"


def classify_failure(
    trajectory: Trajectory, result: RunResult,
    exception: Optional[BaseException] = None,
) -> str:
    return FailureClassifier().classify(trajectory, result, exception)


# --------------------------------------------------------------------------- #
# Grader
# --------------------------------------------------------------------------- #
class Grader:
    def __init__(
        self,
        sandbox_manager: SandboxManager,
        *,
        restore_hidden_tests: bool = True,
    ) -> None:
        self.mgr = sandbox_manager
        self.restore_hidden_tests = restore_hidden_tests
        self.classifier = FailureClassifier()

    def grade(
        self,
        task: TaskSpec,
        sandbox_id: str,
        trajectory: Trajectory,
        *,
        exception: Optional[BaseException] = None,
    ) -> RunResult:
        started = time.time()
        result = RunResult(task_id=task.id, model_id=trajectory.steps[0].model
                           if trajectory.steps else "", passed=False, pass1=False)

        # Restore hidden evaluation tests into the sandbox *after* the agent
        # has finished, so the agent cannot read, edit, or game them.
        if self.restore_hidden_tests and task.hidden_tests:
            result.hidden_tests_restored = self._restore_hidden_tests(task, sandbox_id)

        # Run the evaluation command.
        if task.test_cmd:
            res = self.mgr.exec(sandbox_id, task.test_cmd,
                                timeout_sec=task.timeout_sec)
            result.exit_code = res.exit_code
            result.stdout = res.stdout
            result.stderr = res.stderr
            result.duration_sec = res.duration_sec
            if res.timed_out:
                result.exit_code = -1
            passed_count, failed_count, total = self._parse_counts(res.stdout, res.stderr)
            result.passed_count = passed_count
            result.failed_count = failed_count
            result.total = total
            result.passed = (res.exit_code == 0 and not res.timed_out
                             and failed_count == 0)
        else:
            # No test command: treat the run as passed if no explicit failure.
            result.passed = trajectory.failure is None

        result.pass1 = result.passed
        result.duration_sec = time.time() - started

        if not result.passed:
            result.failure = self.classifier.classify(trajectory, result, exception)

        return result

    # -- internals ---------------------------------------------------------- #
    def _restore_hidden_tests(self, task: TaskSpec, sandbox_id: str) -> bool:
        try:
            handle = self.mgr._get(sandbox_id)
        except SandboxError:
            return False
        repo_root = handle.worktree_path or handle.root
        task_dir = task.task_dir
        restored_any = False
        for rel in task.hidden_tests or []:
            src = task_dir / rel
            if not src.exists():
                continue
            dst = repo_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored_any = True
        return restored_any

    @staticmethod
    def _parse_counts(stdout: str, stderr: str = "") -> tuple[int, int, int]:
        """Parse pytest-style summary counts from test output.

        Falls back to exit-code-based counting when no summary is found.
        """
        passed_match = re.search(r"(\d+) passed", stdout)
        failed_match = re.search(r"(\d+) failed", stdout)
        if passed_match or failed_match:
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            return passed, failed, passed + failed
        # Fallback: no parseable summary.
        if stdout.strip() == "" and stderr.strip() == "":
            return 0, 0, 0
        # Non-empty output with an error/traceback => treat as a single failure;
        # otherwise a single pass.
        if "traceback" in (stdout + stderr).lower() or "error" in (stdout + stderr).lower():
            return 0, 1, 1
        return (1, 0, 1)
