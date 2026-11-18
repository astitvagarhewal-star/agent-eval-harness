"""Task & verification spec for the Sandboxed Coding-Agent Eval Harness.

Defines the canonical `TaskSpec` Pydantic model used throughout the harness,
plus helpers for loading task directories from ``benchmarks/tasks/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskSpec(BaseModel):
    """A single self-contained, deterministic coding benchmark task.

    Each task lives in ``benchmarks/tasks/<task_id>/`` and ships a small,
    dependency-free source repository template, a setup command, a test
    command, hidden evaluation tests, and a verified golden patch.
    """

    id: str = Field(..., description="Unique task identifier (slug).")
    repo_url: str | None = Field(
        None,
        description="Remote repo URL to clone (optional if a local template ships).",
    )
    base_commit: str | None = Field(
        None,
        description="Git commit (or branch/tag) to check out as the failing baseline.",
    )
    prompt: str = Field(..., description="Natural-language task description given to the agent.")
    setup_cmd: str | None = Field(
        None,
        description="Command run once after checkout to install deps / build.",
    )
    test_cmd: str | None = Field(
        None,
        description="Command run by the grader to execute the evaluation tests.",
    )
    hidden_tests: list[str] | None = Field(
        None,
        description="Paths to test files restored into the sandbox after the agent finishes.",
    )
    timeout_sec: int = Field(120, ge=1, le=7200, description="Per-run timeout in seconds.")
    difficulty: str = Field(
        "medium",
        description="Difficulty rating: easy / medium / hard.",
    )
    language: str = Field("python", description="Primary language: python / typescript / go.")

    @field_validator("difficulty")
    @classmethod
    def _validate_difficulty(cls, v: str) -> str:
        allowed = {"easy", "medium", "hard"}
        if v not in allowed:
            raise ValueError(f"difficulty must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("language")
    @classmethod
    def _validate_language(cls, v: str) -> str:
        allowed = {"python", "typescript", "go"}
        if v not in allowed:
            raise ValueError(f"language must be one of {sorted(allowed)}, got {v!r}")
        return v

    @property
    def task_dir(self) -> Path:
        """Return the on-disk task directory for this task id."""
        return Path(__file__).resolve().parent / "tasks" / self.id

    def model_dump_for_patch(self) -> dict[str, Any]:
        """Serialize fields relevant to reconstructing a task sandbox."""
        return {
            "id": self.id,
            "repo_url": self.repo_url,
            "base_commit": self.base_commit,
            "prompt": self.prompt,
            "setup_cmd": self.setup_cmd,
            "test_cmd": self.test_cmd,
            "hidden_tests": self.hidden_tests,
            "timeout_sec": self.timeout_sec,
            "difficulty": self.difficulty,
            "language": self.language,
        }

    class Config:
        use_enum_values = True
        validate_assignment = True


__all__ = ["TaskSpec"]
