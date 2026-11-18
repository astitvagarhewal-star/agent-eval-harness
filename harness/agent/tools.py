"""The agent's five core sandbox tools plus a dispatch registry.

Each tool receives a sandbox context (manager + sandbox id) and returns a
:class:`ToolResult`.  All mutating operations go through the sandbox so they
are isolated and auditable.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from harness.sandbox import SandboxError, SandboxManager


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)


class ToolRegistry:
    """Registry of the five core tools, bound to a single sandbox."""

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        sandbox_id: str,
        *,
        output_truncate_chars: int = 4000,
    ) -> None:
        self.mgr = sandbox_manager
        self.sandbox_id = sandbox_id
        self.output_truncate_chars = output_truncate_chars
        self._calls: list[tuple[str, dict, ToolResult]] = []

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #
    def execute(self, name: str, arguments: dict) -> ToolResult:
        handler = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_repo": self.search_repo,
            "apply_patch": self.apply_patch,
            "run_command": self.run_command,
        }.get(name)
        if handler is None:
            return ToolResult(ok=False, error=f"unknown tool: {name}")
        result = handler(**arguments)
        self._calls.append((name, arguments, result))
        return result

    def call_history(self) -> list[tuple[str, dict, ToolResult]]:
        return list(self._calls)

    # ------------------------------------------------------------------ #
    # Tool 1: list_files
    # ------------------------------------------------------------------ #
    def list_files(self, path: str = ".", max_depth: int = 3) -> ToolResult:
        try:
            handle = self.mgr._get(self.sandbox_id)
        except SandboxError as exc:
            return ToolResult(ok=False, error=str(exc))
        root = handle.worktree_path or handle.root
        base = root / path
        if not base.exists():
            return ToolResult(ok=False, error=f"path not found: {path}")
        lines: list[str] = []
        try:
            for entry in sorted(base.rglob("*")):
                rel = entry.relative_to(root)
                depth = len(rel.parts) - 1
                if depth > max_depth:
                    continue
                marker = "/" if entry.is_dir() else ""
                lines.append(f"{'  ' * depth}{rel}{marker}")
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(ok=False, error=f"list_files error: {exc}")
        output = "\n".join(lines) or "(empty)"
        return ToolResult(ok=True, output=self._truncate(output))

    # ------------------------------------------------------------------ #
    # Tool 2: read_file
    # ------------------------------------------------------------------ #
    def read_file(
        self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None
    ) -> ToolResult:
        try:
            handle = self.mgr._get(self.sandbox_id)
        except SandboxError as exc:
            return ToolResult(ok=False, error=str(exc))
        root = handle.worktree_path or handle.root
        target = root / path
        if not target.is_file():
            return ToolResult(ok=False, error=f"file not found: {path}")
        try:
            data = target.read_bytes().decode("utf-8", "replace")
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(ok=False, error=f"read_file error: {exc}")
        lines = data.splitlines()
        total = len(lines)
        start = max((start_line or 1) - 1, 0)
        end = total if end_line is None else min(end_line, total)
        if start >= end:
            return ToolResult(ok=False, error="empty line range")
        selected = lines[start:end]
        numbered = [f"{i + 1:5d}\t{line}" for i, line in enumerate(selected, start=start)]
        output = "\n".join(numbered)
        output += f"\n... (lines {start + 1}-{end} of {total})"
        return ToolResult(ok=True, output=self._truncate(output))

    # ------------------------------------------------------------------ #
    # Tool 3: search_repo
    # ------------------------------------------------------------------ #
    def search_repo(self, pattern: str, glob: str = "*") -> ToolResult:
        try:
            handle = self.mgr._get(self.sandbox_id)
        except SandboxError as exc:
            return ToolResult(ok=False, error=str(exc))
        root = handle.worktree_path or handle.root
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(ok=False, error=f"invalid regex: {exc}")
        matches: list[str] = []
        count = 0
        for file in sorted(root.rglob(glob)):
            if not file.is_file() or file.is_symlink():
                continue
            # Skip git internals and large binary-ish files.
            if ".git" in file.parts:
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            rel = file.relative_to(root)
            for i, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{rel}:{i}:{line}")
                    count += 1
                    if count >= 100:
                        break
            if count >= 100:
                break
        output = "\n".join(matches) or "(no matches)"
        return ToolResult(ok=True, output=self._truncate(output))

    # ------------------------------------------------------------------ #
    # Tool 4: apply_patch
    # ------------------------------------------------------------------ #
    def apply_patch(self, unified_diff: str) -> ToolResult:
        if not unified_diff or not unified_diff.strip():
            return ToolResult(
                ok=False,
                error="patch_syntax_error: empty diff",
            )
        try:
            self.mgr.apply_patch(self.sandbox_id, unified_diff)
        except SandboxError as exc:
            return ToolResult(ok=False, error=f"patch_syntax_error: {exc}")
        return ToolResult(ok=True, output="patch applied successfully")

    # ------------------------------------------------------------------ #
    # Tool 5: run_command
    # ------------------------------------------------------------------ #
    def run_command(
        self,
        cmd: str,
        *,
        timeout_sec: Optional[int] = None,
        cwd: Optional[str] = None,
    ) -> ToolResult:
        try:
            res = self.mgr.exec(
                self.sandbox_id,
                cmd,
                timeout_sec=timeout_sec,
                cwd=cwd,
            )
        except SandboxError as exc:
            return ToolResult(ok=False, error=f"run_command error: {exc}")
        output = res.stdout
        if res.stderr:
            output += f"\n[stderr]\n{res.stderr}"
        if res.timed_out:
            output += f"\n[command timed out after {timeout_sec}s]"
        status = "ok" if res.ok else f"exit_code={res.exit_code}"
        output = f"[{status}]\n{output}"
        return ToolResult(ok=res.ok, output=self._truncate(output))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _truncate(self, text: str) -> str:
        if len(text) <= self.output_truncate_chars:
            return text
        cut = self.output_truncate_chars
        return (
            text[:cut]
            + f"\n... [output truncated: {len(text) - cut} more chars] ..."
        )


__all__ = ["ToolResult", "ToolCall", "ToolRegistry"]
