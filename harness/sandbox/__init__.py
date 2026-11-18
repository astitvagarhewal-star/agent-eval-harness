"""Sandboxed execution backends for the eval harness.

Exposes a unified :class:`SandboxManager` interface backed by Docker when the
daemon is available, falling back automatically to isolated ``git worktree``
+ ``venv`` directories otherwise.
"""

from .manager import SandboxManager, SandboxBackend, SandboxResult, SandboxError

__all__ = ["SandboxManager", "SandboxBackend", "SandboxResult", "SandboxError"]
