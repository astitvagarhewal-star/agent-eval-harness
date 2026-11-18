"""Unified async OpenAI-compatible / OpenCode LLM client.

Features:
* Retries with exponential backoff on 429/5xx and transport errors.
* Per-model rate-limit pacing (requests-per-minute from ``configs/models.yaml``).
* Per-model token & estimated USD cost accounting.
* Optional tool-calling (OpenAI-style ``tools`` / ``tool_choice``).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelSpec(BaseModel):
    id: str
    label: str
    role: str
    endpoint: str
    api_key_env: str
    pricing: dict[str, float] = Field(default_factory=dict)
    rate_limit: dict[str, int] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str = ""
    model: str = ""
    latency_sec: float = 0.0


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class ModelRegistry:
    """Loads and looks up model specs from ``configs/models.yaml``."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self._models: dict[str, ModelSpec] = {}
        self._load()

    def _load(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for entry in data.get("models", []):
            spec = ModelSpec(**entry)
            self._models[spec.id] = spec

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._models:
            raise KeyError(f"unknown model id: {model_id!r}")
        return self._models[model_id]

    def register(self, spec: ModelSpec) -> None:
        """Register (or override) a model spec, useful for tests/mocks."""
        self._models[spec.id] = spec

    def ids(self) -> list[str]:
        return list(self._models.keys())


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class ModelClient:
    """Async chat-completions client with retries, rate pacing, and cost acct."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        max_retries: int = 5,
        base_backoff_sec: float = 1.0,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self._owns_client = http_client is None
        self.http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec

        # Per-model rate pacing: last call timestamp per model id.
        self._last_call: dict[str, float] = {}
        # Usage & cost accrual per model id.
        self.usage: dict[str, Usage] = {}
        self.cost_usd: dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------- #
    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()

    async def __aenter__(self) -> "ModelClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # -- public API --------------------------------------------------------- #
    async def complete(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str | bool = "auto",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        spec = self.registry.get(model_id)
        await self._pace(spec)
        payload = self._build_payload(
            spec, messages, tools, tool_choice, temperature, max_tokens
        )
        headers = self._headers(spec)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            start = time.monotonic()
            try:
                resp = await self.http.post(
                    f"{spec.endpoint.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                await self._sleep(attempt)
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = _HttpError(resp.status_code, resp.text[:500])
                # Honor Retry-After when present.
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        await self._sleep_for(float(retry_after))
                    except ValueError:
                        await self._sleep(attempt)
                else:
                    await self._sleep(attempt)
                continue

            if resp.status_code >= 400:
                raise _HttpError(resp.status_code, resp.text[:1000])

            data = resp.json()
            latency = time.monotonic() - start
            return self._parse_response(spec, data, latency)

        # Exhausted retries.
        raise last_error or _HttpError(500, "request exhausted retries")

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        spec = self.registry.get(model_id)
        p = spec.pricing
        return (
            input_tokens / 1_000_000 * p.get("input_per_1m", 0.0)
            + output_tokens / 1_000_000 * p.get("output_per_1m", 0.0)
        )

    # -- internals ---------------------------------------------------------- #
    async def _pace(self, spec: ModelSpec) -> None:
        rpm = spec.rate_limit.get("rpm") if spec.rate_limit else None
        if not rpm:
            return
        min_interval = 60.0 / rpm
        now = time.monotonic()
        last = self._last_call.get(spec.id, 0.0)
        elapsed = now - last
        if elapsed < min_interval:
            await self._sleep_for(min_interval - elapsed)
        self._last_call[spec.id] = time.monotonic()

    @staticmethod
    async def _sleep(attempt: int) -> None:
        # Exponential backoff with jitter-free deterministic steps.
        delay = 2 ** attempt
        await asyncio.sleep(delay)

    @staticmethod
    async def _sleep_for(sec: float) -> None:
        if sec > 0:
            await asyncio.sleep(sec)

    def _headers(self, spec: ModelSpec) -> dict[str, str]:
        import os

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(spec.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_payload(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        tool_choice: str | bool,
        temperature: float,
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": spec.id,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        return payload

    def _parse_response(
        self, spec: ModelSpec, data: dict[str, Any], latency: float
    ) -> LLMResponse:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        # Normalize tool_calls into {name, arguments} dicts.
        normalized: list[dict[str, Any]] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            normalized.append(
                {"name": fn.get("name", ""), "arguments": fn.get("arguments", "{}")}
            )
        usage_data = data.get("usage", {}) or {}
        input_tokens = int(usage_data.get("prompt_tokens", 0))
        output_tokens = int(usage_data.get("completion_tokens", 0))
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

        # Accumulate per-model usage + cost.
        self.usage[spec.id] = Usage(
            input_tokens=self.usage.get(spec.id, Usage()).input_tokens + input_tokens,
            output_tokens=self.usage.get(spec.id, Usage()).output_tokens + output_tokens,
        )
        self.cost_usd[spec.id] = (
            self.cost_usd.get(spec.id, 0.0)
            + self.estimate_cost(spec.id, input_tokens, output_tokens)
        )

        return LLMResponse(
            content=content,
            tool_calls=normalized,
            usage=usage,
            finish_reason=choice.get("finish_reason", ""),
            model=spec.id,
            latency_sec=latency,
        )


class _HttpError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


__all__ = ["ModelClient", "ModelRegistry", "ModelSpec", "LLMResponse", "Usage"]
