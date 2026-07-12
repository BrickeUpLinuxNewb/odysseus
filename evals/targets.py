"""Targets: the things the harness measures.

A ``Target`` turns (system, prompt) into text plus whatever side metrics the
backend can report. Four real ones:

    selftest    answers every task from its own rubric. Scores 1.0 on a
                well-formed task file -- so it is the task-file linter.
                Anything below 1.0 here is a broken rubric, not a model.
    model:SPEC  one-shot through Odysseus's own resolver (``model@endpoint``
                works, so this measures whatever the app would actually use).
    http:URL::MODEL
                raw OpenAI-compatible endpoint (llama.cpp server, vLLM,
                Ollama). This is the Track A path: point it at two llama.cpp
                servers running different GGUF quantizations of the same
                model and compare runs. Reports true token counts when the
                server returns ``usage``.
    swt:GEN::CRIT[::ANALYZER]
                the full SWT loop as a black box; extra metrics carry rounds
                and convergence so the B4 question (does looping help, or
                does the critic just get tired?) becomes a measured delta:
                compare ``model:GEN`` vs ``swt:GEN::CRIT`` on the same tasks.

All app-stack imports are lazy so this module (and compare) load anywhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from evals.checkers import derive_passing_output
from evals.tasks import Task

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.2   # eval default: low-but-nonzero, like real routing use
DEFAULT_MAX_TOKENS = 512


@dataclass
class TargetResult:
    text: str
    extra: Dict[str, Any] = field(default_factory=dict)  # numeric extras get aggregated


class Target(Protocol):
    name: str

    async def complete(self, task: Task) -> TargetResult: ...


class SelfTestTarget:
    """Answers from the rubric itself; exists to lint task files."""

    name = "selftest"

    async def complete(self, task: Task) -> TargetResult:
        answer = derive_passing_output(task.kind, task.expect)
        if answer is None:
            return TargetResult(
                text="",
                extra={"selftest_underivable": 1},
            )
        return TargetResult(text=answer)


class OdysseusModelTarget:
    """One-shot completion through Odysseus's resolver + HTTP client."""

    def __init__(self, model: str, *, temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS, owner: Optional[str] = None) -> None:
        self.name = f"model:{model}"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.owner = owner
        self._adapter = None

    async def complete(self, task: Task) -> TargetResult:
        if self._adapter is None:
            from src.swt.models import OdysseusModelAdapter

            self._adapter = OdysseusModelAdapter()
        messages = []
        if task.system:
            messages.append({"role": "system", "content": task.system})
        messages.append({"role": "user", "content": task.prompt})
        text = await self._adapter.complete(
            self.model, messages,
            temperature=self.temperature, max_tokens=self.max_tokens, owner=self.owner,
        )
        return TargetResult(text=text or "")


class OpenAICompatTarget:
    """Raw OpenAI-compatible chat endpoint -- the quantization-sweep path."""

    def __init__(self, base_url: str, model: str, *, api_key: Optional[str] = None,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS, timeout: float = 120.0) -> None:
        self.name = f"http:{base_url}::{model}"
        self.url = self._chat_url(base_url)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    @staticmethod
    def _chat_url(base: str) -> str:
        base = base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    async def complete(self, task: Task) -> TargetResult:
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        messages = []
        if task.system:
            messages.append({"role": "system", "content": task.system})
        messages.append({"role": "user", "content": task.prompt})
        resp = await self._client.post(self.url, headers=headers, json={
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        extra: Dict[str, Any] = {}
        usage = data.get("usage") or {}
        if isinstance(usage.get("completion_tokens"), int):
            extra["completion_tokens"] = usage["completion_tokens"]
        return TargetResult(text=text, extra=extra)


class SwtLoopTarget:
    """The whole generator/critic/analyzer loop as one measurable unit."""

    def __init__(self, generator: str, critic: str, analyzer: str = "", *,
                 max_rounds: int = 4, owner: Optional[str] = None) -> None:
        self.name = f"swt:{generator}::{critic}" + (f"::{analyzer}" if analyzer else "")
        self.generator = generator
        self.critic = critic
        self.analyzer = analyzer
        self.max_rounds = max_rounds
        self.owner = owner

    async def complete(self, task: Task) -> TargetResult:
        from src.swt import (
            LoopConfig, OdysseusModelAdapter, PreferenceModel, create_store, run_loop,
        )

        cfg = LoopConfig(
            prompt=(f"{task.system}\n\n{task.prompt}" if task.system else task.prompt),
            generator_model=self.generator,
            critic_model=self.critic,
            analyzer_model=self.analyzer,
            max_rounds=self.max_rounds,
            # Isolate the measurement: no preference model (needs feedback
            # history), no cross-run seeding (would let earlier eval rounds
            # contaminate later ones). This measures the loop mechanism.
            use_preference_model=False,
            use_experience=False,
            owner=self.owner,
        )
        store = create_store()
        adapter = OdysseusModelAdapter()
        final: Dict[str, Any] = {}
        async for event in run_loop(cfg, adapter, store, PreferenceModel(store)):
            if event.get("type") == "loop_complete" and event.get("result"):
                final = event["result"]
        rounds = len(final.get("rounds") or [])
        return TargetResult(
            text=final.get("final_answer", "") or "",
            extra={"rounds": rounds, "converged": 1 if final.get("converged") else 0},
        )


def make_target(spec: str, *, temperature: float = DEFAULT_TEMPERATURE,
                max_tokens: int = DEFAULT_MAX_TOKENS) -> Target:
    """Parse a CLI target spec. ``::`` separates fields because model specs
    legitimately contain ``:`` (ollama tags) and ``@`` (endpoint pinning)."""
    spec = spec.strip()
    if spec == "selftest":
        return SelfTestTarget()
    scheme, _, rest = spec.partition(":")
    if scheme == "model" and rest:
        return OdysseusModelTarget(rest, temperature=temperature, max_tokens=max_tokens)
    if scheme == "http" and rest:
        parts = rest.split("::")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("http target needs http:BASE_URL::MODEL_ID")
        return OpenAICompatTarget(parts[0], parts[1],
                                  temperature=temperature, max_tokens=max_tokens)
    if scheme == "swt" and rest:
        parts = rest.split("::")
        if len(parts) not in (2, 3) or not all(parts[:2]):
            raise ValueError("swt target needs swt:GENERATOR::CRITIC[::ANALYZER]")
        analyzer = parts[2] if len(parts) == 3 else ""
        return SwtLoopTarget(parts[0], parts[1], analyzer)
    raise ValueError(
        f"unknown target spec {spec!r} -- use selftest, model:SPEC, "
        "http:BASE_URL::MODEL_ID, or swt:GEN::CRIT[::ANALYZER]"
    )
