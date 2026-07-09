"""Model adapter for the SWT loop.

The engine never imports Odysseus's LLM stack directly. It talks to a small
:class:`ModelAdapter` protocol so the loop logic stays testable with a fake and
so all model calls flow through Odysseus's own resolver and HTTP client -- the
same endpoints, keys, and models the user already configured in the GUI.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class ModelAdapter(Protocol):
    """Minimal surface the engine needs from a model backend."""

    async def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        owner: Optional[str] = None,
    ) -> str:
        ...


class OdysseusModelAdapter:
    """Routes SWT calls through Odysseus's ``_resolve_model`` + ``llm_call_async``.

    ``model`` may be a bare model name (``"hermes3:8b"``) or the
    ``name@endpoint`` form the rest of Odysseus accepts. Because every SWT
    role (generator / critic / analyzer / librarian) passes its own spec
    through here, ``name@endpoint`` per role is how a loop spans multiple
    nodes -- e.g. generator on the GPU box, critic on a second machine --
    with no extra plumbing. Resolution is cached per (spec, owner) for the
    life of the adapter so a single loop does not re-probe endpoints on
    every round.
    """

    def __init__(self) -> None:
        self._resolve_cache: Dict[tuple, tuple] = {}

    def _resolve(self, model: str, owner: Optional[str]) -> tuple:
        key = (model, owner or "")
        if key not in self._resolve_cache:
            from src.ai_interaction import _resolve_model
            self._resolve_cache[key] = _resolve_model(model, owner=owner)
        return self._resolve_cache[key]

    async def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        owner: Optional[str] = None,
    ) -> str:
        from src.llm_core import llm_call_async

        url, model_id, headers = self._resolve(model, owner)
        return await llm_call_async(
            url,
            model_id,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
        )
