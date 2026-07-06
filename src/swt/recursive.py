"""Recursive context handling for the SWT loop (RLM-inspired).

Long reference material causes "context rot" -- an LLM's accuracy drops as the
prompt grows. Recursive Language Models (Zhang & Khattab, MIT 2025) avoid this
by keeping the context out of the prompt and letting the model recursively
explore and partition it.

Odysseus runs against local models over its own endpoints, so rather than hand
control to the external ``rlm`` package's own provider config, we implement the
same *idea* natively: split oversized context into chunks, summarize each chunk
toward the query, then recursively fold the summaries until they fit a budget.
Every call flows through the same :class:`ModelAdapter` the rest of the loop
uses, so it runs on the user's local hardware.

If the ``rlm`` package is installed it is reported as available (for the UI to
surface), but the native path is what the loop uses so behavior is deterministic
and offline-testable.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from src.swt.models import ModelAdapter

logger = logging.getLogger(__name__)

# Roughly 4 chars/token; keep chunks well under a small local model's window.
_DEFAULT_CHUNK_CHARS = 6000
_DEFAULT_BUDGET_CHARS = 6000


def rlm_available() -> bool:
    """True when a Recursive Language Model package is importable."""
    try:
        import rlm  # noqa: F401  (either the `rlm` or `recursive-llm` distribution)

        return True
    except Exception:
        return False


def _split(text: str, chunk_chars: int) -> List[str]:
    """Split on paragraph boundaries where possible, else hard-slice."""
    if len(text) <= chunk_chars:
        return [text]
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for para in text.split("\n\n"):
        piece = para + "\n\n"
        if size + len(piece) > chunk_chars and buf:
            chunks.append("".join(buf))
            buf, size = [], 0
        if len(piece) > chunk_chars:
            # A single huge paragraph: hard-slice it.
            for i in range(0, len(piece), chunk_chars):
                chunks.append(piece[i : i + chunk_chars])
            continue
        buf.append(piece)
        size += len(piece)
    if buf:
        chunks.append("".join(buf))
    return chunks


async def condense_context(
    adapter: ModelAdapter,
    query: str,
    context: str,
    model: str,
    *,
    owner: Optional[str] = None,
    budget_chars: int = _DEFAULT_BUDGET_CHARS,
    chunk_chars: int = _DEFAULT_CHUNK_CHARS,
    max_depth: int = 3,
) -> str:
    """Recursively distill ``context`` down to what matters for ``query``.

    Returns context unchanged when it already fits the budget, so short prompts
    pay nothing. Guards against runaway recursion with ``max_depth`` and a
    no-progress check.
    """
    context = (context or "").strip()
    if not context or len(context) <= budget_chars:
        return context

    depth = 0
    current = context
    while len(current) > budget_chars and depth < max_depth:
        chunks = _split(current, chunk_chars)
        if len(chunks) <= 1:
            break
        summaries: List[str] = []
        for idx, chunk in enumerate(chunks):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are extracting only the parts of a document relevant "
                        "to a question. Return the relevant facts verbatim or "
                        "tightly summarized. If nothing in this section is "
                        "relevant, reply with exactly 'NOTHING RELEVANT'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"QUESTION:\n{query}\n\nSECTION {idx + 1}/{len(chunks)}:\n{chunk}",
                },
            ]
            try:
                out = await adapter.complete(
                    model, messages, temperature=0.1, max_tokens=512, owner=owner
                )
            except Exception as exc:  # keep the loop alive on a bad chunk
                logger.warning("SWT recursive summarize failed on chunk %s: %s", idx, exc)
                out = ""
            out = (out or "").strip()
            if out and "NOTHING RELEVANT" not in out.upper():
                summaries.append(out)
        folded = "\n\n".join(summaries).strip()
        if not folded or len(folded) >= len(current):
            # No progress -- stop rather than spin.
            break
        current = folded
        depth += 1
    return current[:budget_chars]
