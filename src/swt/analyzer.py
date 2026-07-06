"""The analyzer -- the keystone of the SWT loop.

In the ASI-EVOLVE framing the analyzer is the bridge that closes the loop
without a human: it takes the critic's finding and asks three structured
questions, then writes a prescription the next round applies automatically.

    1. Was the missing context actually available but not retrieved?
       -> retrieval_failure  -> widen / re-run retrieval
    2. Was the context never captured at all?
       -> knowledge_gap      -> tell the generator to state assumptions / ask
    3. Was the prompt or model wrong for this class of task?
       -> orchestration_failure -> adjust the prompt template or swap the model

It is LLM-backed with a strict JSON contract, and degrades to a keyword
heuristic when the model is unavailable or returns unparseable output, so the
loop never stalls on the analyzer.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.swt.models import ModelAdapter
from src.swt.schemas import Critique, Diagnosis, DiagnosisCategory

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are the ANALYZER in a self-improving answer loop. A generator produced "
    "an answer and a critic found problems with it. Your job is to diagnose WHY "
    "the answer fell short and prescribe one concrete adjustment for the next "
    "round. Classify the root cause into exactly one category:\n"
    "  retrieval_failure: the needed information was available in the provided "
    "context/reference material but the generator did not use it.\n"
    "  knowledge_gap: the needed information was never provided at all.\n"
    "  orchestration_failure: the answer was wrong because the instructions, "
    "framing, or chosen model were a poor fit for this task.\n"
    "Respond with ONLY a JSON object, no prose, with keys: category (one of the "
    "three strings above), rationale (one sentence), next_prompt_hint (an "
    "instruction to append to the generator's system prompt, or empty), "
    "retrieval_adjustment (how to change retrieval, or empty), "
    "switch_generator_to (a different model name if orchestration_failure "
    "clearly calls for it, else empty)."
)


def _extract_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    # Direct parse, then first balanced {...} block.
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _heuristic(critique: Critique, had_context: bool) -> Diagnosis:
    """Fallback diagnosis from keywords when the LLM path is unavailable."""
    blob = " ".join(
        [critique.summary] + critique.issues + critique.missing_context
    ).lower()
    if critique.missing_context or any(
        k in blob for k in ("missing", "no source", "not provided", "unknown", "unclear where")
    ):
        if had_context:
            return Diagnosis(
                category=DiagnosisCategory.RETRIEVAL_FAILURE,
                rationale="Critic flagged missing information that context should have covered.",
                retrieval_adjustment="Broaden retrieval and re-scan the provided context.",
                raw="heuristic",
            )
        return Diagnosis(
            category=DiagnosisCategory.KNOWLEDGE_GAP,
            rationale="Critic flagged information that was never provided to the loop.",
            next_prompt_hint="State assumptions explicitly and flag what you cannot verify.",
            raw="heuristic",
        )
    if any(k in blob for k in ("format", "instruction", "off-topic", "tone", "wrong", "did not follow")):
        return Diagnosis(
            category=DiagnosisCategory.ORCHESTRATION_FAILURE,
            rationale="Critic flagged an instruction/format mismatch rather than a fact gap.",
            next_prompt_hint="Follow the requested format and address the question directly.",
            raw="heuristic",
        )
    return Diagnosis(
        category=DiagnosisCategory.ORCHESTRATION_FAILURE,
        rationale="Unclassified critique; treat as a prompt-fit issue by default.",
        next_prompt_hint="Re-read the request and correct the specific issues the critic raised.",
        raw="heuristic",
    )


async def analyze(
    adapter: ModelAdapter,
    model: str,
    prompt: str,
    answer: str,
    critique: Critique,
    *,
    had_context: bool,
    owner: Optional[str] = None,
) -> Diagnosis:
    """Diagnose one critique into a category plus a next-round prescription."""
    if critique.accepted:
        return Diagnosis(
            category=DiagnosisCategory.NONE,
            rationale="Critic accepted the answer; no adjustment required.",
        )

    user = (
        f"ORIGINAL REQUEST:\n{prompt}\n\n"
        f"GENERATOR ANSWER:\n{answer}\n\n"
        f"CRITIC SUMMARY:\n{critique.summary}\n\n"
        f"CRITIC ISSUES:\n- " + "\n- ".join(critique.issues or ["(none listed)"]) + "\n\n"
        f"REFERENCE CONTEXT WAS PROVIDED: {'yes' if had_context else 'no'}"
    )
    try:
        raw = await adapter.complete(
            model,
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=400,
            owner=owner,
        )
    except Exception as exc:
        logger.warning("SWT analyzer LLM call failed, using heuristic: %s", exc)
        return _heuristic(critique, had_context)

    data = _extract_json(raw)
    if not data:
        return _heuristic(critique, had_context)

    try:
        category = DiagnosisCategory(str(data.get("category", "")).strip().lower())
    except ValueError:
        category = _heuristic(critique, had_context).category

    return Diagnosis(
        category=category,
        rationale=str(data.get("rationale", "")).strip() or "(no rationale)",
        next_prompt_hint=str(data.get("next_prompt_hint", "")).strip(),
        retrieval_adjustment=str(data.get("retrieval_adjustment", "")).strip(),
        switch_generator_to=str(data.get("switch_generator_to", "")).strip(),
        raw=raw,
    )
