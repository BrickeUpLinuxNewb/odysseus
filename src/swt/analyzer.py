"""The analyzer -- the stage that closes the SWT loop.

Following the ASI-EVOLVE loop (learn -> design -> experiment -> analyze), the
analyzer is what lets the loop improve without a human: it takes the critic's
finding, classifies the root cause, and emits a typed
:class:`~src.swt.schemas.Prescription` the engine dispatches on. Each category
has exactly one cure:

    1. Was the needed information available but not retrieved/used?
       -> retrieval_failure  -> RetrievalAdjustment: re-run retrieval next
          round, wider and/or with a rewritten query
    2. Was the information never captured at all?
       -> knowledge_gap      -> write_to_kb: record the gap in the knowledge
          base so future runs retrieve it instead of re-hitting the wall
    3. Was the prompt or model wrong for this class of task?
       -> orchestration_failure -> prompt hint and/or switch_generator_to

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
from src.swt.schemas import (
    Critique,
    Diagnosis,
    DiagnosisCategory,
    Prescription,
    RetrievalAdjustment,
)

logger = logging.getLogger(__name__)

# How much wider each retrieval_failure round searches (added to top-k).
WIDEN_K_STEP = 2

_SYSTEM = (
    "You are the ANALYZER in a self-improving answer loop. A generator produced "
    "an answer and a critic found problems with it. Your job is to diagnose WHY "
    "the answer fell short and prescribe one concrete adjustment for the next "
    "round. Classify the root cause into exactly one category:\n"
    "  retrieval_failure: the needed information was available in the provided "
    "context/reference material/knowledge base but the generator did not use it.\n"
    "  knowledge_gap: the needed information was never provided at all.\n"
    "  orchestration_failure: the answer was wrong because the instructions, "
    "framing, or chosen model were a poor fit for this task.\n"
    "Respond with ONLY a JSON object, no prose, with keys: category (one of the "
    "three strings above), rationale (one sentence), next_prompt_hint (an "
    "instruction to append to the generator's system prompt, or empty), "
    "retrieval_query_rewrite (retrieval_failure only: a better search query for "
    "the knowledge base, or empty to keep the current one), "
    "knowledge_note (knowledge_gap only: one factual sentence naming exactly "
    "what information is missing, suitable for storing in a knowledge base), "
    "switch_generator_to (orchestration_failure only: a different model name if "
    "clearly warranted, else empty)."
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


def _gap_note(critique: Critique, prompt: str) -> str:
    """Default knowledge-base note when the analyzer LLM doesn't supply one."""
    missing = "; ".join(critique.missing_context) or critique.summary
    return f"Known gap: {missing} (surfaced answering: {prompt[:200]})"


def _prescription(
    category: DiagnosisCategory,
    *,
    prompt_hint: str = "",
    query_rewrite: str = "",
    knowledge_note: str = "",
    switch_generator_to: str = "",
    retrieval_note: str = "",
) -> Prescription:
    """Build the one lever the category calls for (plus any prompt hint)."""
    p = Prescription(prompt_hint=prompt_hint)
    if category == DiagnosisCategory.RETRIEVAL_FAILURE:
        p.retrieval = RetrievalAdjustment(
            widen_k=WIDEN_K_STEP, query_rewrite=query_rewrite, note=retrieval_note
        )
    elif category == DiagnosisCategory.KNOWLEDGE_GAP:
        p.write_to_kb = knowledge_note
    elif category == DiagnosisCategory.ORCHESTRATION_FAILURE:
        p.switch_generator_to = switch_generator_to
    return p


def _heuristic(critique: Critique, had_context: bool, prompt: str = "") -> Diagnosis:
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
                prescription=_prescription(
                    DiagnosisCategory.RETRIEVAL_FAILURE,
                    query_rewrite=" ".join(critique.missing_context)[:300],
                    retrieval_note="Broaden retrieval and re-scan the available context.",
                ),
                raw="heuristic",
            )
        return Diagnosis(
            category=DiagnosisCategory.KNOWLEDGE_GAP,
            rationale="Critic flagged information that was never provided to the loop.",
            prescription=_prescription(
                DiagnosisCategory.KNOWLEDGE_GAP,
                prompt_hint="State assumptions explicitly and flag what you cannot verify.",
                knowledge_note=_gap_note(critique, prompt),
            ),
            raw="heuristic",
        )
    if any(k in blob for k in ("format", "instruction", "off-topic", "tone", "wrong", "did not follow")):
        return Diagnosis(
            category=DiagnosisCategory.ORCHESTRATION_FAILURE,
            rationale="Critic flagged an instruction/format mismatch rather than a fact gap.",
            prescription=_prescription(
                DiagnosisCategory.ORCHESTRATION_FAILURE,
                prompt_hint="Follow the requested format and address the question directly.",
            ),
            raw="heuristic",
        )
    return Diagnosis(
        category=DiagnosisCategory.ORCHESTRATION_FAILURE,
        rationale="Unclassified critique; treat as a prompt-fit issue by default.",
        prescription=_prescription(
            DiagnosisCategory.ORCHESTRATION_FAILURE,
            prompt_hint="Re-read the request and correct the specific issues the critic raised.",
        ),
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
    """Diagnose one critique into a category plus a typed prescription."""
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
        return _heuristic(critique, had_context, prompt)

    data = _extract_json(raw)
    if not data:
        return _heuristic(critique, had_context, prompt)

    try:
        category = DiagnosisCategory(str(data.get("category", "")).strip().lower())
    except ValueError:
        category = _heuristic(critique, had_context, prompt).category

    knowledge_note = str(data.get("knowledge_note", "")).strip()
    if category == DiagnosisCategory.KNOWLEDGE_GAP and not knowledge_note:
        knowledge_note = _gap_note(critique, prompt)

    return Diagnosis(
        category=category,
        rationale=str(data.get("rationale", "")).strip() or "(no rationale)",
        prescription=_prescription(
            category,
            prompt_hint=str(data.get("next_prompt_hint", "")).strip(),
            query_rewrite=str(data.get("retrieval_query_rewrite", "")).strip(),
            knowledge_note=knowledge_note,
            switch_generator_to=str(data.get("switch_generator_to", "")).strip(),
        ),
        raw=raw,
    )
