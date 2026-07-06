"""The SWT loop engine: generator -> critic -> analyzer -> cognitive model.

One :func:`run_loop` call drives a full round-based loop and yields events as an
async generator, so the route can stream them straight to the browser as SSE.
Each round:

    1. GENERATOR answers, using any condensed context plus the analyzer's
       accumulated prompt hints from prior rounds.
    2. CRITIC judges the answer and returns a structured accept/revise verdict.
    3. ANALYZER diagnoses why the critic objected and prescribes the next
       round's adjustment (the ASI-EVOLVE keystone).
    4. COGNITIVE MODEL predicts whether the user will accept the answer.

The loop converges when the critic accepts AND the cognitive model's
satisfaction prediction clears the threshold, or when rounds run out (in which
case the best answer so far is returned).
"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator, Dict, List, Optional

from src.swt.analyzer import analyze
from src.swt.cognitive_model import CognitiveModel
from src.swt.models import ModelAdapter
from src.swt.recursive import condense_context
from src.swt.schemas import (
    Critique,
    Diagnosis,
    DiagnosisCategory,
    LoopConfig,
    LoopResult,
    RoundResult,
    new_id,
)
from src.swt.store import SwtStore

logger = logging.getLogger(__name__)

_CRITIC_SYSTEM = (
    "You are a rigorous CRITIC. You are given a request and a candidate answer. "
    "Find real problems: factual errors, missing pieces, unsupported claims, and "
    "failures to follow the request. Be specific and do not invent problems that "
    "are not there. Respond with ONLY a JSON object with keys: accepted (boolean "
    "-- true only if the answer is genuinely good enough to ship), summary (one "
    "sentence), issues (array of specific problems, empty if accepted), "
    "missing_context (array naming information the answer needed but lacked, "
    "empty if none)."
)


def _extract_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
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


def _parse_critique(raw: str) -> Critique:
    data = _extract_json(raw)
    if not data:
        # Unparseable critic output: treat as "needs work" so the loop keeps
        # improving rather than shipping on a malformed accept.
        return Critique(
            accepted=False,
            summary="Critic response could not be parsed as a verdict.",
            issues=["Critic did not return a structured verdict."],
            raw=raw,
        )
    return Critique(
        accepted=bool(data.get("accepted", False)),
        summary=str(data.get("summary", "")).strip() or "(no summary)",
        issues=[str(x) for x in (data.get("issues") or [])],
        missing_context=[str(x) for x in (data.get("missing_context") or [])],
        raw=raw,
    )


def _generator_messages(
    cfg: LoopConfig, condensed_context: str, prompt_hints: List[str]
) -> List[Dict[str, str]]:
    system = "You are a careful assistant. Answer the user's request completely and accurately."
    if prompt_hints:
        # Analyzer prescriptions accumulate across rounds so fixes stick.
        system += "\n\nApply these lessons from previous attempts:\n- " + "\n- ".join(prompt_hints)
    user = cfg.prompt
    if condensed_context:
        user = f"REFERENCE MATERIAL:\n{condensed_context}\n\n---\n\nREQUEST:\n{cfg.prompt}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def run_loop(
    cfg: LoopConfig,
    adapter: ModelAdapter,
    store: SwtStore,
    cognitive: CognitiveModel,
) -> AsyncIterator[dict]:
    """Run the loop, yielding SSE-ready event dicts and persisting the result."""
    loop_id = new_id("swt")
    result = LoopResult(loop_id=loop_id, prompt=cfg.prompt)
    yield {"type": "loop_start", "loop_id": loop_id, "config": {
        "generator_model": cfg.generator_model,
        "critic_model": cfg.critic_model,
        "analyzer_model": cfg.resolved_analyzer(),
        "max_rounds": cfg.max_rounds,
    }}

    # Condense long context once up front (RLM-style), reused every round.
    condensed = cfg.context or ""
    if cfg.context and cfg.use_recursive_context:
        yield {"type": "context_condensing", "original_chars": len(cfg.context)}
        try:
            condensed = await condense_context(
                adapter, cfg.prompt, cfg.context, cfg.resolved_analyzer(), owner=cfg.owner
            )
        except Exception as exc:
            logger.warning("SWT context condense failed: %s", exc)
            condensed = cfg.context
        yield {"type": "context_condensed", "condensed_chars": len(condensed)}

    had_context = bool(condensed.strip())
    prompt_hints: List[str] = []
    active_generator = cfg.generator_model
    best_round: Optional[RoundResult] = None

    for i in range(cfg.max_rounds):
        yield {"type": "round_start", "round": i, "generator_model": active_generator}

        # 1. GENERATE
        try:
            answer = await adapter.complete(
                active_generator,
                _generator_messages(cfg, condensed, prompt_hints),
                temperature=cfg.temperature,
                max_tokens=1500,
                owner=cfg.owner,
            )
        except Exception as exc:
            yield {"type": "error", "stage": "generator", "message": str(exc)[:400]}
            break
        yield {"type": "generation", "round": i, "answer": answer}

        # 2. CRITIQUE
        try:
            critic_raw = await adapter.complete(
                cfg.critic_model,
                [
                    {"role": "system", "content": _CRITIC_SYSTEM},
                    {"role": "user", "content": f"REQUEST:\n{cfg.prompt}\n\nCANDIDATE ANSWER:\n{answer}"},
                ],
                temperature=0.3,
                max_tokens=600,
                owner=cfg.owner,
            )
            critique = _parse_critique(critic_raw)
        except Exception as exc:
            yield {"type": "error", "stage": "critic", "message": str(exc)[:400]}
            critique = Critique(accepted=True, summary="Critic unavailable; accepting answer.", raw="")
        yield {"type": "critique", "round": i, "critique": critique.to_dict()}

        # 3. ANALYZE
        diagnosis: Diagnosis = await analyze(
            adapter,
            cfg.resolved_analyzer(),
            cfg.prompt,
            answer,
            critique,
            had_context=had_context,
            owner=cfg.owner,
        )
        yield {"type": "diagnosis", "round": i, "diagnosis": diagnosis.to_dict()}

        # 4. COGNITIVE MODEL
        satisfaction = None
        if cfg.use_cognitive_model:
            satisfaction = cognitive.predict(cfg.owner, cfg.prompt, answer)
            yield {"type": "satisfaction", "round": i, "satisfaction": satisfaction.to_dict()}

        round_result = RoundResult(
            round_index=i,
            answer=answer,
            critique=critique,
            diagnosis=diagnosis,
            satisfaction=satisfaction,
            generator_model=active_generator,
            critic_model=cfg.critic_model,
        )
        result.rounds.append(round_result)
        if best_round is None or (critique.accepted and not best_round.critique.accepted):
            best_round = round_result

        # Convergence: critic happy AND (no cognitive model, or it agrees).
        sat_ok = satisfaction is None or satisfaction.score >= cfg.satisfaction_threshold
        if critique.accepted and sat_ok:
            result.converged = True
            result.converged_reason = (
                "Critic accepted"
                + ("" if satisfaction is None else f" and predicted satisfaction {satisfaction.score:.2f}")
            )
            best_round = round_result
            break
        if critique.accepted and not sat_ok:
            # Critic likes it but the user historically would not: keep going and
            # tell the generator so via the analyzer channel.
            prompt_hints.append(
                "The last answer was technically correct but did not match the "
                "user's usual preferences; make it more aligned with what they accept."
            )

        # Apply the analyzer's prescription to the next round.
        if diagnosis.next_prompt_hint:
            prompt_hints.append(diagnosis.next_prompt_hint)
        if (
            diagnosis.category == DiagnosisCategory.ORCHESTRATION_FAILURE
            and diagnosis.switch_generator_to
            and diagnosis.switch_generator_to != active_generator
        ):
            yield {
                "type": "model_switch",
                "round": i,
                "from": active_generator,
                "to": diagnosis.switch_generator_to,
            }
            active_generator = diagnosis.switch_generator_to

    # Finalize.
    if best_round is not None:
        result.final_answer = best_round.answer
    if not result.converged and not result.converged_reason:
        result.converged_reason = "Reached max rounds without full convergence; returning best answer."

    try:
        store.save_loop(result, cfg.owner)
    except Exception as exc:
        logger.warning("SWT store.save_loop failed: %s", exc)

    yield {"type": "loop_complete", "result": result.to_dict()}
