"""The SWT loop engine: generator -> critic -> analyzer -> preference model.

One :func:`run_loop` call drives a full round-based loop and yields events as an
async generator, so the route can stream them straight to the browser as SSE.

Before round 0 the engine reads the experience store for prior diagnoses on
similar prompts and seeds their prescriptions -- this is what makes the loop
ASI-EVOLVE-shaped: insight persists across runs instead of evaporating when a
loop ends. Then each round:

    1. RETRIEVE top-k knowledge-base chunks for the current retrieval query
       (skipped when retrieval is off or unavailable).
    2. GENERATOR answers, using retrieved chunks + condensed user context +
       the analyzer's accumulated prompt hints.
    3. CRITIC judges the answer and returns a structured accept/revise verdict.
    4. ANALYZER diagnoses why the critic objected and prescribes a typed
       adjustment, which the engine dispatches:
         retrieval_failure     -> widen / rewrite next round's retrieval
         knowledge_gap         -> record the gap in the knowledge base
         orchestration_failure -> swap the generator model
    5. PREFERENCE MODEL predicts whether the user will accept the answer.

The loop converges when the critic accepts AND the preference model's
satisfaction prediction clears the threshold, or when rounds run out (in which
case the best answer so far is returned). Every stage degrades gracefully; a
missing retriever, store, or model never stalls the loop.
"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator, Dict, List, Optional

from src.swt.analyzer import analyze
from src.swt.models import ModelAdapter
from src.swt.preference import PreferenceModel
from src.swt.recursive import condense_context
from src.swt.retrieval import KnowledgeWriter, MAX_RETRIEVAL_K, Retriever
from src.swt.schemas import (
    Critique,
    Diagnosis,
    DiagnosisCategory,
    LoopConfig,
    LoopResult,
    RoundResult,
    new_id,
)
from src.swt.store import ExperienceStore

logger = logging.getLogger(__name__)

# How many prior similar diagnoses seed a new run, and how many of their
# prompt hints are carried in (the freshest, most similar first).
_EXPERIENCE_K = 3
_MAX_SEED_HINTS = 3

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
    cfg: LoopConfig,
    condensed_context: str,
    retrieved: List[str],
    prompt_hints: List[str],
) -> List[Dict[str, str]]:
    system = "You are a careful assistant. Answer the user's request completely and accurately."
    if prompt_hints:
        # Analyzer prescriptions accumulate across rounds so fixes stick.
        system += "\n\nApply these lessons from previous attempts:\n- " + "\n- ".join(prompt_hints)
    sections: List[str] = []
    if retrieved:
        numbered = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(retrieved))
        sections.append(f"RETRIEVED KNOWLEDGE:\n{numbered}")
    if condensed_context:
        sections.append(f"REFERENCE MATERIAL:\n{condensed_context}")
    sections.append(f"REQUEST:\n{cfg.prompt}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n---\n\n".join(sections)},
    ]


class _RetrievalState:
    """Current retrieval query/k, mutated by retrieval_failure prescriptions."""

    def __init__(self, query: str, k: int) -> None:
        self.query = query
        self.k = max(1, min(k, MAX_RETRIEVAL_K))
        self.dirty = True  # retrieve on the first round

    def adjust(self, widen_k: int, query_rewrite: str) -> None:
        if widen_k:
            self.k = max(1, min(self.k + widen_k, MAX_RETRIEVAL_K))
        if query_rewrite.strip():
            self.query = query_rewrite.strip()
        self.dirty = True


def _seed_from_experience(
    store: ExperienceStore, cfg: LoopConfig, retrieval: _RetrievalState, hints: List[str]
) -> Dict:
    """Read prior diagnoses on similar prompts and pre-apply their lessons."""
    matches = store.similar_prior_diagnoses(cfg.prompt, cfg.owner, k=_EXPERIENCE_K)
    for m in matches:
        diagnosis: Diagnosis = m["diagnosis"]
        hint = diagnosis.prescription.prompt_hint
        if hint and hint not in hints and len(hints) < _MAX_SEED_HINTS:
            hints.append(hint)
        adj = diagnosis.prescription.retrieval
        if adj is not None:
            # A similar prompt already needed wider retrieval; start there.
            retrieval.adjust(adj.widen_k, adj.query_rewrite)
    return {
        "type": "experience_seeded",
        "matches": len(matches),
        "hints": list(hints),
        "retrieval_k": retrieval.k,
    }


async def run_loop(
    cfg: LoopConfig,
    adapter: ModelAdapter,
    store: ExperienceStore,
    preference: PreferenceModel,
    retriever: Optional[Retriever] = None,
    knowledge_writer: Optional[KnowledgeWriter] = None,
) -> AsyncIterator[dict]:
    """Run the loop, yielding SSE-ready event dicts and persisting the result."""
    loop_id = new_id("swt")
    result = LoopResult(loop_id=loop_id, prompt=cfg.prompt)
    if knowledge_writer is None and hasattr(retriever, "record_gap"):
        knowledge_writer = retriever  # RagRetriever serves both roles
    yield {"type": "loop_start", "loop_id": loop_id, "config": {
        "generator_model": cfg.generator_model,
        "critic_model": cfg.critic_model,
        "analyzer_model": cfg.resolved_analyzer(),
        "librarian_model": cfg.resolved_librarian(),
        "max_rounds": cfg.max_rounds,
    }}

    retrieval = _RetrievalState(cfg.prompt, cfg.retrieval_k)
    prompt_hints: List[str] = []

    # Learn across runs: seed this loop with what prior similar loops learned.
    if cfg.use_experience:
        try:
            yield _seed_from_experience(store, cfg, retrieval, prompt_hints)
        except Exception as exc:
            logger.warning("SWT experience seeding failed: %s", exc)

    # Condense long context once up front (RLM-style), reused every round.
    # This is librarian work: on a distributed deployment point
    # librarian_model at the node that should carry it.
    condensed = cfg.context or ""
    if cfg.context and cfg.use_recursive_context:
        yield {"type": "context_condensing", "original_chars": len(cfg.context)}
        try:
            condensed = await condense_context(
                adapter, cfg.prompt, cfg.context, cfg.resolved_librarian(), owner=cfg.owner
            )
        except Exception as exc:
            logger.warning("SWT context condense failed: %s", exc)
            condensed = cfg.context
        yield {"type": "context_condensed", "condensed_chars": len(condensed)}

    retrieved: List[str] = []
    active_generator = cfg.generator_model
    best_round: Optional[RoundResult] = None
    recorded_gaps: set = set()

    for i in range(cfg.max_rounds):
        yield {"type": "round_start", "round": i, "generator_model": active_generator}

        # 0. RETRIEVE (first round, then again whenever a prescription changed
        # the parameters -- this is the cure for retrieval_failure).
        if cfg.use_retrieval and retriever is not None and retrieval.dirty:
            retrieval.dirty = False
            try:
                retrieved = await retriever.retrieve(
                    retrieval.query, k=retrieval.k, owner=cfg.owner
                )
            except Exception as exc:
                logger.warning("SWT retrieval failed: %s", exc)
                retrieved = []
            yield {
                "type": "retrieval",
                "round": i,
                "k": retrieval.k,
                "chunks": len(retrieved),
            }

        had_context = bool(condensed.strip()) or bool(retrieved)

        # 1. GENERATE
        try:
            answer = await adapter.complete(
                active_generator,
                _generator_messages(cfg, condensed, retrieved, prompt_hints),
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

        # Persist the diagnosis as structured insight future runs query back.
        if diagnosis.category != DiagnosisCategory.NONE:
            try:
                store.save_diagnosis(loop_id, i, diagnosis, cfg.owner, prompt=cfg.prompt)
            except Exception as exc:
                logger.warning("SWT store.save_diagnosis failed: %s", exc)

        # 4. PREFERENCE MODEL
        satisfaction = None
        if cfg.use_preference_model:
            satisfaction = preference.predict(cfg.owner, cfg.prompt, answer)
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

        # Convergence: critic happy AND (no preference model, or it agrees).
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
            # Critic likes it but the user historically would not: keep going
            # and tell the generator so.
            prompt_hints.append(
                "The last answer was technically correct but did not match the "
                "user's usual preferences; make it more aligned with what they accept."
            )

        # Dispatch the analyzer's typed prescription onto the next round.
        prescription = diagnosis.prescription
        if prescription.prompt_hint:
            prompt_hints.append(prescription.prompt_hint)

        if diagnosis.category == DiagnosisCategory.RETRIEVAL_FAILURE and prescription.retrieval:
            retrieval.adjust(prescription.retrieval.widen_k, prescription.retrieval.query_rewrite)
            yield {
                "type": "retrieval_adjusted",
                "round": i,
                "k": retrieval.k,
                "query_rewritten": bool(prescription.retrieval.query_rewrite),
            }

        if (
            diagnosis.category == DiagnosisCategory.KNOWLEDGE_GAP
            and prescription.write_to_kb
            and knowledge_writer is not None
            and prescription.write_to_kb not in recorded_gaps
        ):
            recorded_gaps.add(prescription.write_to_kb)
            try:
                ok = await knowledge_writer.record_gap(
                    prescription.write_to_kb, owner=cfg.owner, loop_id=loop_id
                )
            except Exception as exc:
                logger.warning("SWT knowledge-gap write-back failed: %s", exc)
                ok = False
            yield {"type": "knowledge_gap_recorded", "round": i, "ok": ok}

        if (
            diagnosis.category == DiagnosisCategory.ORCHESTRATION_FAILURE
            and prescription.switch_generator_to
            and prescription.switch_generator_to != active_generator
        ):
            yield {
                "type": "model_switch",
                "round": i,
                "from": active_generator,
                "to": prescription.switch_generator_to,
            }
            active_generator = prescription.switch_generator_to

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
