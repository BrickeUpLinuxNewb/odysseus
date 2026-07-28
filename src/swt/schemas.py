"""Data structures for the SWT (Save The World) loop.

SWT is Odysseus's automated generator -> critic -> analyzer loop. Where
``Compare`` runs models side by side for a human to judge, SWT closes the
judging loop itself: a generator answers, a critic finds what is wrong, and an
analyzer diagnoses *why* it was wrong and prescribes the next round's
adjustment. The three diagnosis categories follow the ASI-EVOLVE analyzer
taxonomy:

    retrieval_failure    -- the context existed but the generator never saw it
    knowledge_gap        -- the context was never stored in the first place
    orchestration_failure-- the prompt / model was wrong for this task class

Each category maps to a typed lever in :class:`Prescription`, which the engine
dispatches instead of interpreting free text.

These dataclasses are the wire format between the engine, the store, and the
API. They are intentionally free of any LLM or DB dependency so they can be
constructed and tested in isolation.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class DiagnosisCategory(str, Enum):
    """The three questions the analyzer asks about every critique."""

    RETRIEVAL_FAILURE = "retrieval_failure"
    KNOWLEDGE_GAP = "knowledge_gap"
    ORCHESTRATION_FAILURE = "orchestration_failure"
    NONE = "none"  # critic was satisfied; nothing to diagnose


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class LoopConfig:
    """Everything needed to run one loop, resolved from the request.

    Each role's model field is a full Odysseus model spec, so
    ``"hermes3:8b@gpu-node"`` routes that role to a specific configured
    endpoint (``_resolve_model``'s ``model@endpoint`` syntax). This is how the
    distributed deployment maps generator / critic / analyzer / librarian to
    different nodes; with bare model names everything resolves on one box.
    """

    prompt: str
    generator_model: str
    critic_model: str
    analyzer_model: str = ""            # defaults to critic_model when empty
    librarian_model: str = ""           # context work (condense); defaults to analyzer
    max_rounds: int = 4
    # Convergence: stop early once the critic accepts AND the preference model
    # predicts the user will accept with at least this confidence.
    satisfaction_threshold: float = 0.7
    context: str = ""                   # optional long reference material
    use_recursive_context: bool = True  # condense long context via RLM
    use_preference_model: bool = True   # predict user satisfaction each round
    use_retrieval: bool = True          # pull top-k KB chunks before generating
    retrieval_k: int = 4                # initial top-k for KB retrieval
    use_experience: bool = True         # seed hints from prior similar diagnoses
    temperature: float = 0.7
    owner: Optional[str] = None

    def resolved_analyzer(self) -> str:
        return self.analyzer_model or self.critic_model

    def resolved_librarian(self) -> str:
        return self.librarian_model or self.resolved_analyzer()


@dataclass
class Critique:
    """The critic's verdict on one generated answer."""

    accepted: bool
    summary: str
    issues: List[str] = field(default_factory=list)
    missing_context: List[str] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalAdjustment:
    """How the next round's KB retrieval should differ from this round's."""

    widen_k: int = 0        # add this many results to the current top-k
    query_rewrite: str = "" # replacement retrieval query; empty keeps the current one
    note: str = ""          # analyzer's own words, kept for the record

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalAdjustment":
        return cls(
            widen_k=int(data.get("widen_k") or 0),
            query_rewrite=str(data.get("query_rewrite") or ""),
            note=str(data.get("note") or ""),
        )


@dataclass
class Prescription:
    """The analyzer's typed levers; the engine dispatches on these fields.

    One lever per category:
      retrieval_failure     -> ``retrieval`` (re-run retrieval, wider / rewritten)
      knowledge_gap         -> ``write_to_kb`` (record the gap so future runs see it)
      orchestration_failure -> ``switch_generator_to``
    ``prompt_hint`` applies to any category and accumulates across rounds.
    """

    prompt_hint: str = ""
    retrieval: Optional[RetrievalAdjustment] = None
    switch_generator_to: str = ""
    write_to_kb: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hint": self.prompt_hint,
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "switch_generator_to": self.switch_generator_to,
            "write_to_kb": self.write_to_kb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prescription":
        retrieval = data.get("retrieval")
        return cls(
            prompt_hint=str(data.get("prompt_hint") or ""),
            retrieval=RetrievalAdjustment.from_dict(retrieval) if isinstance(retrieval, dict) else None,
            switch_generator_to=str(data.get("switch_generator_to") or ""),
            write_to_kb=str(data.get("write_to_kb") or ""),
        )


@dataclass
class Diagnosis:
    """The analyzer's structured read of a critique plus its prescription."""

    category: DiagnosisCategory
    rationale: str
    prescription: Prescription = field(default_factory=Prescription)
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "rationale": self.rationale,
            "prescription": self.prescription.to_dict(),
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Diagnosis":
        try:
            category = DiagnosisCategory(str(data.get("category", "none")))
        except ValueError:
            category = DiagnosisCategory.NONE
        prescription = data.get("prescription")
        return cls(
            category=category,
            rationale=str(data.get("rationale") or ""),
            prescription=Prescription.from_dict(prescription) if isinstance(prescription, dict) else Prescription(),
            raw=str(data.get("raw") or ""),
        )


@dataclass
class SatisfactionPrediction:
    """The preference model's guess at whether the user will accept an answer.

    Learned entirely from this user's accept/reject history -- a proxy for
    their preferences, not a claim to model cognition. Below the history
    threshold the score stays neutral (0.5) and the critic decides alone.
    """

    score: float                        # 0..1, higher == more likely accepted
    reasons: List[str] = field(default_factory=list)
    basis: str = "cold-start"           # "embedding", "lexical", "cold-start"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoundResult:
    """One full generator -> critic -> analyzer -> preference pass."""

    round_index: int
    answer: str
    critique: Critique
    diagnosis: Diagnosis
    satisfaction: Optional[SatisfactionPrediction] = None
    generator_model: str = ""
    critic_model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "answer": self.answer,
            "critique": self.critique.to_dict(),
            "diagnosis": self.diagnosis.to_dict(),
            "satisfaction": self.satisfaction.to_dict() if self.satisfaction else None,
            "generator_model": self.generator_model,
            "critic_model": self.critic_model,
        }


@dataclass
class LoopResult:
    """The finished loop: every round plus the converged answer."""

    loop_id: str
    prompt: str
    rounds: List[RoundResult] = field(default_factory=list)
    final_answer: str = ""
    converged: bool = False
    converged_reason: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "prompt": self.prompt,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_answer": self.final_answer,
            "converged": self.converged,
            "converged_reason": self.converged_reason,
            "created_at": self.created_at,
        }
