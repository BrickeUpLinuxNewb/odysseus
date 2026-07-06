"""Data structures for the SWT (Save The World) loop.

SWT is Odysseus's automated generator -> critic -> analyzer loop. Where
``Compare`` runs models side by side for a human to judge, SWT closes the
judging loop itself: a generator answers, a critic finds what is wrong, and an
analyzer diagnoses *why* it was wrong and adjusts the next round. The three
diagnosis categories come straight from the ASI-EVOLVE analyzer framework:

    retrieval_failure    -- the context existed but the generator never saw it
    knowledge_gap        -- the context was never stored in the first place
    orchestration_failure-- the prompt / model was wrong for this task class

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
    """Everything needed to run one loop, resolved from the request."""

    prompt: str
    generator_model: str
    critic_model: str
    analyzer_model: str = ""            # defaults to critic_model when empty
    max_rounds: int = 4
    # Convergence: stop early once the critic accepts AND the cognitive model
    # predicts the user will accept with at least this confidence.
    satisfaction_threshold: float = 0.7
    context: str = ""                   # optional long reference material
    use_recursive_context: bool = True  # condense long context via RLM
    use_cognitive_model: bool = True    # predict user satisfaction each round
    temperature: float = 0.7
    owner: Optional[str] = None

    def resolved_analyzer(self) -> str:
        return self.analyzer_model or self.critic_model


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
class Diagnosis:
    """The analyzer's structured read of a critique plus its prescription."""

    category: DiagnosisCategory
    rationale: str
    # Concrete adjustments the engine applies to the next round.
    next_prompt_hint: str = ""          # appended to the generator system prompt
    retrieval_adjustment: str = ""      # how to change context retrieval
    switch_generator_to: str = ""       # model swap suggestion (orchestration)
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class SatisfactionPrediction:
    """TRIBE v2-inspired prediction of whether the user will accept an answer.

    The real TRIBE v2 predicts brain responses to stimuli with zero
    task-specific training. Applied locally, we predict the *user's* response
    to an answer from the history of what they have accepted and rejected --
    a proxy user-cognition layer, not a claim to model a brain.
    """

    score: float                        # 0..1, higher == more likely accepted
    reasons: List[str] = field(default_factory=list)
    basis: str = "cold-start"           # "history", "embedding", "cold-start"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoundResult:
    """One full generator -> critic -> analyzer -> cognitive pass."""

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
