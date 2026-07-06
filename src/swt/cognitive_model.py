"""TRIBE v2-inspired cognitive model: predict user satisfaction.

TRIBE v2 (Meta, 2026) predicts a brain's response to a stimulus with zero
task-specific training. Applied to SWT the analogy is deliberate and bounded:
we predict *this user's* response to an answer from the history of what they
have accepted and rejected. It is a proxy user-cognition layer built from
interaction history -- not a claim to model anyone's brain.

The critic asks "is this answer good?". The cognitive model asks the different,
TRIBE-style question: "will THIS user accept it?" -- so the loop can converge on
what the user actually wants, and can flag an answer the critic likes but the
user historically would not.

Prediction is similarity-weighted over stored feedback: an answer that looks
like past *accepted* answers scores high; one that looks like past *rejected*
answers scores low. With no history it returns a neutral cold-start score and
lets the critic drive. An embedding function can be injected for semantic
similarity; the default is a dependency-free lexical overlap so the model works
offline.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Callable, List, Optional, Sequence

from src.swt.schemas import SatisfactionPrediction
from src.swt.store import SwtStore

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Sequence[float]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class CognitiveModel:
    """Similarity-weighted accept/reject predictor over stored feedback."""

    # Minimum feedback examples before we trust history over cold-start.
    MIN_HISTORY = 3

    def __init__(self, store: SwtStore, embed_fn: Optional[EmbedFn] = None) -> None:
        self.store = store
        self.embed_fn = embed_fn

    def record(
        self,
        owner: Optional[str],
        prompt: str,
        answer: str,
        accepted: bool,
        *,
        loop_id: Optional[str] = None,
        note: str = "",
    ) -> str:
        return self.store.add_feedback(
            owner, prompt, answer, accepted, loop_id=loop_id, note=note
        )

    def _similarity(self, answer: str, example_answer: str) -> float:
        if self.embed_fn is not None:
            try:
                return _cosine(self.embed_fn(answer), self.embed_fn(example_answer))
            except Exception as exc:  # fall back to lexical on embed failure
                logger.debug("SWT cognitive embed failed, lexical fallback: %s", exc)
        return _jaccard(_tokens(answer), _tokens(example_answer))

    def predict(
        self, owner: Optional[str], prompt: str, answer: str
    ) -> SatisfactionPrediction:
        history: List[dict] = self.store.recent_feedback(owner, limit=200)
        if len(history) < self.MIN_HISTORY:
            return SatisfactionPrediction(
                score=0.5,
                reasons=[
                    f"Only {len(history)} feedback example(s); need "
                    f"{self.MIN_HISTORY} to model preferences. Deferring to the critic."
                ],
                basis="cold-start",
            )

        # Weight each example by how much it looks like the candidate answer.
        num = 0.0
        den = 0.0
        best_pos = (0.0, "")
        best_neg = (0.0, "")
        for row in history:
            sim = self._similarity(answer, row.get("answer", ""))
            if sim <= 0:
                continue
            label = 1.0 if row.get("accepted") else 0.0
            num += sim * label
            den += sim
            if label and sim > best_pos[0]:
                best_pos = (sim, row.get("prompt", ""))
            if not label and sim > best_neg[0]:
                best_neg = (sim, row.get("prompt", ""))

        if den == 0:
            return SatisfactionPrediction(
                score=0.5,
                reasons=["No similar past answers; deferring to the critic."],
                basis="history",
            )

        score = num / den
        reasons: List[str] = []
        if best_pos[0] >= best_neg[0] and best_pos[0] > 0:
            reasons.append(
                f"Resembles a previously ACCEPTED answer (similarity {best_pos[0]:.2f})."
            )
        if best_neg[0] > best_pos[0]:
            reasons.append(
                f"Resembles a previously REJECTED answer (similarity {best_neg[0]:.2f})."
            )
        basis = "embedding" if self.embed_fn is not None else "history"
        return SatisfactionPrediction(score=round(score, 3), reasons=reasons, basis=basis)
