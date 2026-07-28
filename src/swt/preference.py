"""The SWT preference model: predict whether *this user* will accept an answer.

This is a per-user preference model learned from accept/reject feedback --
nothing more. (The idea of predicting a user's reaction to a stimulus from
history is loosely motivated by response-prediction work like TRIBE v2, but
that is motivation, not mechanism: there is no neuroscience here, just
embeddings and interaction statistics over what the user actually accepted.)

Scoring, in order of preference:

1. **Prototype margin (semantic).** When an ``embed_fn`` is available (wire in
   ``src.embeddings.get_embedding_client()``) and history contains at least one
   embedded accepted and one embedded rejected answer, the score is a logistic
   of ``cos(answer, accepted_centroid) - cos(answer, rejected_centroid)``.
2. **Similarity-weighted vote.** Otherwise each history example votes with a
   weight equal to its similarity to the candidate (cosine when both sides
   have vectors, Jaccard token overlap when not), and the score is the
   weighted fraction accepted.
3. **Cold start.** Below :attr:`MIN_HISTORY` examples the score stays a
   neutral 0.5 and the critic decides alone.

On top of the base score, cheap interaction features -- answer length, code
blocks, hedging density -- nudge the score toward whichever class the answer
resembles, because *how* an answer is shaped predicts acceptance for a given
user as much as what it says.

Answer vectors are computed once at feedback time and stored by the
experience store, so a prediction embeds only the candidate answer.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.swt.schemas import SatisfactionPrediction
from src.swt.similarity import cosine, jaccard, tokens
from src.swt.store import ExperienceStore

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Sequence[float]]

# Temperature of the prototype-margin logistic: a centroid-cosine gap of
# ~0.15 is already a strong signal with normalized sentence embeddings.
_MARGIN_TAU = 0.15
# Interaction features may move the base score by at most this much.
_FEATURE_CAP = 0.12
_FEATURE_STEP = 0.04

_HEDGES = (
    "might", "maybe", "perhaps", "possibly", "likely", "i think",
    "it depends", "could be", "not sure", "probably",
)
_CODE_RE = re.compile(r"```")


def _features(text: str) -> Dict[str, float]:
    """Cheap shape features that tend to predict a user's acceptance."""
    words = max(1, len((text or "").split()))
    hedge_hits = sum((text or "").lower().count(h) for h in _HEDGES)
    return {
        "log_length": math.log(max(1, len(text or ""))),
        "has_code": 1.0 if _CODE_RE.search(text or "") else 0.0,
        "hedge_density": hedge_hits / words,
    }

_FEATURE_LABELS = {
    "log_length": "length",
    "has_code": "code-block use",
    "hedge_density": "hedging",
}


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _feature_adjustment(
    answer: str, accepted_texts: List[str], rejected_texts: List[str]
) -> Tuple[float, List[str]]:
    """Nudge toward whichever class the answer's shape is closer to.

    Needs both classes; each feature where the classes actually differ moves
    the score one step toward the nearer class, capped at ``_FEATURE_CAP``.
    """
    if not accepted_texts or not rejected_texts:
        return 0.0, []
    answer_f = _features(answer)
    acc = [_features(t) for t in accepted_texts]
    rej = [_features(t) for t in rejected_texts]
    delta = 0.0
    toward_accept: List[str] = []
    toward_reject: List[str] = []
    for key, value in answer_f.items():
        acc_mean = _mean([f[key] for f in acc])
        rej_mean = _mean([f[key] for f in rej])
        spread = abs(acc_mean - rej_mean)
        if spread < 1e-6:
            continue  # this feature does not separate the classes for this user
        if abs(value - acc_mean) <= abs(value - rej_mean):
            delta += _FEATURE_STEP
            toward_accept.append(_FEATURE_LABELS[key])
        else:
            delta -= _FEATURE_STEP
            toward_reject.append(_FEATURE_LABELS[key])
    delta = max(-_FEATURE_CAP, min(_FEATURE_CAP, delta))
    reasons = []
    if toward_accept:
        reasons.append("Shape matches accepted answers (" + ", ".join(toward_accept) + ").")
    if toward_reject:
        reasons.append("Shape matches rejected answers (" + ", ".join(toward_reject) + ").")
    return delta, reasons


def _centroid(vectors: List[Sequence[float]]) -> Optional[List[float]]:
    if not vectors:
        return None
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            return None
        for i, x in enumerate(v):
            out[i] += float(x)
    return [x / len(vectors) for x in out]


class PreferenceModel:
    """Per-user accept/reject predictor over stored feedback."""

    MIN_HISTORY = 3  # below this, defer to the critic (cold-start)

    def __init__(self, store: ExperienceStore, embed_fn: Optional[EmbedFn] = None) -> None:
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

    def _embed(self, text: str) -> Optional[List[float]]:
        if self.embed_fn is None:
            return None
        try:
            vec = self.embed_fn(text)
            return [float(x) for x in vec] if vec is not None else None
        except Exception as exc:
            logger.debug("SWT preference embed failed, lexical fallback: %s", exc)
            return None

    def predict(self, owner: Optional[str], prompt: str, answer: str) -> SatisfactionPrediction:
        history: List[Dict[str, Any]] = self.store.recent_feedback(owner, limit=200)
        if len(history) < self.MIN_HISTORY:
            return SatisfactionPrediction(
                score=0.5,
                reasons=[
                    f"Only {len(history)} feedback example(s); need {self.MIN_HISTORY} "
                    f"to model preferences. Deferring to the critic."
                ],
                basis="cold-start",
            )

        answer_vec = self._embed(answer)
        accepted_texts = [r.get("answer", "") for r in history if r.get("accepted")]
        rejected_texts = [r.get("answer", "") for r in history if not r.get("accepted")]

        base, basis, reasons = self._base_score(answer, answer_vec, history)
        delta, feature_reasons = _feature_adjustment(answer, accepted_texts, rejected_texts)
        score = max(0.0, min(1.0, base + delta))
        return SatisfactionPrediction(
            score=round(score, 3), reasons=reasons + feature_reasons, basis=basis
        )

    def _base_score(
        self,
        answer: str,
        answer_vec: Optional[List[float]],
        history: List[Dict[str, Any]],
    ) -> Tuple[float, str, List[str]]:
        # 1. Prototype margin, when embeddings cover both classes.
        if answer_vec is not None:
            acc_vecs = [r["answer_vec"] for r in history if r.get("accepted") and r.get("answer_vec")]
            rej_vecs = [r["answer_vec"] for r in history if not r.get("accepted") and r.get("answer_vec")]
            acc_c = _centroid(acc_vecs)
            rej_c = _centroid(rej_vecs)
            if acc_c is not None and rej_c is not None:
                margin = cosine(answer_vec, acc_c) - cosine(answer_vec, rej_c)
                score = 1.0 / (1.0 + math.exp(-margin / _MARGIN_TAU))
                side = "accepted" if margin >= 0 else "rejected"
                return score, "embedding", [
                    f"Semantically closer to previously {side.upper()} answers "
                    f"(margin {margin:+.2f})."
                ]

        # 2. Similarity-weighted vote (semantic per pair when possible).
        answer_toks = tokens(answer)
        num = den = 0.0
        best_pos = best_neg = 0.0
        used_embedding = False
        for row in history:
            row_vec = row.get("answer_vec")
            if answer_vec is not None and row_vec:
                sim = cosine(answer_vec, row_vec)
                used_embedding = True
            else:
                sim = jaccard(answer_toks, tokens(row.get("answer", "")))
            if sim <= 0:
                continue
            label = 1.0 if row.get("accepted") else 0.0
            num += sim * label
            den += sim
            if label:
                best_pos = max(best_pos, sim)
            else:
                best_neg = max(best_neg, sim)
        if den == 0:
            return 0.5, "lexical", ["No similar past answers; deferring to the critic."]
        reasons = []
        if best_pos >= best_neg and best_pos > 0:
            reasons.append(f"Resembles a previously ACCEPTED answer (similarity {best_pos:.2f}).")
        if best_neg > best_pos:
            reasons.append(f"Resembles a previously REJECTED answer (similarity {best_neg:.2f}).")
        return num / den, ("embedding" if used_embedding else "lexical"), reasons


def default_embed_fn() -> Optional[EmbedFn]:
    """The standard embedding hookup: Odysseus's shared embedding client.

    Returns None when no embedding backend is available (no EMBEDDING_URL and
    no local FastEmbed) -- the model then runs lexically, which is honest but
    weaker; the status endpoint surfaces which mode is active.
    """
    try:
        from src.embeddings import get_embedding_client

        client = get_embedding_client()
    except Exception as exc:
        logger.debug("SWT: embedding client unavailable: %s", exc)
        return None
    if client is None:
        return None

    def _embed(text: str) -> Sequence[float]:
        return client.encode([text])[0].tolist()

    return _embed
