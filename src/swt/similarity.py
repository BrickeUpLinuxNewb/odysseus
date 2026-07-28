"""Text/vector similarity helpers shared by the experience store and the
preference model.

Two tiers, chosen per call by whether an embedding is available:
  - cosine over embedding vectors (semantic; vectors from
    ``src.embeddings.get_embedding_client()`` come back L2-normalized, so
    cosine is a plain dot product)
  - Jaccard token overlap (lexical; dependency-free fallback so everything
    still works with no embedding backend at all)
"""
from __future__ import annotations

import math
import re
from typing import Optional, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not len(a) or not len(b) or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def similarity(
    text_a: str,
    text_b: str,
    vec_a: Optional[Sequence[float]] = None,
    vec_b: Optional[Sequence[float]] = None,
) -> float:
    """Cosine when both vectors are present, Jaccard otherwise."""
    if vec_a is not None and vec_b is not None:
        return cosine(vec_a, vec_b)
    return jaccard(tokens(text_a), tokens(text_b))
