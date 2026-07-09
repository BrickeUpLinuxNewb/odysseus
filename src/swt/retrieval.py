"""Knowledge-base retrieval for the SWT loop.

This is the cure for the analyzer's ``retrieval_failure`` diagnosis: before
the generator runs, top-k chunks for the prompt are pulled from the knowledge
base and fed in alongside any user-pasted context, and when the analyzer
prescribes a :class:`~src.swt.schemas.RetrievalAdjustment` the next round
re-retrieves with widened parameters. The same seam handles the
``knowledge_gap`` write-back: a diagnosed gap is recorded as a document so a
future run retrieves it instead of hitting the same wall.

``Retriever`` / ``KnowledgeWriter`` are protocols so the engine stays
offline-testable with fakes; :class:`RagRetriever` is the real implementation
over Odysseus's shared VectorRAG/ChromaDB instance (the distributed
deployment's "vault" node). Every method degrades to a no-op when ChromaDB is
unreachable -- retrieval going away must never stall the loop.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)

# Retrieval widening is bounded so a misbehaving analyzer cannot ask for an
# ever-growing k (each retrieved chunk lands in the generator prompt).
MAX_RETRIEVAL_K = 12
_SNIPPET_CHARS = 1500  # per-chunk cap before chunks enter the prompt


class Retriever(Protocol):
    async def retrieve(self, query: str, *, k: int, owner: Optional[str]) -> List[str]: ...


class KnowledgeWriter(Protocol):
    async def record_gap(self, text: str, *, owner: Optional[str], loop_id: str) -> bool: ...


class RagRetriever:
    """Retriever + KnowledgeWriter over the app's shared VectorRAG instance.

    Uses ``src.rag_singleton.get_rag_manager()`` (lazy, returns ``None`` while
    ChromaDB is unreachable) rather than constructing its own client, so SWT
    shares the vault node's connection, owner scoping, and retry throttling
    with the rest of Odysseus.
    """

    def __init__(self, rag=None) -> None:
        self._rag = rag  # injected in tests; resolved lazily otherwise

    def _manager(self):
        if self._rag is not None:
            return self._rag
        try:
            from src.rag_singleton import get_rag_manager
            return get_rag_manager()
        except Exception as exc:
            logger.debug("SWT retriever: RAG unavailable: %s", exc)
            return None

    def available(self) -> bool:
        return self._manager() is not None

    async def retrieve(self, query: str, *, k: int, owner: Optional[str]) -> List[str]:
        rag = self._manager()
        if rag is None or not query.strip():
            return []
        k = max(1, min(int(k), MAX_RETRIEVAL_K))
        try:
            # VectorRAG.search is synchronous (HTTP to ChromaDB under the
            # hood); keep it off the event loop.
            results = await asyncio.to_thread(rag.search, query, k, owner=owner)
        except Exception as exc:
            logger.warning("SWT retrieval failed: %s", exc)
            return []
        chunks: List[str] = []
        for r in results or []:
            doc = (r.get("document") or "").strip() if isinstance(r, dict) else ""
            if doc:
                chunks.append(doc[:_SNIPPET_CHARS])
        return chunks

    async def record_gap(self, text: str, *, owner: Optional[str], loop_id: str) -> bool:
        rag = self._manager()
        text = (text or "").strip()
        if rag is None or not text:
            return False
        metadata = {
            "source": "swt",
            "kind": "knowledge_gap",
            "loop_id": loop_id,
            "created_at": time.time(),
        }
        if owner:
            metadata["owner"] = owner
        try:
            return bool(await asyncio.to_thread(rag.add_document, text, metadata))
        except Exception as exc:
            logger.warning("SWT knowledge-gap write-back failed: %s", exc)
            return False


def retrieval_available() -> bool:
    """Capability probe for the status endpoint."""
    return RagRetriever().available()
