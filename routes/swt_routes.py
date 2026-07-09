# routes/swt_routes.py
"""SWT loop routes -- the automated generator/critic/analyzer workspace.

Endpoints:
    POST /api/swt/run              stream a loop run as SSE
    POST /api/swt/feedback         record accept/reject (trains the preference model)
    GET  /api/swt/history          recent loops for the current user
    GET  /api/swt/history/{id}     one full loop record
    GET  /api/swt/cognitive/stats  preference-model feedback stats
    GET  /api/swt/status           feature/capability probe

The heavy lifting lives in ``src/swt``; these routes only handle auth,
validation, and streaming.
"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.session_manager import SessionManager
from src.auth_helpers import _auth_disabled, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/swt", tags=["swt"])

# Process-wide singletons: one experience store + one preference model +
# one retriever, created lazily so importing this module never touches the
# filesystem, ChromaDB, or the LLM stack. The model adapter is NOT a
# singleton — it is created per run (see `run`) so its endpoint-resolution
# cache is scoped to a single loop and can never serve a stale API key after
# the user reconfigures an endpoint.
_store = None
_preference = None
_retriever = None
_embed_fn = None
_embed_probed = False


def _get_embed_fn():
    """Embedding hookup for semantic preference/experience matching.

    Probed once; None (no embedding backend) is a valid cached outcome and
    everything downstream degrades to lexical similarity.
    """
    global _embed_fn, _embed_probed
    if not _embed_probed:
        from src.swt import default_embed_fn

        _embed_fn = default_embed_fn()
        _embed_probed = True
    return _embed_fn


def _get_store():
    global _store
    if _store is None:
        from src.swt import create_store

        _store = create_store(embed_fn=_get_embed_fn())
    return _store


def _get_preference():
    global _preference
    if _preference is None:
        from src.swt import PreferenceModel

        _preference = PreferenceModel(_get_store(), embed_fn=_get_embed_fn())
    return _preference


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.swt import RagRetriever

        _retriever = RagRetriever()
    return _retriever


class RunRequest(BaseModel):
    # Length caps bound the work a single request can trigger. Without a cap on
    # `context`, a huge paste fans out into thousands of recursive-condense LLM
    # calls, which can hang a low-power node. ~200 KB is generous for reference
    # material while keeping the chunk count (and cost) bounded.
    prompt: str = Field(..., min_length=1, max_length=100_000)
    # Model fields accept full Odysseus specs, including "name@endpoint" to
    # pin a role to a specific configured endpoint (distributed deployments).
    generator_model: str = Field(..., min_length=1, max_length=200)
    critic_model: str = Field(..., min_length=1, max_length=200)
    analyzer_model: str = Field("", max_length=200)
    librarian_model: str = Field("", max_length=200)
    max_rounds: int = Field(4, ge=1, le=8)
    satisfaction_threshold: float = Field(0.7, ge=0.0, le=1.0)
    context: str = Field("", max_length=200_000)
    use_recursive_context: bool = True
    # Kept as `use_cognitive_model` on the wire for existing clients; maps to
    # LoopConfig.use_preference_model.
    use_cognitive_model: bool = True
    use_retrieval: bool = True
    retrieval_k: int = Field(4, ge=1, le=12)
    use_experience: bool = True
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class FeedbackRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100_000)
    answer: str = Field(..., min_length=1, max_length=200_000)
    accepted: bool
    loop_id: Optional[str] = Field(None, max_length=64)
    note: str = Field("", max_length=2_000)


def setup_swt_routes(session_manager: SessionManager):
    def _require_user(request: Request) -> str:
        user = get_current_user(request)
        if not user:
            if _auth_disabled():
                return ""
            raise HTTPException(401, "Not authenticated")
        return user

    @router.post("/run")
    async def run(request: Request, body: RunRequest):
        owner = _require_user(request) or None
        from src.swt import LoopConfig, run_loop

        cfg = LoopConfig(
            prompt=body.prompt,
            generator_model=body.generator_model,
            critic_model=body.critic_model,
            analyzer_model=body.analyzer_model,
            librarian_model=body.librarian_model,
            max_rounds=body.max_rounds,
            satisfaction_threshold=body.satisfaction_threshold,
            context=body.context,
            use_recursive_context=body.use_recursive_context,
            use_preference_model=body.use_cognitive_model,
            use_retrieval=body.use_retrieval,
            retrieval_k=body.retrieval_k,
            use_experience=body.use_experience,
            temperature=body.temperature,
            owner=owner,
        )

        from src.swt import OdysseusModelAdapter

        adapter = OdysseusModelAdapter()  # per-run: cache scoped to this loop

        async def _generate():
            try:
                async for event in run_loop(
                    cfg, adapter, _get_store(), _get_preference(),
                    retriever=_get_retriever(),
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:  # never leave the stream hanging
                logger.exception("SWT run failed")
                yield f"data: {json.dumps({'type': 'error', 'stage': 'engine', 'message': str(exc)[:400]})}\n\n"
                yield f"data: {json.dumps({'type': 'loop_complete', 'result': None})}\n\n"

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/feedback")
    def feedback(request: Request, body: FeedbackRequest):
        owner = _require_user(request) or None
        fid = _get_preference().record(
            owner, body.prompt, body.answer, body.accepted,
            loop_id=body.loop_id, note=body.note,
        )
        return {"feedback_id": fid, "stats": _get_store().feedback_stats(owner)}

    @router.get("/history")
    def history(request: Request, limit: int = 25):
        owner = _require_user(request) or None
        limit = max(1, min(100, limit))
        return {"loops": _get_store().list_loops(owner, limit=limit)}

    @router.get("/history/{loop_id}")
    def history_one(request: Request, loop_id: str):
        owner = _require_user(request) or None
        loop = _get_store().get_loop(loop_id, owner)
        if loop is None:
            raise HTTPException(404, "Loop not found")
        return loop

    @router.get("/cognitive/stats")
    def cognitive_stats(request: Request):
        owner = _require_user(request) or None
        return _get_store().feedback_stats(owner)

    @router.get("/status")
    def status(request: Request):
        _require_user(request)
        from src.swt.recursive import rlm_available

        return {
            "available": True,
            "rlm_available": rlm_available(),
            "retrieval_available": _get_retriever().available(),
            "embeddings_available": _get_embed_fn() is not None,
            "store_backend": type(_get_store()).__name__,
            "categories": ["retrieval_failure", "knowledge_gap", "orchestration_failure"],
        }

    return router
