# routes/swt_routes.py
"""SWT loop routes -- the automated generator/critic/analyzer workspace.

Endpoints:
    POST /api/swt/run              stream a loop run as SSE
    POST /api/swt/feedback         record accept/reject (trains cognitive model)
    GET  /api/swt/history          recent loops for the current user
    GET  /api/swt/history/{id}     one full loop record
    GET  /api/swt/cognitive/stats  cognitive-model feedback stats
    GET  /api/swt/status           feature/capability probe (rlm availability)

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

# Process-wide singletons: one store + one cognitive model, created lazily so
# importing this module never touches the filesystem or the LLM stack.
_store = None
_cognitive = None
_adapter = None


def _get_store():
    global _store
    if _store is None:
        from src.swt import SwtStore

        _store = SwtStore()
    return _store


def _get_cognitive():
    global _cognitive
    if _cognitive is None:
        from src.swt import CognitiveModel

        _cognitive = CognitiveModel(_get_store())
    return _cognitive


def _get_adapter():
    global _adapter
    if _adapter is None:
        from src.swt import OdysseusModelAdapter

        _adapter = OdysseusModelAdapter()
    return _adapter


class RunRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    generator_model: str = Field(..., min_length=1)
    critic_model: str = Field(..., min_length=1)
    analyzer_model: str = ""
    max_rounds: int = Field(4, ge=1, le=8)
    satisfaction_threshold: float = Field(0.7, ge=0.0, le=1.0)
    context: str = ""
    use_recursive_context: bool = True
    use_cognitive_model: bool = True
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class FeedbackRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    accepted: bool
    loop_id: Optional[str] = None
    note: str = ""


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
            max_rounds=body.max_rounds,
            satisfaction_threshold=body.satisfaction_threshold,
            context=body.context,
            use_recursive_context=body.use_recursive_context,
            use_cognitive_model=body.use_cognitive_model,
            temperature=body.temperature,
            owner=owner,
        )

        async def _generate():
            try:
                async for event in run_loop(
                    cfg, _get_adapter(), _get_store(), _get_cognitive()
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
        fid = _get_cognitive().record(
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
            "categories": ["retrieval_failure", "knowledge_gap", "orchestration_failure"],
        }

    return router
