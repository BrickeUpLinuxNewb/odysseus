"""The SWT experience store.

This is the ASI-EVOLVE "experience database": loops, accept/reject feedback,
and -- first-class, not buried in a result blob -- every diagnosis the
analyzer produced, so future runs can query what went wrong on similar
prompts and start smarter (see ``engine.run_loop``'s experience seeding).

The store is pluggable behind :class:`ExperienceStore`:

  - :class:`SwtStore` -- SQLite (``data/swt.db``), the zero-config default.
  - ``PostgresExperienceStore`` (``src/swt/store_pg.py``) -- the shared
    PostgreSQL database, ``swt_main`` schema, for deployments where the
    experience database is real infrastructure rather than a side file.

:func:`create_store` picks the backend from ``SWT_DATABASE_URL``.

Similarity search stores an embedding of each prompt when an ``embed_fn`` is
provided (see ``src.embeddings.get_embedding_client``) and ranks by cosine;
without embeddings it degrades to lexical Jaccard overlap so read-back still
works offline.

Every method opens a short-lived connection, so the store is safe to share
across the async request workers.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from src.swt.schemas import Diagnosis, LoopResult, new_id
from src.swt.similarity import similarity

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Sequence[float]]


class ExperienceStore(Protocol):
    """What the engine, routes, and preference model need from persistence."""

    def save_loop(self, result: LoopResult, owner: Optional[str]) -> None: ...
    def list_loops(self, owner: Optional[str], limit: int = 25) -> List[Dict[str, Any]]: ...
    def get_loop(self, loop_id: str, owner: Optional[str]) -> Optional[Dict[str, Any]]: ...
    def save_diagnosis(
        self, loop_id: str, round_index: int, diagnosis: Diagnosis,
        owner: Optional[str], *, prompt: str,
    ) -> None: ...
    def similar_prior_diagnoses(
        self, prompt: str, owner: Optional[str], *, k: int = 3,
    ) -> List[Dict[str, Any]]: ...
    def add_feedback(
        self, owner: Optional[str], prompt: str, answer: str, accepted: bool,
        loop_id: Optional[str] = None, note: str = "",
    ) -> str: ...
    def recent_feedback(self, owner: Optional[str], limit: int = 200) -> List[Dict[str, Any]]: ...
    def feedback_stats(self, owner: Optional[str]) -> Dict[str, int]: ...


# -- shared helpers (used by both backends) ---------------------------------

def _encode_vec(embed_fn: Optional[EmbedFn], text: str) -> Optional[str]:
    """Embed text to a JSON-encoded vector; None when embedding is off/broken."""
    if embed_fn is None:
        return None
    try:
        vec = embed_fn(text)
        return json.dumps([float(x) for x in vec]) if vec is not None else None
    except Exception as exc:
        logger.debug("SWT store embedding failed, storing without vector: %s", exc)
        return None


def _decode_vec(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    try:
        vec = json.loads(raw)
        return vec if isinstance(vec, list) else None
    except Exception:
        return None


def rank_similar_diagnoses(
    rows: List[Dict[str, Any]],
    prompt: str,
    prompt_vec: Optional[Sequence[float]],
    k: int,
    min_similarity: float = 0.1,
) -> List[Dict[str, Any]]:
    """Rank raw diagnosis rows by prompt similarity; shared across backends.

    Each row needs ``prompt``, ``prompt_vec`` (JSON or None), ``category``,
    ``rationale``, ``prescription_json``, ``loop_id``, ``created_at``.
    """
    scored: List[Dict[str, Any]] = []
    for row in rows:
        row_vec = _decode_vec(row.get("prompt_vec"))
        vec_pair = (prompt_vec, row_vec) if (prompt_vec is not None and row_vec is not None) else (None, None)
        sim = similarity(prompt, row.get("prompt") or "", vec_pair[0], vec_pair[1])
        if sim < min_similarity:
            continue
        try:
            prescription = json.loads(row.get("prescription_json") or "{}")
        except Exception:
            prescription = {}
        scored.append({
            "loop_id": row.get("loop_id"),
            "prompt": row.get("prompt"),
            "similarity": round(sim, 4),
            "created_at": row.get("created_at"),
            "diagnosis": Diagnosis.from_dict({
                "category": row.get("category"),
                "rationale": row.get("rationale"),
                "prescription": prescription,
            }),
        })
    scored.sort(key=lambda s: s["similarity"], reverse=True)
    return scored[:k]


# -- SQLite backend (zero-config default) ------------------------------------

def _default_db_path() -> str:
    from src.runtime_paths import get_default_data_dir

    data_dir = get_default_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "swt.db")


class SwtStore:
    """SQLite experience store -- self-contained, no shared-schema migration."""

    def __init__(self, db_path: Optional[str] = None, embed_fn: Optional[EmbedFn] = None) -> None:
        self.db_path = db_path or _default_db_path()
        self.embed_fn = embed_fn
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS swt_loops (
                    loop_id     TEXT PRIMARY KEY,
                    owner       TEXT,
                    prompt      TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    converged   INTEGER NOT NULL DEFAULT 0,
                    rounds      INTEGER NOT NULL DEFAULT 0,
                    created_at  REAL NOT NULL
                );

                -- The preference model's training data: what the user accepted
                -- or rejected, so future answers can be scored against it.
                CREATE TABLE IF NOT EXISTS swt_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    owner       TEXT,
                    loop_id     TEXT,
                    prompt      TEXT NOT NULL,
                    answer      TEXT NOT NULL,
                    accepted    INTEGER NOT NULL,
                    note        TEXT,
                    answer_vec  TEXT,
                    created_at  REAL NOT NULL
                );

                -- Every analyzer diagnosis as its own row: the structured
                -- insight future runs query back (ASI-EVOLVE's "experience
                -- database" is only real if it is read, not just written).
                CREATE TABLE IF NOT EXISTS swt_diagnoses (
                    diagnosis_id      TEXT PRIMARY KEY,
                    loop_id           TEXT NOT NULL,
                    round_index       INTEGER NOT NULL,
                    owner             TEXT,
                    prompt            TEXT NOT NULL,
                    prompt_vec        TEXT,
                    category          TEXT NOT NULL,
                    rationale         TEXT,
                    prescription_json TEXT NOT NULL DEFAULT '{}',
                    created_at        REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_loops_owner ON swt_loops(owner, created_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_owner ON swt_feedback(owner, created_at);
                CREATE INDEX IF NOT EXISTS idx_diagnoses_owner ON swt_diagnoses(owner, created_at);
                """
            )
            # Pre-existing swt.db files lack answer_vec; add it in place.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(swt_feedback)")}
            if "answer_vec" not in cols:
                conn.execute("ALTER TABLE swt_feedback ADD COLUMN answer_vec TEXT")

    # -- loops ---------------------------------------------------------------

    def save_loop(self, result: LoopResult, owner: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO swt_loops "
                "(loop_id, owner, prompt, result_json, converged, rounds, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    result.loop_id,
                    owner,
                    result.prompt,
                    json.dumps(result.to_dict()),
                    1 if result.converged else 0,
                    len(result.rounds),
                    result.created_at,
                ),
            )

    def list_loops(self, owner: Optional[str], limit: int = 25) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT loop_id, prompt, converged, rounds, created_at "
                "FROM swt_loops WHERE owner IS ? ORDER BY created_at DESC LIMIT ?",
                (owner, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_loop(self, loop_id: str, owner: Optional[str]) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM swt_loops WHERE loop_id = ? AND owner IS ?",
                (loop_id, owner),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    # -- diagnoses (the queryable experience database) -------------------------

    def save_diagnosis(
        self,
        loop_id: str,
        round_index: int,
        diagnosis: Diagnosis,
        owner: Optional[str],
        *,
        prompt: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO swt_diagnoses "
                "(diagnosis_id, loop_id, round_index, owner, prompt, prompt_vec, "
                " category, rationale, prescription_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    new_id("diag"),
                    loop_id,
                    round_index,
                    owner,
                    prompt,
                    _encode_vec(self.embed_fn, prompt),
                    diagnosis.category.value,
                    diagnosis.rationale,
                    json.dumps(diagnosis.prescription.to_dict()),
                    time.time(),
                ),
            )

    def similar_prior_diagnoses(
        self, prompt: str, owner: Optional[str], *, k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Prior diagnoses on prompts like this one, best match first.

        Returns dicts of ``{loop_id, prompt, similarity, created_at,
        diagnosis}`` where ``diagnosis`` is a reconstructed
        :class:`Diagnosis`. Candidates are the most recent 200 rows for the
        owner, ranked semantically when vectors exist, lexically otherwise.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT loop_id, prompt, prompt_vec, category, rationale, "
                "       prescription_json, created_at "
                "FROM swt_diagnoses WHERE owner IS ? AND category != 'none' "
                "ORDER BY created_at DESC LIMIT 200",
                (owner,),
            ).fetchall()
        prompt_vec = _decode_vec(_encode_vec(self.embed_fn, prompt))
        return rank_similar_diagnoses([dict(r) for r in rows], prompt, prompt_vec, k)

    # -- feedback (preference model training data) ----------------------------

    def add_feedback(
        self,
        owner: Optional[str],
        prompt: str,
        answer: str,
        accepted: bool,
        loop_id: Optional[str] = None,
        note: str = "",
    ) -> str:
        fid = new_id("fb")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO swt_feedback "
                "(feedback_id, owner, loop_id, prompt, answer, accepted, note, answer_vec, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (fid, owner, loop_id, prompt, answer, 1 if accepted else 0, note,
                 _encode_vec(self.embed_fn, answer), time.time()),
            )
        return fid

    def recent_feedback(self, owner: Optional[str], limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT prompt, answer, accepted, note, answer_vec, created_at "
                "FROM swt_feedback WHERE owner IS ? ORDER BY created_at DESC LIMIT ?",
                (owner, limit),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["answer_vec"] = _decode_vec(d.get("answer_vec"))
            out.append(d)
        return out

    def feedback_stats(self, owner: Optional[str]) -> Dict[str, int]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT "
                "  COUNT(*) AS total, "
                "  COALESCE(SUM(accepted), 0) AS accepted "
                "FROM swt_feedback WHERE owner IS ?",
                (owner,),
            ).fetchone()
        total = int(row["total"] or 0)
        accepted = int(row["accepted"] or 0)
        return {"total": total, "accepted": accepted, "rejected": total - accepted}


# -- backend selection --------------------------------------------------------

def create_store(embed_fn: Optional[EmbedFn] = None) -> ExperienceStore:
    """Pick the experience-store backend.

    ``SWT_DATABASE_URL`` set to a ``postgresql://...`` URL selects the shared
    PostgreSQL database (``swt_main`` schema); anything else -- including the
    driver being missing or the database unreachable -- falls back to the
    zero-config SQLite file so the feature never hard-fails on storage.
    """
    url = (os.getenv("SWT_DATABASE_URL") or "").strip()
    if url.startswith(("postgres://", "postgresql://")):
        try:
            from src.swt.store_pg import PostgresExperienceStore

            return PostgresExperienceStore(url, embed_fn=embed_fn)
        except Exception as exc:
            logger.warning(
                "SWT: PostgreSQL experience store unavailable (%s); using SQLite", exc
            )
    return SwtStore(embed_fn=embed_fn)
