"""PostgreSQL experience store for the SWT loop (``swt_main`` schema).

The distributed deployment keeps the experience database in the shared
PostgreSQL instance rather than a per-box SQLite file, so every node reads
the same accumulated insight. Selected by ``create_store`` when
``SWT_DATABASE_URL`` is a ``postgresql://`` URL; requires ``psycopg`` (v3),
which is imported lazily so single-box installs never need it.

All tables live in their own ``swt_main`` schema -- created on first use --
so SWT still touches nothing owned by the rest of the application.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from src.swt.schemas import Diagnosis, LoopResult, new_id
from src.swt.store import EmbedFn, _decode_vec, _encode_vec, rank_similar_diagnoses

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS swt_main;

CREATE TABLE IF NOT EXISTS swt_main.swt_loops (
    loop_id     TEXT PRIMARY KEY,
    owner       TEXT,
    prompt      TEXT NOT NULL,
    result_json TEXT NOT NULL,
    converged   INTEGER NOT NULL DEFAULT 0,
    rounds      INTEGER NOT NULL DEFAULT 0,
    created_at  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS swt_main.swt_feedback (
    feedback_id TEXT PRIMARY KEY,
    owner       TEXT,
    loop_id     TEXT,
    prompt      TEXT NOT NULL,
    answer      TEXT NOT NULL,
    accepted    INTEGER NOT NULL,
    note        TEXT,
    answer_vec  TEXT,
    created_at  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS swt_main.swt_diagnoses (
    diagnosis_id      TEXT PRIMARY KEY,
    loop_id           TEXT NOT NULL,
    round_index       INTEGER NOT NULL,
    owner             TEXT,
    prompt            TEXT NOT NULL,
    prompt_vec        TEXT,
    category          TEXT NOT NULL,
    rationale         TEXT,
    prescription_json TEXT NOT NULL DEFAULT '{}',
    created_at        DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_swt_loops_owner ON swt_main.swt_loops(owner, created_at);
CREATE INDEX IF NOT EXISTS idx_swt_feedback_owner ON swt_main.swt_feedback(owner, created_at);
CREATE INDEX IF NOT EXISTS idx_swt_diagnoses_owner ON swt_main.swt_diagnoses(owner, created_at);
"""


class PostgresExperienceStore:
    """Same contract as :class:`~src.swt.store.SwtStore`, shared-DB backend."""

    def __init__(self, dsn: str, embed_fn: Optional[EmbedFn] = None) -> None:
        import psycopg  # lazy: optional dependency, only for the PG backend

        self._psycopg = psycopg
        self.dsn = dsn
        self.embed_fn = embed_fn
        with self._connect() as conn:
            conn.execute(_SCHEMA_SQL)

    def _connect(self):
        # Short-lived autocommit connections, mirroring the SQLite backend's
        # connection-per-call model so the store shares safely across workers.
        return self._psycopg.connect(self.dsn, autocommit=True)

    # -- loops ---------------------------------------------------------------

    def save_loop(self, result: LoopResult, owner: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO swt_main.swt_loops "
                "(loop_id, owner, prompt, result_json, converged, rounds, created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (loop_id) DO UPDATE SET "
                "result_json = EXCLUDED.result_json, converged = EXCLUDED.converged, "
                "rounds = EXCLUDED.rounds",
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
                "FROM swt_main.swt_loops WHERE owner IS NOT DISTINCT FROM %s "
                "ORDER BY created_at DESC LIMIT %s",
                (owner, limit),
            ).fetchall()
        cols = ("loop_id", "prompt", "converged", "rounds", "created_at")
        return [dict(zip(cols, r)) for r in rows]

    def get_loop(self, loop_id: str, owner: Optional[str]) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM swt_main.swt_loops "
                "WHERE loop_id = %s AND owner IS NOT DISTINCT FROM %s",
                (loop_id, owner),
            ).fetchone()
        return json.loads(row[0]) if row else None

    # -- diagnoses -------------------------------------------------------------

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
                "INSERT INTO swt_main.swt_diagnoses "
                "(diagnosis_id, loop_id, round_index, owner, prompt, prompt_vec, "
                " category, rationale, prescription_json, created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
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
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT loop_id, prompt, prompt_vec, category, rationale, "
                "       prescription_json, created_at "
                "FROM swt_main.swt_diagnoses "
                "WHERE owner IS NOT DISTINCT FROM %s AND category != 'none' "
                "ORDER BY created_at DESC LIMIT 200",
                (owner,),
            ).fetchall()
        cols = ("loop_id", "prompt", "prompt_vec", "category", "rationale",
                "prescription_json", "created_at")
        prompt_vec = _decode_vec(_encode_vec(self.embed_fn, prompt))
        return rank_similar_diagnoses([dict(zip(cols, r)) for r in rows], prompt, prompt_vec, k)

    # -- feedback --------------------------------------------------------------

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
                "INSERT INTO swt_main.swt_feedback "
                "(feedback_id, owner, loop_id, prompt, answer, accepted, note, answer_vec, created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (fid, owner, loop_id, prompt, answer, 1 if accepted else 0, note,
                 _encode_vec(self.embed_fn, answer), time.time()),
            )
        return fid

    def recent_feedback(self, owner: Optional[str], limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT prompt, answer, accepted, note, answer_vec, created_at "
                "FROM swt_main.swt_feedback WHERE owner IS NOT DISTINCT FROM %s "
                "ORDER BY created_at DESC LIMIT %s",
                (owner, limit),
            ).fetchall()
        cols = ("prompt", "answer", "accepted", "note", "answer_vec", "created_at")
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            d["answer_vec"] = _decode_vec(d.get("answer_vec"))
            out.append(d)
        return out

    def feedback_stats(self, owner: Optional[str]) -> Dict[str, int]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(accepted), 0) "
                "FROM swt_main.swt_feedback WHERE owner IS NOT DISTINCT FROM %s",
                (owner,),
            ).fetchone()
        total = int(row[0] or 0)
        accepted = int(row[1] or 0)
        return {"total": total, "accepted": accepted, "rejected": total - accepted}
