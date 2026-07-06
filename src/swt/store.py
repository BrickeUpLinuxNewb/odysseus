"""Isolated persistence for the SWT loop.

SWT keeps its own SQLite database (``data/swt.db``) instead of adding tables to
Odysseus's main schema. That keeps the feature self-contained -- no migration
of the shared database, no risk to existing tables -- while still giving the
analyzer a durable experience database (the ASI-EVOLVE "experience database"
role) and the cognitive model a place to learn accept/reject history.

Every method opens a short-lived connection, so the store is safe to share
across the async request workers.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from src.swt.schemas import LoopResult, new_id


def _default_db_path() -> str:
    from src.runtime_paths import get_default_data_dir

    data_dir = get_default_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "swt.db")


class SwtStore:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or _default_db_path()
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

                -- The cognitive model's training data: what the user accepted
                -- or rejected, so future answers can be scored against it.
                CREATE TABLE IF NOT EXISTS swt_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    owner       TEXT,
                    loop_id     TEXT,
                    prompt      TEXT NOT NULL,
                    answer      TEXT NOT NULL,
                    accepted    INTEGER NOT NULL,
                    note        TEXT,
                    created_at  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_loops_owner ON swt_loops(owner, created_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_owner ON swt_feedback(owner, created_at);
                """
            )

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

    # -- feedback (cognitive model training data) ----------------------------

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
                "(feedback_id, owner, loop_id, prompt, answer, accepted, note, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (fid, owner, loop_id, prompt, answer, 1 if accepted else 0, note, time.time()),
            )
        return fid

    def recent_feedback(self, owner: Optional[str], limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT prompt, answer, accepted, note, created_at "
                "FROM swt_feedback WHERE owner IS ? ORDER BY created_at DESC LIMIT ?",
                (owner, limit),
            ).fetchall()
        return [dict(r) for r in rows]

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
