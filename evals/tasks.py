"""Task loading for the eval harness.

Tasks live in JSONL files (one JSON object per line, ``#`` comments and blank
lines allowed). Rubrics are written IN the file, in advance -- the blind part
of the protocol. Format:

    {"id": "route-001", "kind": "choice", "tags": ["routing"],
     "system": "Answer with exactly one word from: email, code, ...",
     "prompt": "Reply to Sarah's message about the Q3 invoice",
     "expect": {"choices": ["email", "code"], "answer": "email"}}

``kind`` must be one of ``checkers.KNOWN_KINDS``. Mined-but-unrubric'd tasks
use ``"kind": "todo"`` and are rejected at load time with a pointed error --
a task without a pre-set rubric is not a task yet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from evals.checkers import KNOWN_KINDS


@dataclass
class Task:
    id: str
    kind: str
    prompt: str
    expect: Dict[str, Any] = field(default_factory=dict)
    system: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "")),
            prompt=str(data.get("prompt", "")),
            expect=dict(data.get("expect") or {}),
            system=str(data.get("system", "")),
            tags=[str(t) for t in (data.get("tags") or [])],
        )


def load_tasks(path: str) -> List[Task]:
    tasks: List[Task] = []
    seen: set = set()
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        task = Task.from_dict(data)
        if not task.id:
            raise ValueError(f"{path}:{lineno}: task missing 'id'")
        if task.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate task id {task.id!r}")
        seen.add(task.id)
        if task.kind == "todo":
            raise ValueError(
                f"{path}:{lineno}: task {task.id!r} is kind 'todo' -- mined "
                "prompts need a rubric written before they can be run. "
                "Decide what a good answer looks like, set a real kind, "
                "then run."
            )
        if task.kind not in KNOWN_KINDS:
            raise ValueError(
                f"{path}:{lineno}: task {task.id!r} has unknown kind "
                f"{task.kind!r} (known: {', '.join(KNOWN_KINDS)})"
            )
        if not task.prompt:
            raise ValueError(f"{path}:{lineno}: task {task.id!r} missing 'prompt'")
        tasks.append(task)
    if not tasks:
        raise ValueError(f"{path}: no tasks found")
    return tasks
