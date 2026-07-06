"""Offline tests for the SWT loop.

A scripted fake model adapter drives the whole generator -> critic -> analyzer
-> cognitive loop with no network, so the orchestration, JSON parsing,
convergence, and persistence logic are exercised deterministically.
"""
import asyncio
import json
import os
import tempfile

import pytest

from src.swt import (
    CognitiveModel,
    LoopConfig,
    SwtStore,
    run_loop,
)
from src.swt.analyzer import analyze, _heuristic
from src.swt.cognitive_model import CognitiveModel as CM
from src.swt.recursive import condense_context, _split
from src.swt.schemas import Critique, DiagnosisCategory


class FakeAdapter:
    """Returns canned responses keyed by which persona is calling.

    The engine's system prompts are distinctive per role, so we route on a
    substring of the system message.
    """

    def __init__(self, script):
        self.script = script
        self.calls = []

    async def complete(self, model, messages, *, temperature=0.7, max_tokens=1024, owner=None):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        self.calls.append((model, system[:40]))
        for needle, response in self.script.items():
            if needle in system:
                return response(self) if callable(response) else response
        return "unmatched"


def _store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return SwtStore(db_path=path), path


async def _collect(cfg, adapter, store, cognitive):
    return [ev async for ev in run_loop(cfg, adapter, store, cognitive)]


def test_loop_converges_when_critic_accepts():
    store, path = _store()
    try:
        adapter = FakeAdapter({
            "careful assistant": "The capital of France is Paris.",
            "rigorous CRITIC": json.dumps({
                "accepted": True, "summary": "Correct and complete.",
                "issues": [], "missing_context": [],
            }),
        })
        cfg = LoopConfig(
            prompt="What is the capital of France?",
            generator_model="gen", critic_model="crit",
            use_cognitive_model=False, max_rounds=3,
        )
        events = asyncio.run(_collect(cfg, adapter, store, CM(store)))
        types = [e["type"] for e in events]
        assert types[0] == "loop_start"
        assert types[-1] == "loop_complete"
        final = events[-1]["result"]
        assert final["converged"] is True
        assert final["final_answer"] == "The capital of France is Paris."
        assert len(final["rounds"]) == 1  # accepted on round 0
    finally:
        os.remove(path)


def test_loop_iterates_and_applies_analyzer_hint():
    store, path = _store()
    try:
        state = {"gen_calls": 0}

        def gen(_a):
            state["gen_calls"] += 1
            return f"answer v{state['gen_calls']}"

        def critic(_a):
            # Reject the first answer, accept the second.
            if state["gen_calls"] <= 1:
                return json.dumps({
                    "accepted": False, "summary": "Wrong format.",
                    "issues": ["did not follow the requested format"],
                    "missing_context": [],
                })
            return json.dumps({"accepted": True, "summary": "Good now.", "issues": [], "missing_context": []})

        adapter = FakeAdapter({
            "careful assistant": gen,
            "rigorous CRITIC": critic,
            "ANALYZER": json.dumps({
                "category": "orchestration_failure",
                "rationale": "Format mismatch.",
                "next_prompt_hint": "Use a numbered list.",
                "retrieval_adjustment": "",
                "switch_generator_to": "",
            }),
        })
        cfg = LoopConfig(
            prompt="List three primes.", generator_model="gen", critic_model="crit",
            use_cognitive_model=False, max_rounds=4,
        )
        events = asyncio.run(_collect(cfg, adapter, store, CM(store)))
        final = events[-1]["result"]
        assert final["converged"] is True
        assert len(final["rounds"]) == 2
        # Round 0 diagnosis should be orchestration_failure.
        assert final["rounds"][0]["diagnosis"]["category"] == "orchestration_failure"
        assert final["final_answer"] == "answer v2"
    finally:
        os.remove(path)


def test_loop_stops_at_max_rounds_and_returns_best():
    store, path = _store()
    try:
        adapter = FakeAdapter({
            "careful assistant": "still wrong",
            "rigorous CRITIC": json.dumps({
                "accepted": False, "summary": "Nope.", "issues": ["bad"], "missing_context": [],
            }),
            "ANALYZER": json.dumps({
                "category": "knowledge_gap", "rationale": "Missing info.",
                "next_prompt_hint": "State assumptions.", "retrieval_adjustment": "", "switch_generator_to": "",
            }),
        })
        cfg = LoopConfig(
            prompt="Unanswerable?", generator_model="gen", critic_model="crit",
            use_cognitive_model=False, max_rounds=2,
        )
        events = asyncio.run(_collect(cfg, adapter, store, CM(store)))
        final = events[-1]["result"]
        assert final["converged"] is False
        assert len(final["rounds"]) == 2
        assert final["final_answer"] == "still wrong"
    finally:
        os.remove(path)


def test_unparseable_critic_treated_as_not_accepted():
    store, path = _store()
    try:
        adapter = FakeAdapter({
            "careful assistant": "answer",
            "rigorous CRITIC": "I think it's fine, ship it!",  # not JSON
            "ANALYZER": json.dumps({
                "category": "orchestration_failure", "rationale": "x",
                "next_prompt_hint": "", "retrieval_adjustment": "", "switch_generator_to": "",
            }),
        })
        cfg = LoopConfig(
            prompt="q", generator_model="gen", critic_model="crit",
            use_cognitive_model=False, max_rounds=1,
        )
        events = asyncio.run(_collect(cfg, adapter, store, CM(store)))
        crit_event = next(e for e in events if e["type"] == "critique")
        assert crit_event["critique"]["accepted"] is False
    finally:
        os.remove(path)


def test_generator_error_emits_error_event():
    store, path = _store()
    try:
        class Boom:
            async def complete(self, *a, **k):
                raise RuntimeError("endpoint down")

        cfg = LoopConfig(prompt="q", generator_model="gen", critic_model="crit",
                         use_cognitive_model=False, max_rounds=2)
        events = asyncio.run(_collect(cfg, Boom(), store, CM(store)))
        assert any(e["type"] == "error" and e["stage"] == "generator" for e in events)
        assert events[-1]["type"] == "loop_complete"
    finally:
        os.remove(path)


# -- cognitive model --------------------------------------------------------

def test_cognitive_cold_start_is_neutral():
    store, path = _store()
    try:
        cm = CM(store)
        pred = cm.predict("u", "prompt", "some answer")
        assert pred.basis == "cold-start"
        assert pred.score == 0.5
    finally:
        os.remove(path)


def test_cognitive_learns_from_feedback():
    store, path = _store()
    try:
        cm = CM(store)
        # Teach it: answers about apples accepted, answers about tanks rejected.
        cm.record("u", "q1", "apples are a healthy fruit rich in fiber", True)
        cm.record("u", "q2", "apples grow on trees and are crisp", True)
        cm.record("u", "q3", "military tanks use heavy armor plating", False)
        cm.record("u", "q4", "tanks fire large caliber shells in combat", False)
        good = cm.predict("u", "q", "fresh apples are a crisp healthy fruit")
        bad = cm.predict("u", "q", "the tank fired armor piercing shells")
        assert good.score > bad.score
        assert good.basis in ("history", "embedding")
    finally:
        os.remove(path)


def test_feedback_stats():
    store, path = _store()
    try:
        cm = CM(store)
        cm.record("u", "q", "a", True)
        cm.record("u", "q", "b", False)
        stats = store.feedback_stats("u")
        assert stats == {"total": 2, "accepted": 1, "rejected": 1}
    finally:
        os.remove(path)


# -- analyzer heuristic -----------------------------------------------------

def test_analyzer_heuristic_retrieval_vs_knowledge_gap():
    missing = Critique(accepted=False, summary="x", issues=[], missing_context=["the spec"])
    with_ctx = _heuristic(missing, had_context=True)
    without_ctx = _heuristic(missing, had_context=False)
    assert with_ctx.category == DiagnosisCategory.RETRIEVAL_FAILURE
    assert without_ctx.category == DiagnosisCategory.KNOWLEDGE_GAP


def test_analyzer_accepts_short_circuits():
    accepted = Critique(accepted=True, summary="great")
    diag = asyncio.run(analyze(FakeAdapter({}), "m", "p", "a", accepted, had_context=False))
    assert diag.category == DiagnosisCategory.NONE


# -- recursive splitter -----------------------------------------------------

def test_split_respects_budget():
    text = "\n\n".join(f"paragraph {i} " * 50 for i in range(20))
    chunks = _split(text, 2000)
    assert len(chunks) > 1
    assert all(len(c) <= 2000 + 600 for c in chunks)  # slack for paragraph boundaries


def test_condense_returns_short_context_unchanged():
    async def go():
        return await condense_context(FakeAdapter({}), "q", "short context", "m", budget_chars=6000)
    assert asyncio.run(go()) == "short context"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
