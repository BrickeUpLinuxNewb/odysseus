"""Offline tests for the SWT loop.

A scripted fake model adapter drives the whole generator -> critic -> analyzer
-> preference loop with no network, so the orchestration, JSON parsing,
convergence, retrieval dispatch, experience read-back, and persistence logic
are exercised deterministically. Fake Retriever / KnowledgeWriter / embed_fn
implementations stand in for ChromaDB and the embedding client the same way.
"""
import asyncio
import json
import os
import tempfile

import pytest

from src.swt import (
    LoopConfig,
    PreferenceModel,
    SwtStore,
    create_store,
    run_loop,
)
from src.swt.analyzer import WIDEN_K_STEP, analyze, _heuristic
from src.swt.preference import _feature_adjustment
from src.swt.recursive import condense_context, _split
from src.swt.schemas import (
    Critique,
    Diagnosis,
    DiagnosisCategory,
    Prescription,
)


class FakeAdapter:
    """Returns canned responses keyed by which persona is calling.

    The engine's system prompts are distinctive per role, so we route on a
    substring of the system message. Full messages are recorded so tests can
    assert on what each role was shown.
    """

    def __init__(self, script):
        self.script = script
        self.calls = []

    async def complete(self, model, messages, *, temperature=0.7, max_tokens=1024, owner=None):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        self.calls.append((model, messages))
        for needle, response in self.script.items():
            if needle in system:
                return response(self) if callable(response) else response
        return "unmatched"

    def generator_messages(self):
        """Messages of every call routed to the generator persona."""
        return [m for _, m in self.calls if "careful assistant" in m[0]["content"]]


class FakeRetriever:
    """Scripted Retriever + KnowledgeWriter; records every call."""

    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.queries = []           # (query, k) per retrieve call
        self.gaps = []              # text per record_gap call

    async def retrieve(self, query, *, k, owner):
        self.queries.append((query, k))
        return list(self.chunks)

    async def record_gap(self, text, *, owner, loop_id):
        self.gaps.append(text)
        return True


def _store(embed_fn=None):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return SwtStore(db_path=path, embed_fn=embed_fn), path


async def _collect(cfg, adapter, store, preference, retriever=None):
    return [ev async for ev in run_loop(cfg, adapter, store, preference, retriever=retriever)]


_ACCEPT = json.dumps({"accepted": True, "summary": "Good.", "issues": [], "missing_context": []})


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
            use_preference_model=False, max_rounds=3,
        )
        events = asyncio.run(_collect(cfg, adapter, store, PreferenceModel(store)))
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
            return _ACCEPT

        adapter = FakeAdapter({
            "careful assistant": gen,
            "rigorous CRITIC": critic,
            "ANALYZER": json.dumps({
                "category": "orchestration_failure",
                "rationale": "Format mismatch.",
                "next_prompt_hint": "Use a numbered list.",
                "switch_generator_to": "",
            }),
        })
        cfg = LoopConfig(
            prompt="List three primes.", generator_model="gen", critic_model="crit",
            use_preference_model=False, max_rounds=4,
        )
        events = asyncio.run(_collect(cfg, adapter, store, PreferenceModel(store)))
        final = events[-1]["result"]
        assert final["converged"] is True
        assert len(final["rounds"]) == 2
        # Round 0 diagnosis should be orchestration_failure.
        assert final["rounds"][0]["diagnosis"]["category"] == "orchestration_failure"
        assert final["final_answer"] == "answer v2"
        # The hint was applied to round 1's generator system prompt.
        round1_system = adapter.generator_messages()[1][0]["content"]
        assert "Use a numbered list." in round1_system
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
                "next_prompt_hint": "State assumptions.",
            }),
        })
        cfg = LoopConfig(
            prompt="Unanswerable?", generator_model="gen", critic_model="crit",
            use_preference_model=False, max_rounds=2,
        )
        events = asyncio.run(_collect(cfg, adapter, store, PreferenceModel(store)))
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
            }),
        })
        cfg = LoopConfig(
            prompt="q", generator_model="gen", critic_model="crit",
            use_preference_model=False, max_rounds=1,
        )
        events = asyncio.run(_collect(cfg, adapter, store, PreferenceModel(store)))
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
                         use_preference_model=False, max_rounds=2)
        events = asyncio.run(_collect(cfg, Boom(), store, PreferenceModel(store)))
        assert any(e["type"] == "error" and e["stage"] == "generator" for e in events)
        assert events[-1]["type"] == "loop_complete"
    finally:
        os.remove(path)


# -- retrieval (D1: the cure for retrieval_failure) ---------------------------

def test_retrieved_chunks_reach_the_generator():
    store, path = _store()
    try:
        adapter = FakeAdapter({
            "careful assistant": "answer",
            "rigorous CRITIC": _ACCEPT,
        })
        retriever = FakeRetriever(chunks=["the launch code is 0000"])
        cfg = LoopConfig(prompt="what is the launch code?", generator_model="gen",
                         critic_model="crit", use_preference_model=False, max_rounds=1)
        events = asyncio.run(_collect(cfg, adapter, store, PreferenceModel(store), retriever))
        assert retriever.queries == [("what is the launch code?", cfg.retrieval_k)]
        user_msg = adapter.generator_messages()[0][1]["content"]
        assert "RETRIEVED KNOWLEDGE" in user_msg
        assert "the launch code is 0000" in user_msg
        retrieval_ev = next(e for e in events if e["type"] == "retrieval")
        assert retrieval_ev["chunks"] == 1
    finally:
        os.remove(path)


def test_retrieval_failure_prescription_reruns_retrieval_widened():
    store, path = _store()
    try:
        state = {"round": 0}

        def critic(_a):
            state["round"] += 1
            if state["round"] == 1:
                return json.dumps({
                    "accepted": False, "summary": "Missed the spec.",
                    "issues": ["ignored the provided spec"],
                    "missing_context": ["the spec"],
                })
            return _ACCEPT

        adapter = FakeAdapter({
            "careful assistant": "answer",
            "rigorous CRITIC": critic,
            "ANALYZER": json.dumps({
                "category": "retrieval_failure",
                "rationale": "Context existed but was not surfaced.",
                "next_prompt_hint": "",
                "retrieval_query_rewrite": "project spec requirements",
            }),
        })
        retriever = FakeRetriever(chunks=["spec chunk"])
        cfg = LoopConfig(prompt="summarize the spec", generator_model="gen",
                         critic_model="crit", use_preference_model=False,
                         max_rounds=3, retrieval_k=4)
        events = asyncio.run(_collect(cfg, adapter, store, PreferenceModel(store), retriever))
        # Round 0 retrieves with the prompt; the prescription rewrites the
        # query and widens k for round 1.
        assert retriever.queries[0] == ("summarize the spec", 4)
        assert retriever.queries[1] == ("project spec requirements", 4 + WIDEN_K_STEP)
        assert any(e["type"] == "retrieval_adjusted" for e in events)
    finally:
        os.remove(path)


def test_knowledge_gap_written_back_to_kb():
    store, path = _store()
    try:
        adapter = FakeAdapter({
            "careful assistant": "I do not know the deploy target.",
            "rigorous CRITIC": json.dumps({
                "accepted": False, "summary": "Missing deploy target.",
                "issues": ["deploy target unknown"],
                "missing_context": ["the deploy target"],
            }),
            "ANALYZER": json.dumps({
                "category": "knowledge_gap",
                "rationale": "Never provided.",
                "knowledge_note": "Known gap: the deploy target is undocumented.",
            }),
        })
        retriever = FakeRetriever()
        cfg = LoopConfig(prompt="where do we deploy?", generator_model="gen",
                         critic_model="crit", use_preference_model=False, max_rounds=2)
        events = asyncio.run(_collect(cfg, adapter, store, PreferenceModel(store), retriever))
        # Written once, despite the same diagnosis in both rounds.
        assert retriever.gaps == ["Known gap: the deploy target is undocumented."]
        assert any(e["type"] == "knowledge_gap_recorded" and e["ok"] for e in events)
    finally:
        os.remove(path)


# -- experience store (D2/D3: insight persists and is read back) --------------

def _diagnosis(hint="", category=DiagnosisCategory.ORCHESTRATION_FAILURE, retrieval=None):
    return Diagnosis(
        category=category,
        rationale="r",
        prescription=Prescription(prompt_hint=hint, retrieval=retrieval),
    )


def test_diagnoses_are_persisted_and_queryable():
    store, path = _store()
    try:
        store.save_diagnosis("loop1", 0, _diagnosis("Use SI units."), "u",
                             prompt="convert 5 miles to km")
        store.save_diagnosis("loop2", 0, _diagnosis("Cite sources."), "u",
                             prompt="history of the roman senate")
        matches = store.similar_prior_diagnoses("convert 3 miles to km", "u", k=2)
        assert matches
        assert matches[0]["prompt"] == "convert 5 miles to km"
        assert matches[0]["diagnosis"].prescription.prompt_hint == "Use SI units."
        # Owner scoping: another user sees nothing.
        assert store.similar_prior_diagnoses("convert 3 miles to km", "someone-else", k=2) == []
    finally:
        os.remove(path)


def test_new_run_is_seeded_from_prior_diagnoses():
    store, path = _store()
    try:
        store.save_diagnosis(
            "old-loop", 0, _diagnosis("Always include worked examples."), None,
            prompt="explain the quadratic formula step by step",
        )
        adapter = FakeAdapter({
            "careful assistant": "answer",
            "rigorous CRITIC": _ACCEPT,
        })
        cfg = LoopConfig(prompt="explain the quadratic formula for beginners",
                         generator_model="gen", critic_model="crit",
                         use_preference_model=False, max_rounds=1)
        events = asyncio.run(_collect(cfg, adapter, store, PreferenceModel(store)))
        seeded = next(e for e in events if e["type"] == "experience_seeded")
        assert seeded["matches"] >= 1
        assert "Always include worked examples." in seeded["hints"]
        # The seeded hint reaches round 0's generator system prompt.
        system = adapter.generator_messages()[0][0]["content"]
        assert "Always include worked examples." in system
    finally:
        os.remove(path)


def test_engine_saves_diagnoses_rows():
    store, path = _store()
    try:
        adapter = FakeAdapter({
            "careful assistant": "answer",
            "rigorous CRITIC": json.dumps({
                "accepted": False, "summary": "Bad format.",
                "issues": ["did not follow the requested format"], "missing_context": [],
            }),
            "ANALYZER": json.dumps({
                "category": "orchestration_failure", "rationale": "fit",
                "next_prompt_hint": "Answer directly.",
            }),
        })
        cfg = LoopConfig(prompt="a prompt about widgets", generator_model="gen",
                         critic_model="crit", use_preference_model=False,
                         max_rounds=1, use_experience=False)
        asyncio.run(_collect(cfg, adapter, store, PreferenceModel(store)))
        matches = store.similar_prior_diagnoses("a prompt about widgets", None, k=1)
        assert matches and matches[0]["diagnosis"].category == DiagnosisCategory.ORCHESTRATION_FAILURE
    finally:
        os.remove(path)


def test_create_store_falls_back_to_sqlite(monkeypatch, tmp_path):
    # A postgres URL with no driver/database available must degrade, not fail.
    monkeypatch.setenv("SWT_DATABASE_URL", "postgresql://nobody@localhost:1/nope")
    import src.swt.store as store_mod
    monkeypatch.setattr(store_mod, "_default_db_path", lambda: str(tmp_path / "swt.db"))
    store = create_store()
    assert isinstance(store, SwtStore)


# -- preference model (D4) ----------------------------------------------------

def test_preference_cold_start_is_neutral():
    store, path = _store()
    try:
        pm = PreferenceModel(store)
        pred = pm.predict("u", "prompt", "some answer")
        assert pred.basis == "cold-start"
        assert pred.score == 0.5
    finally:
        os.remove(path)


def test_preference_learns_from_feedback_lexical():
    store, path = _store()
    try:
        pm = PreferenceModel(store)
        # Teach it: answers about apples accepted, answers about tanks rejected.
        pm.record("u", "q1", "apples are a healthy fruit rich in fiber", True)
        pm.record("u", "q2", "apples grow on trees and are crisp", True)
        pm.record("u", "q3", "military tanks use heavy armor plating", False)
        pm.record("u", "q4", "tanks fire large caliber shells in combat", False)
        good = pm.predict("u", "q", "fresh apples are a crisp healthy fruit")
        bad = pm.predict("u", "q", "the tank fired armor piercing shells")
        assert good.score > bad.score
        assert good.basis == "lexical"
    finally:
        os.remove(path)


def _fake_embed(text):
    """Deterministic 3-dim 'embedding': fruit axis, military axis, bias."""
    t = text.lower()
    fruit = float(t.count("apple") + t.count("fruit"))
    war = float(t.count("tank") + t.count("shell") + t.count("armor"))
    return [fruit, war, 0.5]


def test_preference_prototype_margin_with_embeddings():
    store, path = _store(embed_fn=_fake_embed)
    try:
        pm = PreferenceModel(store, embed_fn=_fake_embed)
        pm.record("u", "q1", "apples are a fruit", True)
        pm.record("u", "q2", "an apple a day", True)
        pm.record("u", "q3", "tanks have armor", False)
        pm.record("u", "q4", "the shell hit the tank", False)
        good = pm.predict("u", "q", "apple fruit salad")
        bad = pm.predict("u", "q", "tank armor shell")
        assert good.basis == "embedding"
        assert bad.basis == "embedding"
        assert good.score > 0.5 > bad.score
    finally:
        os.remove(path)


def test_feature_adjustment_tracks_answer_shape():
    accepted = ["short answer with code:\n```py\nx=1\n```", "```js\nlet a=2\n``` done"]
    rejected = ["a very long meandering answer " * 40, "another long one " * 60]
    with_code = "here you go:\n```py\nprint(1)\n```"
    delta, reasons = _feature_adjustment(with_code, accepted, rejected)
    assert delta > 0
    assert any("code-block" in r for r in reasons)
    long_prose = "let me explain at great length " * 50
    delta2, _ = _feature_adjustment(long_prose, accepted, rejected)
    assert delta2 < delta


def test_feedback_stats():
    store, path = _store()
    try:
        pm = PreferenceModel(store)
        pm.record("u", "q", "a", True)
        pm.record("u", "q", "b", False)
        stats = store.feedback_stats("u")
        assert stats == {"total": 2, "accepted": 1, "rejected": 1}
    finally:
        os.remove(path)


# -- analyzer (typed prescriptions) -------------------------------------------

def test_analyzer_heuristic_retrieval_vs_knowledge_gap():
    missing = Critique(accepted=False, summary="x", issues=[], missing_context=["the spec"])
    with_ctx = _heuristic(missing, had_context=True)
    without_ctx = _heuristic(missing, had_context=False)
    assert with_ctx.category == DiagnosisCategory.RETRIEVAL_FAILURE
    assert with_ctx.prescription.retrieval is not None
    assert with_ctx.prescription.retrieval.widen_k == WIDEN_K_STEP
    assert without_ctx.category == DiagnosisCategory.KNOWLEDGE_GAP
    assert "the spec" in without_ctx.prescription.write_to_kb


def test_analyzer_parses_typed_prescription():
    raw = json.dumps({
        "category": "retrieval_failure",
        "rationale": "Did not use the doc.",
        "next_prompt_hint": "Quote the doc.",
        "retrieval_query_rewrite": "doc section on auth",
        "switch_generator_to": "",
    })
    critique = Critique(accepted=False, summary="missed it", issues=["missed"])
    diag = asyncio.run(analyze(
        FakeAdapter({"ANALYZER": raw}), "m", "p", "a", critique, had_context=True,
    ))
    assert diag.category == DiagnosisCategory.RETRIEVAL_FAILURE
    assert diag.prescription.prompt_hint == "Quote the doc."
    assert diag.prescription.retrieval.query_rewrite == "doc section on auth"
    # The category's lever is the only one set.
    assert diag.prescription.switch_generator_to == ""
    assert diag.prescription.write_to_kb == ""


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


def test_condense_bounds_chunk_fanout():
    # A pathologically large context must not fan out into unbounded model
    # calls: the per-pass chunk cap limits how many completions run.
    from src.swt.recursive import _MAX_CHUNKS_PER_PASS

    class Counter:
        def __init__(self):
            self.calls = 0

        async def complete(self, model, messages, *, temperature=0.7, max_tokens=1024, owner=None):
            self.calls += 1
            return "relevant fact"

    huge = "\n\n".join(f"section {i} " * 200 for i in range(5000))
    counter = Counter()

    async def go():
        return await condense_context(
            counter, "q", huge, "m", budget_chars=6000, chunk_chars=6000, max_depth=1
        )

    asyncio.run(go())
    assert counter.calls <= _MAX_CHUNKS_PER_PASS


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
