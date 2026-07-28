"""Runner / compare / task-loading / mining tests for the eval harness.

Everything offline: scripted fake targets stand in for models, exactly like
the SWT fake-adapter pattern.
"""
import asyncio
import json
import os
import tempfile

import pytest

from evals.compare import compare_runs
from evals.runner import load_run, run_eval, save_run
from evals.targets import SelfTestTarget, TargetResult, make_target
from evals.tasks import Task, load_tasks

CORE_TASKS = "evals/tasks/core.jsonl"


class ScriptedTarget:
    """Answers from a {task_id: output-or-callable} script; default wrong."""

    name = "scripted"

    def __init__(self, script=None, latency_s=0.0):
        self.script = script or {}
        self.latency_s = latency_s
        self.calls = 0

    async def complete(self, task):
        self.calls += 1
        entry = self.script.get(task.id, "wrong answer, zero points")
        text = entry(self.calls) if callable(entry) else entry
        return TargetResult(text=text, extra={"completion_tokens": max(1, len(text) // 4)})


def _tasks():
    return [
        Task(id="t1", kind="exact", prompt="p1", expect={"value": "alpha"}),
        Task(id="t2", kind="exact", prompt="p2", expect={"value": "beta"}),
        Task(id="t3", kind="numeric", prompt="p3", expect={"value": 42, "tol": 0}),
        Task(id="t4", kind="choice", prompt="p4",
             expect={"choices": ["yes", "no"], "answer": "yes"}),
    ]


def _run(tasks, target, repeats=3):
    return asyncio.run(run_eval(tasks, target, repeats=repeats))


# -- runner math ---------------------------------------------------------------

def test_perfect_run_scores_one_with_zero_std():
    target = ScriptedTarget({"t1": "alpha", "t2": "beta", "t3": "42", "t4": "yes"})
    results = _run(_tasks(), target)
    s = results["summary"]
    assert s["overall_mean"] == 1.0
    assert s["overall_std"] == 0.0
    assert s["per_repeat"] == [1.0, 1.0, 1.0]
    assert s["errors"] == 0
    assert target.calls == 12  # 4 tasks x 3 repeats
    assert s["tokens_per_s_mean"] is None or s["tokens_per_s_mean"] > 0

def test_half_right_run_scores_half():
    target = ScriptedTarget({"t1": "alpha", "t3": "42"})
    results = _run(_tasks(), target)
    assert results["summary"]["overall_mean"] == 0.5
    assert results["per_task"]["t2"]["mean"] == 0.0
    assert results["per_task"]["t1"]["mean"] == 1.0

def test_flaky_target_produces_nonzero_std():
    # t1 correct only on odd calls -> per-repeat overalls differ -> std > 0.
    flip = {"n": 0}

    def flaky(_calls):
        flip["n"] += 1
        return "alpha" if flip["n"] % 2 else "nope"

    target = ScriptedTarget({"t1": flaky, "t2": "beta", "t3": "42", "t4": "yes"})
    results = _run(_tasks(), target, repeats=4)
    s = results["summary"]
    assert 0.75 <= s["overall_mean"] <= 1.0
    assert s["overall_std"] > 0.0

def test_target_exception_scores_zero_and_counts_error():
    class Boom:
        name = "boom"

        async def complete(self, task):
            if task.id == "t2":
                raise RuntimeError("endpoint down")
            return TargetResult(text="alpha")

    tasks = _tasks()[:2]
    results = _run(tasks, Boom(), repeats=2)
    assert results["summary"]["errors"] == 2
    assert results["per_task"]["t2"]["mean"] == 0.0
    assert results["per_task"]["t1"]["mean"] == 1.0  # t1 'alpha' passes

def test_single_repeat_has_zero_std_not_crash():
    target = ScriptedTarget({"t1": "alpha", "t2": "beta", "t3": "42", "t4": "yes"})
    results = _run(_tasks(), target, repeats=1)
    assert results["summary"]["overall_std"] == 0.0
    assert results["repeats"] == 1


# -- save / load / compare -------------------------------------------------------

def _saved_pair(score_script_a, score_script_b, repeats=3):
    tasks = _tasks()
    run_a = _run(tasks, ScriptedTarget(score_script_a), repeats=repeats)
    run_b = _run(tasks, ScriptedTarget(score_script_b), repeats=repeats)
    with tempfile.TemporaryDirectory() as tmp:
        pa = save_run(run_a, "baseline", out_dir=tmp)
        pb = save_run(run_b, "candidate", out_dir=tmp)
        return load_run(pa), load_run(pb)


def test_compare_better_beyond_noise():
    a, b = _saved_pair({"t1": "alpha"},  # 0.25
                       {"t1": "alpha", "t2": "beta", "t3": "42", "t4": "yes"})  # 1.0
    cmp = compare_runs(a, b)
    assert cmp["verdict"] == "better"
    assert cmp["delta"] == pytest.approx(0.75)
    assert [e["task_id"] for e in cmp["improvements"]] == ["t2", "t3", "t4"]

def test_compare_identical_runs_within_or_unknown_noise():
    script = {"t1": "alpha", "t2": "beta", "t3": "42", "t4": "yes"}
    a, b = _saved_pair(script, script)
    cmp = compare_runs(a, b)
    # Deterministic identical runs: delta 0, std 0 -> within-noise at 3 repeats.
    assert cmp["verdict"] == "within-noise"
    assert cmp["delta"] == 0.0
    assert not cmp["regressions"] and not cmp["improvements"]

def test_compare_single_repeat_is_unknown_noise():
    a, b = _saved_pair({"t1": "alpha"}, {}, repeats=1)
    cmp = compare_runs(a, b)
    assert cmp["verdict"] == "unknown-noise"

def test_compare_flags_task_regressions():
    a, b = _saved_pair({"t1": "alpha", "t2": "beta", "t3": "42", "t4": "yes"},
                       {"t1": "alpha", "t2": "beta", "t3": "42"})
    cmp = compare_runs(a, b)
    assert [e["task_id"] for e in cmp["regressions"]] == ["t4"]

def test_save_run_writes_label_and_loads_back():
    results = _run(_tasks()[:1], ScriptedTarget({"t1": "alpha"}), repeats=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_run(results, "my label/with weird chars!", out_dir=tmp)
        loaded = load_run(path)
        assert loaded["label"] == "my label/with weird chars!"
        assert os.path.basename(path).endswith(".json")


# -- task loading -----------------------------------------------------------------

def test_core_task_file_loads_and_selftest_passes():
    tasks = load_tasks(CORE_TASKS)
    assert len(tasks) == 30
    results = _run(tasks, SelfTestTarget(), repeats=1)
    assert results["summary"]["overall_mean"] == 1.0, (
        "core.jsonl has a rubric even a known-good answer cannot pass: "
        + str([tid for tid, t in results["per_task"].items() if t["mean"] < 1.0])
    )

def _write_tasks(tmp, lines):
    path = os.path.join(tmp, "tasks.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path

def test_load_rejects_todo_kind():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_tasks(tmp, [json.dumps(
            {"id": "m1", "kind": "todo", "prompt": "p", "expect": {}})])
        with pytest.raises(ValueError, match="rubric"):
            load_tasks(path)

def test_load_rejects_duplicate_ids_and_unknown_kind():
    with tempfile.TemporaryDirectory() as tmp:
        dup = _write_tasks(tmp, [
            json.dumps({"id": "a", "kind": "exact", "prompt": "p", "expect": {"value": "x"}}),
            json.dumps({"id": "a", "kind": "exact", "prompt": "p", "expect": {"value": "x"}}),
        ])
        with pytest.raises(ValueError, match="duplicate"):
            load_tasks(dup)
    with tempfile.TemporaryDirectory() as tmp:
        bad = _write_tasks(tmp, [json.dumps(
            {"id": "b", "kind": "vibes", "prompt": "p", "expect": {}})])
        with pytest.raises(ValueError, match="unknown kind"):
            load_tasks(bad)


# -- target spec parsing ------------------------------------------------------------

def test_make_target_specs():
    assert make_target("selftest").name == "selftest"
    t = make_target("model:qwen2.5:7b@gpu-node")
    assert t.model == "qwen2.5:7b@gpu-node"  # ':' and '@' survive in the spec
    h = make_target("http:http://127.0.0.1:8080::qwen-q4")
    assert h.url == "http://127.0.0.1:8080/v1/chat/completions"
    assert h.model == "qwen-q4"
    s = make_target("swt:gen:8b::critic:7b")
    assert s.generator == "gen:8b" and s.critic == "critic:7b"

def test_make_target_rejects_garbage():
    for bad in ("", "ftp:x", "http:only-url", "swt:only-gen"):
        with pytest.raises(ValueError):
            make_target(bad)


# -- mining -------------------------------------------------------------------------

def test_mine_pulls_prompts_from_swt_store(capsys):
    from src.swt.schemas import LoopResult
    from src.swt.store import SwtStore
    from evals.__main__ import main

    fd, db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        store = SwtStore(db_path=db)
        store.save_loop(LoopResult(loop_id="l1", prompt="how do I rack the UPS?"), None)
        store.save_loop(LoopResult(loop_id="l2", prompt="summarize the invoice email"), None)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "mined.jsonl")
            rc = main(["mine", "--db", db, "--limit", "10", "--out", out])
            assert rc == 0
            body = open(out, encoding="utf-8").read()
            assert "how do I rack the UPS?" in body
            assert '"kind": "todo"' in body
            # And the loader refuses to run them until rubrics are written.
            with pytest.raises(ValueError, match="rubric"):
                load_tasks(out)
    finally:
        os.remove(db)
