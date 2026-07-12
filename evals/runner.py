"""The eval runner: tasks x repeats -> score, noise floor, latency.

The variance protocol: every task runs ``repeats`` times. Repeat r's overall
score is the mean across tasks of that repeat's scores, so the run yields
``repeats`` independent overall scores; the summary reports their mean and
sample standard deviation. That std IS the noise floor -- ``compare`` uses it
to refuse to call an improvement real when it is inside run-to-run wobble.

Latency and throughput are recorded per call (wall-clock, chars/sec, plus
true completion tokens/sec when the backend reports usage), because Track A
decisions trade quality against speed and a quality-only ruler cannot see
the trade.

Failures are data: a target exception scores 0.0 with the error in
``detail`` and increments ``errors``. A model that times out on hard tasks
should lose points for it, not have those tasks quietly dropped.
"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from evals.checkers import check
from evals.tasks import Task
from evals.targets import Target

logger = logging.getLogger(__name__)

_EXCERPT_CHARS = 400


@dataclass
class CallResult:
    task_id: str
    kind: str
    repeat: int
    score: float
    detail: str
    latency_s: float
    output_chars: int
    output_excerpt: str
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "kind": self.kind, "repeat": self.repeat,
            "score": self.score, "detail": self.detail,
            "latency_s": round(self.latency_s, 4),
            "output_chars": self.output_chars,
            "output_excerpt": self.output_excerpt,
            "error": self.error, "extra": self.extra,
        }


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(p * (len(sorted_values) - 1))))
    return sorted_values[idx]


async def _run_one(target: Target, task: Task, repeat: int,
                   semaphore: asyncio.Semaphore) -> CallResult:
    async with semaphore:
        start = time.perf_counter()
        try:
            result = await target.complete(task)
            latency = time.perf_counter() - start
        except Exception as exc:
            latency = time.perf_counter() - start
            logger.warning("eval call failed on %s (repeat %d): %s", task.id, repeat, exc)
            return CallResult(
                task_id=task.id, kind=task.kind, repeat=repeat,
                score=0.0, detail=f"target error: {str(exc)[:200]}",
                latency_s=latency, output_chars=0, output_excerpt="",
                error=str(exc)[:200],
            )
    score, detail = check(task.kind, result.text, task.expect)
    return CallResult(
        task_id=task.id, kind=task.kind, repeat=repeat,
        score=score, detail=detail, latency_s=latency,
        output_chars=len(result.text),
        output_excerpt=result.text[:_EXCERPT_CHARS],
        extra=result.extra,
    )


async def run_eval(tasks: List[Task], target: Target, *, repeats: int = 3,
                   concurrency: int = 1) -> Dict[str, Any]:
    """Run the full grid and return the results document (see summarize)."""
    repeats = max(1, repeats)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    calls: List[CallResult] = []
    # Repeat-major order so repeat r is a complete pass over the task set --
    # that is what makes per-repeat overall scores meaningful.
    for repeat in range(repeats):
        batch = await asyncio.gather(*(
            _run_one(target, task, repeat, semaphore) for task in tasks
        ))
        calls.extend(batch)
        done = sum(1 for c in batch if not c.error)
        logger.info("repeat %d/%d: %d/%d calls ok", repeat + 1, repeats, done, len(batch))
    return summarize(tasks, target.name, repeats, calls)


def summarize(tasks: List[Task], target_name: str, repeats: int,
              calls: List[CallResult]) -> Dict[str, Any]:
    by_repeat: Dict[int, List[CallResult]] = {}
    by_task: Dict[str, List[CallResult]] = {}
    for c in calls:
        by_repeat.setdefault(c.repeat, []).append(c)
        by_task.setdefault(c.task_id, []).append(c)

    per_repeat_scores = [
        statistics.mean(c.score for c in group) if group else 0.0
        for _, group in sorted(by_repeat.items())
    ]
    overall_mean = statistics.mean(per_repeat_scores) if per_repeat_scores else 0.0
    overall_std = statistics.stdev(per_repeat_scores) if len(per_repeat_scores) >= 2 else 0.0

    latencies = sorted(c.latency_s for c in calls)
    chars_per_s = [c.output_chars / c.latency_s for c in calls
                   if c.latency_s > 0 and c.output_chars > 0]

    # Aggregate any numeric extras targets reported (rounds, converged,
    # completion_tokens...) so e.g. the SWT target's round count shows up.
    extra_means: Dict[str, float] = {}
    extra_values: Dict[str, List[float]] = {}
    for c in calls:
        for key, value in c.extra.items():
            if isinstance(value, bool):
                value = 1 if value else 0
            if isinstance(value, (int, float)):
                extra_values.setdefault(key, []).append(float(value))
    for key, values in extra_values.items():
        extra_means[key] = round(statistics.mean(values), 4)
    token_latencies = [
        c.extra["completion_tokens"] / c.latency_s for c in calls
        if isinstance(c.extra.get("completion_tokens"), int) and c.latency_s > 0
    ]

    per_task = {
        task.id: {
            "kind": task.kind,
            "tags": task.tags,
            "scores": [c.score for c in sorted(by_task.get(task.id, []), key=lambda c: c.repeat)],
            "mean": round(statistics.mean([c.score for c in by_task[task.id]]), 4)
            if by_task.get(task.id) else 0.0,
            "details": [c.detail for c in sorted(by_task.get(task.id, []), key=lambda c: c.repeat)],
        }
        for task in tasks
    }

    return {
        "target": target_name,
        "repeats": repeats,
        "task_count": len(tasks),
        "created_at": time.time(),
        "summary": {
            "overall_mean": round(overall_mean, 4),
            "overall_std": round(overall_std, 4),
            "per_repeat": [round(s, 4) for s in per_repeat_scores],
            "errors": sum(1 for c in calls if c.error),
            "latency_p50_s": round(_percentile(latencies, 0.50), 3),
            "latency_p95_s": round(_percentile(latencies, 0.95), 3),
            "latency_total_s": round(sum(latencies), 1),
            "chars_per_s_mean": round(statistics.mean(chars_per_s), 1) if chars_per_s else None,
            "tokens_per_s_mean": round(statistics.mean(token_latencies), 1) if token_latencies else None,
            "extra_means": extra_means,
        },
        "per_task": per_task,
        "calls": [c.to_dict() for c in calls],
    }


def save_run(results: Dict[str, Any], label: str, out_dir: str = "data/evals") -> str:
    """Persist a run as JSON; filename carries label + timestamp."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in label) or "run"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out / f"{stamp}-{safe_label}.json"
    results = dict(results)
    results["label"] = label
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(path)


def load_run(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_summary(results: Dict[str, Any]) -> str:
    s = results["summary"]
    lines = [
        f"target       {results['target']}",
        f"tasks        {results['task_count']}  x{results['repeats']} repeats"
        f"  ({s['errors']} errors)",
        f"score        {s['overall_mean']:.3f} ± {s['overall_std']:.3f}"
        f"   per-repeat {s['per_repeat']}",
        f"latency      p50 {s['latency_p50_s']}s  p95 {s['latency_p95_s']}s"
        f"  total {s['latency_total_s']}s",
    ]
    if s.get("tokens_per_s_mean") is not None:
        lines.append(f"throughput   {s['tokens_per_s_mean']} tok/s")
    elif s.get("chars_per_s_mean") is not None:
        lines.append(f"throughput   {s['chars_per_s_mean']} chars/s (no usage reported)")
    if s.get("extra_means"):
        lines.append(f"extras       {s['extra_means']}")
    failing = [
        (tid, t) for tid, t in results["per_task"].items() if t["mean"] < 1.0
    ]
    if failing:
        lines.append("weak tasks   " + ", ".join(
            f"{tid}({t['mean']:.2f})" for tid, t in
            sorted(failing, key=lambda x: x[1]["mean"])[:10]
        ))
    return "\n".join(lines)
