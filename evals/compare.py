"""Compare two eval runs -- with the noise floor in the verdict.

The rule this module exists to enforce: **a delta smaller than the combined
run-to-run wobble of the two runs is not a result.** The pooled noise floor
is sqrt(std_a^2 + std_b^2); a delta is called real only when it clears twice
that. Crude (it's a ~2-sigma eyeball, not a t-test), but honest and cheap,
and it kills the "score moved 2%, ship it" failure mode where the 2% is pure
sampling noise.

Also reports the quality/speed trade explicitly (score delta vs latency
delta) and the per-task regressions, because an overall mean can hide "fixed
three easy tasks, broke the one that mattered".
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

# A per-task mean that drops by at least this much is flagged as a regression.
_TASK_REGRESSION = 0.5


def compare_runs(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> Dict[str, Any]:
    """Compare run_b (candidate) against run_a (baseline)."""
    sa, sb = run_a["summary"], run_b["summary"]
    delta = sb["overall_mean"] - sa["overall_mean"]
    noise = math.sqrt(sa["overall_std"] ** 2 + sb["overall_std"] ** 2)
    threshold = 2 * noise

    if noise == 0 and (run_a["repeats"] < 2 or run_b["repeats"] < 2):
        verdict = "unknown-noise"  # single-repeat runs have no measured wobble
    elif abs(delta) <= threshold:
        verdict = "within-noise"
    elif delta > 0:
        verdict = "better"
    else:
        verdict = "worse"

    ids_a = set(run_a["per_task"])
    ids_b = set(run_b["per_task"])
    shared = sorted(ids_a & ids_b)
    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    for tid in shared:
        d = run_b["per_task"][tid]["mean"] - run_a["per_task"][tid]["mean"]
        entry = {
            "task_id": tid,
            "baseline": run_a["per_task"][tid]["mean"],
            "candidate": run_b["per_task"][tid]["mean"],
            "delta": round(d, 4),
        }
        if d <= -_TASK_REGRESSION:
            regressions.append(entry)
        elif d >= _TASK_REGRESSION:
            improvements.append(entry)

    return {
        "baseline": {"label": run_a.get("label", "?"), "target": run_a["target"],
                     "score": sa["overall_mean"], "std": sa["overall_std"],
                     "latency_p50_s": sa["latency_p50_s"]},
        "candidate": {"label": run_b.get("label", "?"), "target": run_b["target"],
                      "score": sb["overall_mean"], "std": sb["overall_std"],
                      "latency_p50_s": sb["latency_p50_s"]},
        "delta": round(delta, 4),
        "noise_floor": round(noise, 4),
        "threshold": round(threshold, 4),
        "verdict": verdict,
        "latency_delta_p50_s": round(sb["latency_p50_s"] - sa["latency_p50_s"], 3),
        "tasks_only_in_baseline": sorted(ids_a - ids_b),
        "tasks_only_in_candidate": sorted(ids_b - ids_a),
        "regressions": sorted(regressions, key=lambda e: e["delta"]),
        "improvements": sorted(improvements, key=lambda e: -e["delta"]),
    }


def format_comparison(cmp: Dict[str, Any]) -> str:
    a, b = cmp["baseline"], cmp["candidate"]
    lines = [
        f"baseline     {a['label']}  [{a['target']}]  "
        f"{a['score']:.3f} ± {a['std']:.3f}  (p50 {a['latency_p50_s']}s)",
        f"candidate    {b['label']}  [{b['target']}]  "
        f"{b['score']:.3f} ± {b['std']:.3f}  (p50 {b['latency_p50_s']}s)",
        f"delta        {cmp['delta']:+.3f}  vs noise floor ±{cmp['threshold']:.3f} "
        f"(pooled std {cmp['noise_floor']:.3f})",
        f"latency      {cmp['latency_delta_p50_s']:+.3f}s p50",
        f"verdict      {cmp['verdict'].upper()}",
    ]
    if cmp["verdict"] == "within-noise":
        lines.append("             (the score moved, but not more than the runs "
                     "wobble on their own -- do not ship a decision on this)")
    if cmp["verdict"] == "unknown-noise":
        lines.append("             (a single-repeat run has no measured wobble -- "
                     "rerun with --repeats 3+ before trusting any delta)")
    if cmp["regressions"]:
        lines.append("regressions  " + "; ".join(
            f"{e['task_id']} {e['baseline']:.2f}->{e['candidate']:.2f}"
            for e in cmp["regressions"][:10]))
    if cmp["improvements"]:
        lines.append("improved     " + "; ".join(
            f"{e['task_id']} {e['baseline']:.2f}->{e['candidate']:.2f}"
            for e in cmp["improvements"][:10]))
    if cmp["tasks_only_in_baseline"] or cmp["tasks_only_in_candidate"]:
        lines.append("WARNING      task sets differ -- comparison only covers the "
                     "shared subset; treat with suspicion")
    return "\n".join(lines)
