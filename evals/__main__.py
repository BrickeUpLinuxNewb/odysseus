"""CLI for the eval harness.

    python -m evals run --tasks evals/tasks/core.jsonl --target selftest --label lint
    python -m evals run --tasks evals/tasks/core.jsonl \
        --target http:http://127.0.0.1:8080::qwen2.5-7b-q4_k_m \
        --repeats 3 --label q4km-baseline
    python -m evals compare data/evals/<baseline>.json data/evals/<candidate>.json
    python -m evals mine --limit 50 --out evals/tasks/mined.todo.jsonl

Run ``--target selftest`` on every new/edited task file before trusting it:
selftest answers each task from its own rubric, so anything under 1.0 is a
broken rubric, not a model.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from evals.compare import compare_runs, format_comparison
from evals.runner import format_summary, load_run, run_eval, save_run
from evals.targets import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, make_target
from evals.tasks import load_tasks


def _cmd_run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    target = make_target(args.target, temperature=args.temperature,
                         max_tokens=args.max_tokens)
    results = asyncio.run(run_eval(
        tasks, target, repeats=args.repeats, concurrency=args.concurrency,
    ))
    path = save_run(results, args.label, out_dir=args.out)
    print(format_summary(results))
    print(f"saved        {path}")
    if args.target == "selftest" and results["summary"]["overall_mean"] < 1.0:
        print("SELFTEST FAILED: some rubrics cannot be passed even by a "
              "known-good answer. Fix the tasks listed above before running "
              "a real target.", file=sys.stderr)
        return 1
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    cmp = compare_runs(load_run(args.baseline), load_run(args.candidate))
    print(format_comparison(cmp))
    if args.json:
        print(json.dumps(cmp, indent=2))
    return 0


def _cmd_mine(args: argparse.Namespace) -> int:
    """Pull real prompts out of the SWT experience store as task candidates.

    Emits ``kind: todo`` lines on purpose: the loader refuses to run them
    until a rubric is written. Mining gives you real workload; it cannot give
    you the blind grading standard -- that part stays manual by design.
    """
    from src.swt.store import SwtStore

    store = SwtStore(db_path=args.db) if args.db else SwtStore()
    prompts: list = []
    seen: set = set()
    for row in store.list_loops(args.owner, limit=args.limit * 3):
        p = (row.get("prompt") or "").strip()
        if p and p not in seen:
            seen.add(p)
            prompts.append(p)
        if len(prompts) >= args.limit:
            break
    lines = [
        json.dumps({
            "id": f"mined-{i:03d}",
            "kind": "todo",
            "tags": ["mined"],
            "prompt": p,
            "expect": {},
        }, ensure_ascii=False)
        for i, p in enumerate(prompts, start=1)
    ]
    body = "\n".join(
        ["# Mined from the SWT store. For each keeper: decide what a good",
         "# answer looks like, set a real kind + expect rubric, move it into",
         "# a real task file. Delete the rest. 'todo' tasks will not run."]
        + lines
    ) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(body)
        print(f"wrote {len(prompts)} candidate prompts to {args.out}")
    else:
        print(body, end="")
    if not prompts:
        print("no prompts found -- the SWT store is empty. Use the system "
              "for real work first; the eval set should come from real "
              "workload, not imagination.", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="python -m evals",
                                     description="Project Inari eval harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run a task file against a target")
    run_p.add_argument("--tasks", required=True, help="path to a .jsonl task file")
    run_p.add_argument("--target", required=True,
                       help="selftest | model:SPEC | http:URL::MODEL | swt:GEN::CRIT[::ANALYZER]")
    run_p.add_argument("--label", required=True,
                       help="name this configuration (e.g. 'qwen7b-q4km-baseline')")
    run_p.add_argument("--repeats", type=int, default=3,
                       help="passes over the task set; 3+ needed for a noise floor (default 3)")
    run_p.add_argument("--concurrency", type=int, default=1,
                       help="parallel calls (default 1 -- constrained hardware)")
    run_p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    run_p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run_p.add_argument("--out", default="data/evals", help="results directory")
    run_p.set_defaults(fn=_cmd_run)

    cmp_p = sub.add_parser("compare", help="compare candidate run vs baseline run")
    cmp_p.add_argument("baseline")
    cmp_p.add_argument("candidate")
    cmp_p.add_argument("--json", action="store_true", help="also print machine-readable diff")
    cmp_p.set_defaults(fn=_cmd_compare)

    mine_p = sub.add_parser("mine", help="pull real prompts from the SWT store as task candidates")
    mine_p.add_argument("--limit", type=int, default=50)
    mine_p.add_argument("--owner", default=None)
    mine_p.add_argument("--db", default=None, help="path to swt.db (default: app data dir)")
    mine_p.add_argument("--out", default=None, help="write JSONL here instead of stdout")
    mine_p.set_defaults(fn=_cmd_mine)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
