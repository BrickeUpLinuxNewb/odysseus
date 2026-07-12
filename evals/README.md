# Project Inari eval harness

The ruler. Every efficiency or reliability change (quantization tier, DSPy
module, loop tweak, model swap, new node) gets measured against this before
it gets believed.

## Design decisions (and why)

- **Programmatic scoring only.** Every task kind is machine-checkable —
  label choice, exact value, substring, regex, numeric tolerance, JSON shape.
  There is deliberately **no LLM judge**: on this cluster the judge would be
  a small local model, i.e. the same rubber the ruler is supposed to measure.
  Free-form generation quality is out of scope for v1; the harness measures
  the constrained jobs (routing, extraction, structured output, closed QA)
  where a number can be trusted.
- **Variance is part of the answer.** Default 3 repeats; each repeat is a
  full pass over the task set, so the run reports `mean ± std`. That std is
  the noise floor. `compare` refuses to call a delta real unless it clears
  **2× the pooled noise floor** — "score moved 2%" means nothing when the
  runs wobble 3% on their own.
- **Quality is not the only axis.** Every call records latency (p50/p95) and
  throughput (true tokens/sec when the backend reports usage, chars/sec
  otherwise). A quantization sweep is a quality-vs-speed trade; the harness
  shows both sides. **VRAM you record by hand per config** — it can't be
  measured portably from here, so note it next to the run label.
- **Cheap to run.** 30 tasks × 3 repeats, sequential. If a full run doesn't
  fit in a coffee break on Node 1, shrink the task set before you shrink the
  repeats — repeats are what make the noise floor real.

## Quickstart (Node 1)

```bash
# 1. Lint the task file (selftest answers from the rubric itself;
#    anything under 1.000 is a broken rubric, not a model):
python -m evals run --tasks evals/tasks/core.jsonl --target selftest --label lint

# 2. Baseline the model the system actually uses (Odysseus resolver path,
#    so model@endpoint works):
python -m evals run --tasks evals/tasks/core.jsonl \
    --target model:hermes3:8b --repeats 3 --label hermes8b-baseline

# 3. Change ONE thing (e.g. a different GGUF quantization served by
#    llama.cpp), run again against the raw endpoint:
python -m evals run --tasks evals/tasks/core.jsonl \
    --target http:http://127.0.0.1:8080::qwen2.5-7b-q4_k_m \
    --repeats 3 --label qwen7b-q4km

# 4. Compare candidate vs baseline — the verdict includes the noise floor:
python -m evals compare data/evals/<baseline>.json data/evals/<candidate>.json
```

Results land in `data/evals/` (gitignored) as timestamped JSON.

## Targets

| Spec | What it measures |
| --- | --- |
| `selftest` | The task file itself. Run after every rubric edit. |
| `model:SPEC` | One-shot through Odysseus's resolver (`name` or `name@endpoint`). |
| `http:BASE_URL::MODEL_ID` | A raw OpenAI-compatible endpoint (llama.cpp server, vLLM, Ollama). The Track A path: two llama.cpp servers with different quantizations of the same model = one measured sweep. Reports true tokens/sec. |
| `swt:GEN::CRIT[::ANALYZER]` | The full SWT loop as a black box. Extra metrics carry mean rounds + convergence rate. |

**The B4 experiment is two commands:** run `model:GEN` and `swt:GEN::CRIT`
on the same tasks, compare. If the loop's score isn't above the one-shot's
noise floor, the loop is adding latency, not quality — now it's a measurement
instead of a suspicion.

## Task format

JSONL, one task per line, `#` comments allowed. The rubric (`expect`) is
written **in advance** — that's the blind part of the protocol.

```json
{"id": "route-001", "kind": "choice", "tags": ["routing"],
 "system": "Answer with exactly one word from: email, code, calendar, search, notes.",
 "prompt": "Reply to Sarah's message about the Q3 invoice",
 "expect": {"choices": ["email", "code", "calendar", "search", "notes"], "answer": "email"}}
```

Kinds: `choice`, `exact`, `contains`, `contains_all` (partial credit),
`regex` (add `sample_pass` so selftest can lint it), `numeric` (`tol` /
`rel_tol`; lenient to restated operands, `"strict": true` to demand exactly
one number), `json` (`required` / `types` / `enum` / `values`).

## Growing the task set from real usage

```bash
python -m evals mine --limit 50 --out evals/tasks/mined.todo.jsonl
```

pulls distinct prompts from the SWT experience store, newest first, as
`kind: "todo"` entries. The loader **refuses to run** `todo` tasks: mining
supplies real workload, but the grading standard must still be written by a
human, per prompt, before the prompt becomes a task. Keep `core.jsonl` as the
stable spine; add mined tasks to a second file and run both.

The starter set (30 tasks: 10 routing, 10 extraction, 5 JSON, 5 closed QA) is
placeholder-but-real — it exercises the job shapes the system does, but it
was written from imagination. Expect to replace most of it with mined tasks
within a few weeks of real use. **Do not tune prompts against the eval set**
— when a task leaks into prompt engineering, retire it to a regression file
and mine a fresh one.

## Reading a comparison

```
delta        +0.033  vs noise floor ±0.058 (pooled std 0.029)
verdict      WITHIN-NOISE
```

- `BETTER` / `WORSE` — the delta cleared 2× the pooled std. Act on it.
- `WITHIN-NOISE` — the score moved less than the runs wobble on their own.
  Do not ship a decision on this; either the change doesn't matter or the
  task set is too small to see it.
- `UNKNOWN-NOISE` — a run had fewer than 2 repeats, so there is no measured
  wobble. Rerun with `--repeats 3`.

Per-task regressions are listed separately because a flat mean can hide
"fixed three easy tasks, broke the one that mattered."

## Stage-1 exit criteria (from the roadmap, so it's written down)

Stage 1 is **done** when: (1) selftest scores 1.000 on the task file,
(2) a baseline run of the current production model exists in `data/evals/`
with ≥3 repeats, and (3) the baseline's `overall_std` is below ~0.05 — if
the wobble is bigger than that, the task set needs more/tighter tasks before
any Track A/B comparison will be readable. Then move to the quantization
sweep (A1), one variable at a time.
