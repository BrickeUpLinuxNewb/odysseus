# Odysseus — Full Session Summary & Recommendations

**Fork:** `BrickeUpLinuxNewb/odysseus` &nbsp;·&nbsp; **Branch:** `claude/refinement-n1ibom` &nbsp;·&nbsp; **PR:** [#1](https://github.com/BrickeUpLinuxNewb/odysseus/pull/1)
**Date:** 2026-07-12

This document is the complete record of what was built, reviewed, and recommended across this session. It is organized as:

1. [Everything that was done](#1-everything-that-was-done)
2. [Commit-by-commit ledger](#2-commit-by-commit-ledger)
3. [Security & secret-leak review results](#3-security--secret-leak-review-results)
4. [Recommendations](#4-recommendations)
5. [What to do next (operator checklist)](#5-what-to-do-next-operator-checklist)

---

## 1. Everything that was done

### 1.1 SWT Loop refinement (commit `b0426fd`)

The SWT ("Save The World") Loop is Odysseus's automated **generator → critic → analyzer** loop. It began as a working-but-shallow single-box demo; this session deepened it into the connected system it was designed to be. Seven work items (D1–D7):

- **D1 — Retrieval wired in.** New `src/swt/retrieval.py` adds a `Retriever` / `KnowledgeWriter` seam with a `RagRetriever` over the app's shared VectorRAG/ChromaDB instance. Top-k knowledge-base chunks now feed the generator alongside pasted context, and a `retrieval_failure` diagnosis actually re-runs retrieval next round with a widened `k` and/or rewritten query (hard-capped at k=12).
- **D2 — Pluggable experience store.** Persistence sits behind an `ExperienceStore` protocol. SQLite (`data/swt.db`) stays the zero-config default; a new `PostgresExperienceStore` (selected by `SWT_DATABASE_URL`, lazy `psycopg`) keeps its tables in a dedicated `swt_main` schema. Diagnoses are stored as first-class rows with prompt embeddings.
- **D3 — Cross-run learning.** A new run queries prior diagnoses on similar prompts (cosine over stored vectors, lexical fallback) and seeds their hints and retrieval adjustments before round 0. `knowledge_gap` diagnoses write the gap back into the knowledge base (tagged `source: swt`) so a future run retrieves it.
- **D4 — Preference model.** The oversold "TRIBE v2 cognitive model" was renamed and rebuilt honestly as `PreferenceModel` (`src/swt/preference.py`). It uses real embeddings — a prototype margin between accepted/rejected answer centroids — plus interaction-shape features (length, code blocks, hedging), with the lexical vote and cold-start deferral kept as fallbacks. `CognitiveModel` remains a deprecated alias.
- **D5 — Per-role endpoints.** Generator / critic / analyzer / a new **librarian** role each take a full model spec, so `model@endpoint` routes roles to different nodes in a distributed deployment. Bare names keep single-box behavior. This reused the existing owner-scoped resolver rather than adding a new credential surface.
- **D6 — Typed prescriptions.** The analyzer's loose `next_prompt_hint` / `retrieval_adjustment` / `switch_generator_to` strings became a typed `Prescription` dataclass the engine dispatches on — one concrete cure per diagnosis category.
- **D7 — Polish.** Docs rewritten with an honest "not yet benchmarked" caveat, overselling comments removed, routes expose the new toggles, `/status` reports capabilities. Tests grew from 12 to 23, still fully offline via fakes. The full repo suite (4,468 tests) stays green.

**Files:** new `src/swt/retrieval.py`, `src/swt/preference.py`, `src/swt/store_pg.py`, `src/swt/similarity.py`; rewritten `src/swt/{engine,analyzer,store,schemas,models,__init__}.py`, `routes/swt_routes.py`, `docs/swt.md`; `cognitive_model.py` reduced to a back-compat shim.

### 1.2 Critique report (commit `f544fda`)

`SWT_REFINEMENT_REPORT.md` — a standalone document capturing the full change with a section of **14 known limitations** and a section of **falsifiable claims**, written specifically to be fed to a critique loop.

### 1.3 Project Inari eval harness (commit `af3f34c`)

A test rig for non-deterministic model output — the "ruler" every efficiency (Track A) and reliability (Track B) change is measured against before it is believed. New `evals/` package with a CLI: `python -m evals {run, compare, mine}`.

Design decisions baked in (each from the roadmap review):

- **Programmatic scoring only** — seven checker kinds (choice / exact / contains / contains_all / regex / numeric / json). No LLM judge, because on this cluster the judge would be the same rubber it is measuring. Free-form generation quality is out of scope for v1.
- **Variance is part of the answer** — every task runs N times; the summary reports `mean ± std`, and `compare` refuses to call a delta real unless it clears **2× the pooled noise floor**. It also flags per-task regressions a flat mean would hide.
- **Quality/speed recorded together** — per-call latency (p50/p95) and throughput (true tokens/sec when the backend reports usage). VRAM stays operator-recorded per config.
- **Four targets** — `selftest` (answers from the rubric itself: the task-file linter), `model:SPEC` (Odysseus resolver path), `http:URL::MODEL` (raw llama.cpp/vLLM endpoint — the quantization-sweep path), and `swt:GEN::CRIT` (the whole loop as a black box, making the "does looping help?" question a two-command experiment).
- **`mine`** pulls real prompts from the SWT store as `kind: todo` candidates; the loader refuses to run them until a rubric is written — mining supplies workload, never the grading standard.

**Starter set:** 30 tasks (10 routing, 10 extraction, 5 JSON, 5 closed QA), selftest-linted to 1.000. 41 offline tests, no model required.

**Demonstration (in-sandbox, simulated models — the numbers below are from a simulator, not real quantizations):**

```
selftest lint             score 1.000 ± 0.000   (all rubrics passable)
sim-q4 baseline           score 0.667 ± 0.000
sim-q4 rerun              score 0.644 ± 0.019
sim-q5 candidate          score 0.922 ± 0.051

compare q4-baseline vs q4-rerun   delta -0.022  ->  WITHIN-NOISE  (correctly refused)
compare q4-baseline vs q5         delta +0.256  ->  BETTER + flagged one regression
```

### 1.4 Session handoff + cross-platform setup manual (commit `b4e9547`)

`SESSION_HANDOFF.md` (also rendered to a downloadable PDF) documents the session and gives a full step-by-step setup manual for running the fork on **Windows 11 → Fedora → Linux Mint**, with both Docker and native paths, a first-login checklist, ports reference, and troubleshooting.

---

## 2. Commit-by-commit ledger

All on `claude/refinement-n1ibom`, all in PR #1:

| Commit | Summary |
|---|---|
| `b4e9547` | Session handoff + cross-platform setup manual |
| `af3f34c` | Project Inari eval harness (checkers, runner, compare, mine, 30-task starter set, 41 tests) |
| `f544fda` | SWT refinement implementation report (for critique) |
| `b0426fd` | SWT Loop refinement (retrieval, experience store, cross-run learning, preference model, typed prescriptions, per-role endpoints) |
| `2b520aa` | (pre-session, cherry-picked) Harden SWT Loop: XSS, DoS, stale-key fixes |
| `d79c869` | (pre-session, cherry-picked) Add SWT Loop feature |

Note: the fork's `dev` branch is pure upstream (`35f867c`); all of your unique work lives on the two `claude/` branches.

---

## 3. Security & secret-leak review results

### 3.1 Secret / API-key leak scan — **CLEAN**

Scanned all four repos (odysseus, recursive-llm, rlm, tribev2):

- **Working trees** — grepped for OpenAI/Anthropic (`sk-…`, `sk-ant-…`), GitHub (`ghp_`, `github_pat_`), AWS (`AKIA…`), Slack, Google, Hugging Face, Telegram, private-key blocks, and generic `api_key=/secret=/password=` literals. **Zero real hits** (only false positives from HTML id strings).
- **Full git history** — every filename ever committed on any branch checked for `.env`, `.db`, `auth.json`, SSH keys, `.pem`, credentials: **none, ever**. Full patch-level history of recursive-llm (6 commits), rlm (81), tribev2 (8), and all 8 fork-unique odysseus commits scanned line-by-line: **clean**.
- GitHub's server-side scanner requires Advanced Security (not enabled on the fork); local scans are authoritative.

**Why:** nothing in these sessions handled a credential. The code reads keys only from runtime config/env, never embeds them; `.env` and `data/` are gitignored and were never tracked.

### 3.2 Security vulnerability review — **NO FINDINGS**

A dedicated security-review pass examined the full branch diff (~273 KB, all 33 changed files) plus cross-cutting dependencies. Result: **0 findings at or above reporting confidence.**

| Surface | Verdict | Why |
|---|---|---|
| **XSS** (model output → DOM) | Safe | Every dynamic value reaching `innerHTML`/`insertAdjacentHTML` in `static/js/swt/index.js` passes `_esc()` (escapes `& < > " '` — text and attribute contexts). Status text uses `textContent`. |
| **SQL injection** | Safe | SQLite: `?` placeholders, NULL-safe `owner IS ?`, static DDL only. Postgres: `%s` placeholders, `IS NOT DISTINCT FROM`, static schema; DSN from operator env, not requests. |
| **AuthN/AuthZ** | Safe | All six `/api/swt/*` endpoints gate on `_require_user`. Every read/write is owner-filtered in both backends; `history/{loop_id}` cannot cross users. |
| **Cross-user retrieval** | Safe | Retrieval passes `owner` to ChromaDB's `where` filter; gap write-backs are owner-stamped. |
| **Credential escalation via `model@endpoint`** | Safe | No raw URL/key fields in the API; specs resolve through the owner-scoped `_resolve_model`, so a user (or prompt-injected analyzer) can only reach their own endpoints. |
| **Code execution / deserialization** | Safe | No `eval`/`exec`/`pickle`/`yaml.load`/subprocess in the diff; all parsing is `json.loads`; SSE frames are `json.dumps`-encoded. |
| **evals/ harness** | Not a surface | CLI-only, never imported by `app.py`, unreachable over HTTP; paths are operator args; run labels sanitized before use as filenames. |

**Two low-confidence candidates were traced and rejected** (both below the reporting bar):

1. **Shared `"api"` pseudo-user silo** — API tokens from different users map to one `"api"` identity, sharing an SWT history silo. Rejected: this is the codebase-wide inherited pattern (same in `compare_routes.py`), not introduced by this branch. Optional hardening: switch `_require_user` to `effective_user`.
2. **Knowledge-base self-poisoning via prompt injection** — crafted context could steer the analyzer into writing junk gap-notes to ChromaDB. Rejected: writes are owner-stamped and retrieval owner-filtered, so a user can only degrade their *own* future answers; already documented, notes tagged `source: swt` for purging.

A supplementary pattern sweep of the wider upstream codebase found no `eval`/`exec`, no unsafe deserialization; the f-string SQL in `task_routes.py`/`email_helpers.py` interpolates only hardcoded table names (not injectable); and `shell=True` in `src/builtin_actions.py` is the intentional, privilege-gated agent shell feature.

---

## 4. Recommendations

### 4.1 Highest-leverage: measure before you build

Both handoffs admit it and it remains true — **no one yet knows whether the SWT loop improves answers or just adds latency/cost.** The eval harness now exists to answer this. Before adding any new subsystem:

- Run `model:GEN` vs `swt:GEN::CRIT` on the same 30 tasks and `compare`. If the loop's score doesn't clear the one-shot's noise floor, the loop is adding latency, not quality — and that's now a measurement, not a suspicion.
- Make the harness the gate for every Track A / Track B change (quantization tier, DSPy module, model swap). One variable at a time, compared against the baseline.

### 4.2 Guardrails to keep the discipline

- **Log real usage now** so the eval set can be mined from real workload in a couple of weeks, not invented from imagination. The starter 30 tasks are placeholder-but-real; expect to replace most of them.
- **Never tune prompts against the eval set.** When a task leaks into prompt engineering, retire it to a regression file and mine a fresh one.
- **Record VRAM per config by hand** next to each run label — the harness can't measure it portably, and the quality/speed/memory triple is what the quantization sweep actually needs.
- **Exit criteria for Stage 1** (from `evals/README.md`): selftest at 1.000, a ≥3-repeat baseline of the production model saved, and that baseline's `overall_std` below ~0.05. If the wobble is bigger than that, tighten the task set before trusting any comparison.

### 4.3 Security posture (not vulnerabilities — standing hygiene)

- **Enable secret-scanning push protection** on the fork: GitHub → Settings → Code security → Secret scanning → "Push protection." It blocks a recognized key format before it can ever land.
- If a key is ever committed: **rotate at the provider first** — deleting the commit does not unleak it.
- Keep `AUTH_ENABLED=true`; never expose port 7000 directly to the internet (the app is an admin console: shell tool, file access). Use a reverse proxy or Tailscale + HTTPS for remote access.
- When deploying the Postgres experience store, give `SWT_DATABASE_URL` a role with rights only on the `swt_main` schema.
- Optional multi-user hardening: if you hand API tokens to more than one person, switch SWT's `_require_user` to `effective_user` so token histories don't share the `"api"` silo.

### 4.4 Product / engineering habits (from the outside-perspective review)

- **Own the code, not just the vision** — read every diff before merge; write one test per feature by hand. If you can't test a module, you don't understand it yet.
- **Watch the paper-grafting reflex** — no new concept (RLM, TRIBE, ASI-EVOLVE, DSPy, etc.) enters the system until the last one has data showing it works. Apply the Reality Filter at design time, not cleanup time.
- **Deploy the small version before designing the big one** — get a real two-node loop running before architecting for N nodes.
- **Dogfood** — run your daily questions through the system for two weeks; real usage will tell you which of the known limitations actually matter.

### 4.5 Open technical follow-ups (deferred, non-blocking)

From the refinement report's known-limitations list, the ones worth revisiting when they start to bite:

- The Postgres experience store is review-verified only (no live-DB integration test, no connection pooling).
- `save_diagnosis` embeds the prompt synchronously inside the async engine — a slow HTTP embed could block the event loop; move it to a thread if it shows up in latency.
- Preference-model feature weights are hand-set, not learned; the logistic-head option is deferred until there's enough feedback data.
- The frontend still says "cognitive model" and doesn't render the four new SSE event types or expose the new toggles — API-only until a frontend pass.

---

## 5. What to do next (operator checklist)

On Node 1, from inside the repo with the app's Python environment:

1. `git fetch origin claude/refinement-n1ibom && git checkout claude/refinement-n1ibom`
2. `python -m evals run --tasks evals/tasks/core.jsonl --target selftest --label lint` → expect `score 1.000 ± 0.000`.
3. `python -m evals run --tasks evals/tasks/core.jsonl --target model:<your-model> --repeats 3 --label baseline` → **this is the first real baseline number.**
4. Check its `± std`: if above ~0.05, tighten/expand the task set before trusting comparisons.
5. Then start the quantization sweep (A1): serve two GGUF quantizations via llama.cpp, run each with `--target http:...`, and `compare` against the baseline — one variable at a time.

---

*Generated for `BrickeUpLinuxNewb/odysseus`, branch `claude/refinement-n1ibom`, PR #1. Companion documents in the repo: `SESSION_HANDOFF.md` (setup manual), `SWT_REFINEMENT_REPORT.md` (critique detail), `docs/swt.md` (feature docs), `evals/README.md` (harness usage).*
