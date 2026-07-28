# SWT Loop Refinement — Implementation Report (for critique)

**Repo:** `BrickeUpLinuxNewb/odysseus` · **Branch:** `claude/refinement-n1ibom`
**Commit under review:** `b0426fd` — "Refine SWT Loop: retrieval, experience store, cross-run learning, preference model"
**Base:** the branch carries the two original SWT commits (`e339a97` feature, `de12f2e` hardening) cherry-picked onto `dev` tip `35f867c`, then one refinement commit on top.
**Diff size:** 14 files, +1,789 / −380. Tests: 12 → 23, all offline, all passing. Full repo suite (4,468 tests) green.

**Assignment being executed** (from the handoff doc): make the SWT loop *more deeply
connected* to Odysseus per work items D1–D7 — wire real retrieval, make the
experience store pluggable (SQLite + Postgres `swt_main`), make the loop learn
across runs, re-tailor the "TRIBE v2 cognitive model" into an honest
embedding-based Preference Model, support per-role endpoints, replace free-text
analyzer prescriptions with typed actions, and polish docs/tests. Security
hardening and frontend polish were explicitly **out of scope**.

---

## 1. File inventory after the change

| File | Lines | Status | Role |
|---|---|---|---|
| `src/swt/schemas.py` | 242 | rewritten | dataclasses; new `Prescription`, `RetrievalAdjustment`; `LoopConfig` grew role/retrieval/experience fields |
| `src/swt/engine.py` | 382 | rewritten | orchestrator: experience seeding → condense → per-round retrieve/generate/critique/analyze/predict → typed dispatch |
| `src/swt/analyzer.py` | 219 | rewritten | LLM + heuristic diagnosis, now emits typed `Prescription`; new JSON contract keys |
| `src/swt/store.py` | 361 | rewritten | `ExperienceStore` protocol, SQLite impl, diagnosis rows + similarity read-back, `create_store()` factory |
| `src/swt/store_pg.py` | 222 | **new** | PostgreSQL backend, `swt_main` schema, lazy `psycopg` |
| `src/swt/preference.py` | 258 | **new** | `PreferenceModel` (prototype margin + weighted vote + interaction features), `default_embed_fn()` |
| `src/swt/retrieval.py` | 106 | **new** | `Retriever`/`KnowledgeWriter` protocols, `RagRetriever` over shared VectorRAG, `MAX_RETRIEVAL_K = 12` |
| `src/swt/similarity.py` | 49 | **new** | shared `tokens`/`jaccard`/`cosine`/`similarity` helpers |
| `src/swt/cognitive_model.py` | 12 | gutted | back-compat shim: `CognitiveModel = PreferenceModel` |
| `src/swt/models.py` | 73 | touched | docstring only: documents per-role `model@endpoint` |
| `src/swt/__init__.py` | 69 | rewritten | new exports; `CognitiveModel` deprecated alias kept |
| `routes/swt_routes.py` | 214 | rewritten | wires embed_fn/retriever/store factory; new request fields; richer `/status` |
| `tests/test_swt_engine.py` | 560 | rewritten | 23 tests; fake adapter/retriever/embed_fn |
| `docs/swt.md` | — | rewritten | architecture, storage backends, API, security notes, honest caveats |

Untouched on purpose: `src/swt/recursive.py` (condenser kept as-is),
`static/js/swt/index.js` (frontend out of scope), `app.py` registration.

---

## 2. D6 — Typed prescriptions (done first; everything dispatches on it)

### Schema changes (`schemas.py`)

```python
@dataclass
class RetrievalAdjustment:
    widen_k: int = 0        # add this many results to the current top-k
    query_rewrite: str = "" # replacement retrieval query; empty keeps current
    note: str = ""

@dataclass
class Prescription:
    prompt_hint: str = ""                       # any category; accumulates
    retrieval: Optional[RetrievalAdjustment] = None  # retrieval_failure only
    switch_generator_to: str = ""               # orchestration_failure only
    write_to_kb: str = ""                       # knowledge_gap only

@dataclass
class Diagnosis:
    category: DiagnosisCategory
    rationale: str
    prescription: Prescription = field(default_factory=Prescription)
    raw: str = ""
```

The old flat `Diagnosis` fields (`next_prompt_hint`, `retrieval_adjustment`,
`switch_generator_to`) are **removed**, not aliased. `Diagnosis.to_dict()` now
nests `prescription`; `from_dict()` classmethods were added on
`Diagnosis`/`Prescription`/`RetrievalAdjustment` for store read-back.

**Compatibility decision:** the frontend only reads `diagnosis.category` and
`diagnosis.rationale` (verified by reading `_renderDiagnosis` in
`static/js/swt/index.js`), so restructuring the diagnosis JSON was judged safe
without touching the frontend. Stored `result_json` blobs from *old* loops keep
the old shape; `get_loop`/`history` return raw JSON so old records still render
(frontend only touches the same two keys). Nothing migrates old rows.

### Analyzer changes (`analyzer.py`)

- New LLM JSON contract keys: `category`, `rationale`, `next_prompt_hint`,
  `retrieval_query_rewrite` (retrieval_failure only), `knowledge_note`
  (knowledge_gap only), `switch_generator_to` (orchestration_failure only).
- `_prescription()` builder attaches **only the lever matching the category**
  (e.g. a `switch_generator_to` from the LLM on a `retrieval_failure` diagnosis
  is dropped).
- `WIDEN_K_STEP = 2`: every retrieval_failure prescription carries
  `RetrievalAdjustment(widen_k=2, query_rewrite=<llm's or "">)` — widening is
  engine-owned policy, not something the LLM chooses per-round.
- `knowledge_gap` with no LLM-provided note gets a synthesized default:
  `_gap_note()` = `"Known gap: {missing_context joined or summary} (surfaced answering: {prompt[:200]})"`.
- The keyword heuristic fallback (`_heuristic`) builds the same typed
  prescriptions; its retrieval_failure branch uses
  `query_rewrite=" ".join(critique.missing_context)[:300]`.
- `analyze()` signature unchanged except heuristic calls now pass `prompt`
  through for gap-note synthesis.

---

## 3. D1 — Retrieval wired into the loop

### New seam (`retrieval.py`)

```python
class Retriever(Protocol):
    async def retrieve(self, query: str, *, k: int, owner: Optional[str]) -> List[str]: ...

class KnowledgeWriter(Protocol):
    async def record_gap(self, text: str, *, owner: Optional[str], loop_id: str) -> bool: ...
```

`RagRetriever` implements both over the **app's existing singleton**
(`src.rag_singleton.get_rag_manager()` → `VectorRAG`), deliberately not
constructing a second ChromaDB client. Key properties:

- `retrieve()` runs the synchronous `rag.search(query, k, owner=owner)` via
  `asyncio.to_thread` (VectorRAG does HTTP to ChromaDB under the hood; keeps
  the event loop free).
- Results are the `document` field of each hit, truncated to
  `_SNIPPET_CHARS = 1500` chars each.
- `k` clamped to `MAX_RETRIEVAL_K = 12` inside the retriever **and** in the
  engine's `_RetrievalState`, so analyzer widening can never grow unbounded.
- Every failure path (singleton returns None, search throws) returns `[]` and
  logs a warning — retrieval loss never stalls the loop.
- `record_gap()` calls `rag.add_document(text, metadata)` with metadata
  `{source: "swt", kind: "knowledge_gap", loop_id, created_at, owner?}` so
  write-backs are auditable/purgeable.
- `retrieval_available()` module function backs the `/status` probe.

### Engine integration (`engine.py`)

- `_RetrievalState(query, k)` holds the current retrieval params with a
  `dirty` flag (True initially). The engine retrieves at the top of a round
  **only when dirty**, i.e. round 0 and any round after a prescription changed
  the params — unchanged params reuse the previous chunks rather than
  re-querying ChromaDB every round.
- Retrieved chunks are injected into the generator's user message as a
  numbered `RETRIEVED KNOWLEDGE:` section **alongside** (not replacing) the
  condensed `REFERENCE MATERIAL:` section; sections joined with `---`.
- `had_context` (the analyzer's retrieval_failure-vs-knowledge_gap
  discriminator) is now `bool(condensed) or bool(retrieved)`.
- Dispatch: on `retrieval_failure` with a `prescription.retrieval`, the engine
  calls `retrieval.adjust(widen_k, query_rewrite)` (clamps k, swaps query,
  sets dirty) and emits a `retrieval_adjusted` SSE event.
- `LoopConfig` gains `use_retrieval: bool = True`, `retrieval_k: int = 4`.
- `run_loop()` signature: `run_loop(cfg, adapter, store, preference,
  retriever=None, knowledge_writer=None)` — retriever optional; None = no
  retrieval (all old tests pass unmodified in behavior). If
  `knowledge_writer` is None and the retriever has `record_gap`, the retriever
  serves both roles.

---

## 4. D2 — Pluggable experience store

### Protocol (`store.py`)

`ExperienceStore` protocol = the old store surface plus:

```python
def save_diagnosis(self, loop_id, round_index, diagnosis, owner, *, prompt) -> None
def similar_prior_diagnoses(self, prompt, owner, *, k=3) -> List[Dict]
```

`similar_prior_diagnoses` returns
`{loop_id, prompt, similarity, created_at, diagnosis: Diagnosis}` dicts, best
match first. (Part G of the handoff suggested `List[Diagnosis]`; the dict
carries the match metadata the engine and any future UI need — deviation noted.)

### SQLite backend (default, `SwtStore`)

- New table `swt_diagnoses(diagnosis_id PK, loop_id, round_index, owner,
  prompt, prompt_vec TEXT/JSON, category, rationale, prescription_json,
  created_at)` + owner/created index.
- `swt_feedback` gains `answer_vec TEXT` (JSON-encoded embedding). Existing DBs
  are migrated in place via `PRAGMA table_info` check + `ALTER TABLE ADD COLUMN`.
- `SwtStore(db_path=None, embed_fn=None)`: the store owns embedding at write
  time — `save_diagnosis` embeds the prompt, `add_feedback` embeds the answer
  (`_encode_vec` returns None on any embed failure; row is stored without a
  vector). `recent_feedback` decodes `answer_vec` back to a list.
- `similar_prior_diagnoses` fetches the most recent **200** rows for the owner
  (`WHERE owner IS ? AND category != 'none'`), then ranks in Python via the
  shared `rank_similar_diagnoses()` — cosine when both the query prompt and the
  row have vectors, Jaccard otherwise, `min_similarity = 0.1` filter, top-k.
  Ranking in Python (not SQL) was chosen because SQLite has no vector ops and
  200 rows is trivial; the same helper is reused by the PG backend.

### PostgreSQL backend (`store_pg.py`)

- Same method surface; all tables in `CREATE SCHEMA IF NOT EXISTS swt_main`
  (loops/feedback/diagnoses mirroring SQLite, `DOUBLE PRECISION` timestamps).
- `import psycopg` is **inside `__init__`** so the module imports cleanly with
  no driver installed.
- Short-lived autocommit connections per call, mirroring SQLite's
  connection-per-call model. No pool (documented as a deliberate simplicity
  choice matching the existing style).
- Owner NULL-safe matching via `owner IS NOT DISTINCT FROM %s` (equivalent of
  SQLite's `owner IS ?`).
- `ON CONFLICT (loop_id) DO UPDATE` replicates `INSERT OR REPLACE` for loops.
- Vectors stored as JSON text (no pgvector dependency); ranking reuses the same
  Python helper. **Untested against a live Postgres** — no PG in the sandbox;
  only the module import and the fallback path are covered by tests.

### Factory

`create_store(embed_fn=None)`: `SWT_DATABASE_URL` starting with
`postgres://`/`postgresql://` → try `PostgresExperienceStore`; **any** exception
(missing driver, unreachable DB) logs a warning and falls back to SQLite.
Default (no env var) → SQLite.

---

## 5. D3 — Learning across runs

Two mechanisms, both in `engine.py`:

**Seeding (read path).** With `use_experience=True` (new `LoopConfig` field,
default True), before condensing/round 0 the engine calls
`store.similar_prior_diagnoses(cfg.prompt, cfg.owner, k=_EXPERIENCE_K=3)` and:
- appends up to `_MAX_SEED_HINTS = 3` distinct `prescription.prompt_hint`
  strings to `prompt_hints` (so round 0's generator system prompt already
  carries prior lessons);
- for any prior `RetrievalAdjustment`, calls `retrieval.adjust(...)` — a prompt
  that needed wider retrieval last time starts wide;
- emits an `experience_seeded` SSE event `{matches, hints, retrieval_k}`.
Seeding is wrapped in try/except; store failure = no seed, loop proceeds.

**Write path.** Every non-NONE diagnosis is persisted per round via
`store.save_diagnosis(loop_id, i, diagnosis, owner, prompt=cfg.prompt)`
(try/except, non-fatal). Separately, `knowledge_gap` prescriptions with
`write_to_kb` text are written to the KB via `knowledge_writer.record_gap(...)`,
deduped within the run by a `recorded_gaps` set (the same gap diagnosed in two
rounds writes once), emitting `knowledge_gap_recorded {round, ok}`.

**Consequence acknowledged:** seeding reads diagnoses written by *failed*
rounds of prior loops, including loops that eventually converged. There is no
"was this hint actually useful" feedback signal yet — hints are seeded purely
by prompt similarity + recency.

---

## 6. D4 — Preference model (the TRIBE reframe)

### Naming / honesty

- New module `preference.py`, class `PreferenceModel`. Docstrings state the
  mechanism plainly and demote TRIBE v2 to "loose motivation, not mechanism".
- `cognitive_model.py` reduced to a 12-line shim (`CognitiveModel =
  PreferenceModel`) so external imports keep working; `src/swt/__init__.py`
  exports both names, alias documented as deprecated.
- `SatisfactionPrediction.basis` values changed: `"history"` → `"lexical"`,
  plus `"embedding"` and `"cold-start"`. (Frontend displays basis as opaque
  text, so the rename is cosmetic on the wire.)
- `LoopConfig.use_cognitive_model` renamed to `use_preference_model`; the HTTP
  request field **stays** `use_cognitive_model` for client compatibility and is
  mapped in the route.

### Scoring pipeline (`predict`)

1. `history = store.recent_feedback(owner, limit=200)`; `< MIN_HISTORY (3)` →
   neutral 0.5, basis `cold-start` (unchanged behavior).
2. **Prototype margin** (preferred): if an embed_fn is set and history has ≥1
   embedded accepted and ≥1 embedded rejected answer:
   `margin = cos(answer_vec, accepted_centroid) − cos(answer_vec, rejected_centroid)`;
   `base = sigmoid(margin / _MARGIN_TAU)` with `_MARGIN_TAU = 0.15`. Basis
   `embedding`. Centroids computed on the fly from stored `answer_vec`s
   (mixed-dimension guard: `_centroid` returns None if dims disagree, falls
   through to the vote).
3. **Similarity-weighted vote** (fallback): per-row weight = cosine when both
   candidate and row have vectors, else Jaccard; score = weighted fraction
   accepted; `den == 0` → 0.5. Basis `embedding` if any cosine was used, else
   `lexical`. This preserves the original algorithm as the degraded mode.
4. **Interaction-feature adjustment** on top of the base score:
   features = `log_length`, `has_code` (``` present), `hedge_density`
   (hedge-phrase count / word count). For each feature where accepted-mean and
   rejected-mean differ (>1e-6), move `±_FEATURE_STEP (0.04)` toward the nearer
   class; total clamped to `±_FEATURE_CAP (0.12)`; final score clamped [0,1].
   Requires both classes present, else delta 0. Reasons name the drivers
   ("Shape matches accepted answers (code-block use, length).").
   **The weights are hand-set, not learned** — this is design (b)-lite from the
   handoff; the logistic-head option was deferred.

### Embedding wiring

- Answer vectors are computed **once at feedback time** by the store
  (`add_feedback` embeds), so `predict` embeds only the candidate answer.
  History rows recorded before embeddings were enabled simply lack vectors and
  participate lexically.
- `default_embed_fn()` (in `preference.py`) wraps
  `src.embeddings.get_embedding_client()`:
  `lambda text: client.encode([text])[0].tolist()`; returns None when no
  backend exists. Embeds are one text per call (no batching) — accepted cost
  since predictions embed a single answer.

---

## 7. D5 — Per-role endpoints

Decision: **no new endpoint/credential fields.** Investigation of
`src/ai_interaction.py::_resolve_model` showed the resolver already accepts
`"model@endpoint_name"` and matches endpoints owner-scoped from the DB. So
per-role node routing = each role passing its own spec through the existing
adapter, which already caches per `(spec, owner)`.

Concretely:
- `LoopConfig` gains `librarian_model: str = ""` with
  `resolved_librarian()` → librarian or analyzer or critic. The **librarian**
  role now runs the context condenser (previously the analyzer's model did).
- `RunRequest` gains `librarian_model` (same 200-char cap as other roles).
- Docstrings in `LoopConfig`, `OdysseusModelAdapter`, and `docs/swt.md`
  document the `model@endpoint` pattern per role.
- Rationale for rejecting raw endpoint URLs/keys in the request: it would
  create an SSRF/credential surface the hardening pass just closed;
  `model@endpoint` only selects among endpoints the owner already configured.

---

## 8. Engine control flow after the change (summary)

```
loop_start
if use_experience: seed hints + retrieval params from similar prior diagnoses  → experience_seeded
if context & use_recursive_context: condense once w/ librarian model           → context_condensing/_condensed
for round i in max_rounds:
    round_start
    if use_retrieval and retriever and retrieval.dirty: retrieve(query, k)     → retrieval
    generate (retrieved + condensed + accumulated hints)                        → generation | error(stage=generator)+break
    critique (strict JSON; unparseable ⇒ not accepted; critic error ⇒ accept)   → critique | error(stage=critic)
    analyze → typed Diagnosis                                                   → diagnosis
    if category != none: store.save_diagnosis (non-fatal)
    if use_preference_model: preference.predict                                 → satisfaction
    convergence: critic accepted AND (no pref model OR score ≥ threshold) → break
    if accepted but pref model unhappy: append alignment hint, continue
    dispatch prescription:
        prompt_hint → hints
        retrieval_failure + retrieval → adjust k/query, mark dirty              → retrieval_adjusted
        knowledge_gap + write_to_kb (once per text) → record_gap                → knowledge_gap_recorded
        orchestration_failure + switch → swap generator                         → model_switch
finalize best answer; store.save_loop (non-fatal)                               → loop_complete
```

Invariants preserved from the original: every model/store/retriever failure
degrades (accept-and-continue, heuristic, empty result, raw context) — nothing
stalls the SSE stream; best-answer fallback at max rounds; the
"critic accepted but preference model disagrees" continuation hint.

---

## 9. Routes & API surface

- Lazy process-wide singletons: store (via `create_store`), preference model,
  retriever, and a **probed-once** embed_fn (None is a valid cached outcome).
  The model adapter remains per-run (stale-credential fix from the hardening
  commit preserved).
- `RunRequest` new fields: `librarian_model` (≤200), `use_retrieval` (True),
  `retrieval_k` (4, ge=1 le=12), `use_experience` (True). `use_cognitive_model`
  kept as the wire name → mapped to `use_preference_model`.
- `run_loop` invoked with `retriever=_get_retriever()`.
- `/api/swt/status` now returns `retrieval_available`, `embeddings_available`,
  `store_backend` (class name), alongside the original fields.
- `/api/swt/cognitive/stats` path kept (wire compat) though it's
  preference-model data now.

New SSE event types: `experience_seeded`, `retrieval`, `retrieval_adjusted`,
`knowledge_gap_recorded` (frontend's event switch has a silent `default:`, so
unknown types are ignored — verified before adding).

---

## 10. Tests (23, all offline)

Kept/updated (renames: `use_preference_model`, `PreferenceModel`, basis
`lexical`): convergence on accept; iterate + hint application (now also asserts
the hint text reaches round 1's system prompt via recorded messages);
max-rounds best-answer; unparseable critic ⇒ rejected; generator error ⇒ error
event + clean completion; cold start; lexical learning; feedback stats;
heuristic category split (now also asserts typed levers: `widen_k`, gap note
content); analyzer accept short-circuit; splitter budget; short-context no-op;
condenser fan-out bound.

New:
- **Retrieval**: chunks reach the generator prompt (`RETRIEVED KNOWLEDGE`
  section + chunk text), `retrieval` event chunk count, exact `(query, k)`
  call recording.
- **Widening**: retrieval_failure prescription → second retrieve call is
  `("project spec requirements", 4 + WIDEN_K_STEP)` + `retrieval_adjusted`
  event.
- **Gap write-back**: knowledge_gap → `record_gap` called with the note,
  exactly once across two failing rounds; event `ok=True`.
- **Experience store**: `save_diagnosis` + `similar_prior_diagnoses` ranking
  (closest prompt first), owner isolation (other owner sees `[]`).
- **Seeding**: pre-saved diagnosis hint appears in `experience_seeded` event
  and in round 0's generator system prompt.
- **Engine persistence**: a failing round leaves a queryable diagnosis row.
- **Factory fallback**: `SWT_DATABASE_URL=postgresql://…` with no driver →
  `SwtStore` (default DB path monkeypatched to tmp so tests write no repo
  artifacts).
- **Prototype margin**: deterministic 3-dim fake embed_fn; accepted-like
  answer scores > 0.5 > rejected-like; basis `embedding`.
- **Feature adjustment**: code-block answer moves toward accepted (positive
  delta, reason names "code-block"); long prose scores lower.

Test doubles: `FakeAdapter` (persona routing on system-prompt substrings, full
message recording), `FakeRetriever` (scripted chunks; records queries and
gaps), `_fake_embed` (keyword-count vectors).

Verification runs: `tests/test_swt_engine.py` 23 passed; full suite 4,461+
passed after installing sandbox-missing optional deps (bs4, mcp, markdown, nh3,
icalendar/caldav) — the only failures ever seen were those missing packages,
unrelated to this diff.

---

## 11. Known limitations / deliberate deferrals (critique targets)

1. **Postgres backend never ran against a real server.** Schema SQL, conflict
   clause, and `IS NOT DISTINCT FROM` queries are review-verified only. No
   integration test, no connection pooling, no retry on transient PG errors
   (a failed `save_diagnosis` is swallowed by the engine's try/except).
2. **Vector search is brute-force in Python** over the last 200 rows
   (diagnoses) / 200 feedback rows. Fine at hobby scale; no ANN index, no
   pgvector, embeddings stored as JSON text. Recency cutoff at 200 silently
   drops older experience.
3. **Experience seeding has no efficacy signal.** Hints are seeded by prompt
   similarity + recency only; a bad hint that never helped is re-seeded
   forever. No dedup against semantically-equal-but-differently-worded hints
   (exact-string dedup only). `min_similarity=0.1` Jaccard can seed from
   fairly unrelated prompts on short prompts.
4. **Knowledge-gap write-back stores the *statement of the gap*, not the
   answer.** Next run retrieves "Known gap: X is undocumented" — useful for
   flagging, but nothing ever resolves/expires these notes, and repeated
   distinct-wording gaps accumulate in the KB (dedup is per-run only).
   Poisoning surface: analyzer-authored text enters the shared RAG store
   (mitigated by `source: swt` metadata + owner scoping, documented, but no
   cap/TTL).
5. **Preference-model features are hand-weighted** (±0.04 steps, ±0.12 cap,
   τ=0.15) with no calibration data; the logistic-regression head from the
   handoff's option (b) was deferred. Prototype margin uses one centroid per
   class — multimodal preferences (user accepts two very different styles)
   blur.
6. **Feedback vectors are frozen at record time.** If the embedding model
   changes (different dim/space), old vectors silently mismatch: `_centroid`
   returns None on dim mismatch (falls back to vote) and per-pair cosine
   returns 0.0 on dim mismatch — degraded, not corrupt, but also not detected
   or re-embedded.
7. **Retrieval query rewrite fully replaces the user prompt as the search
   query** on prescription; there's no blending or reversion if the rewrite
   retrieves worse chunks (no relevance feedback on retrieval itself).
8. **Chunks reuse between rounds**: retrieval only re-runs when params change;
   if the KB is updated mid-loop (e.g. by this very loop's gap write-back), the
   current loop won't see it unless a prescription dirties the state. (Gap
   write-back + immediate re-retrieve in the same loop was considered and
   skipped: the note describes a gap, not new knowledge.)
9. **Old flat `Diagnosis` JSON in previously-saved loops is not migrated**;
   `Diagnosis.from_dict` on an old row yields an empty `Prescription`. Only
   affects pre-refinement rows read through `similar_prior_diagnoses` — and
   those can't exist since the diagnoses table is new; old *loop blobs* are
   only ever returned raw to the UI. Believed safe; flagging for review.
10. **`use_cognitive_model` wire-name kept** (mapped to
    `use_preference_model`) and `/cognitive/stats` path kept — honest naming
    stops at the HTTP boundary to avoid breaking the untouched frontend.
11. **Frontend not updated** (out of scope per handoff): UI still says
    "cognitive model", doesn't render the four new SSE event types (silently
    ignored), and has no controls for `librarian_model`/`retrieval_k`/
    `use_retrieval`/`use_experience` — API-only until a frontend pass.
12. **No live-model or end-to-end HTTP test** (no ChromaDB/embedding backend/
    LLM in the sandbox): the route module imports cleanly and the engine is
    fully covered with fakes, but `POST /api/swt/run` was not exercised
    through FastAPI, and whether the loop *improves answers* on real models
    remains unbenchmarked (stated in docs).
13. **Similarity of prompts uses whole-prompt embedding/Jaccard** — no
    normalization (casing handled; no stemming/stopwords), so lexical-mode
    experience matching is crude on long prompts.
14. **`similar_prior_diagnoses` returns dicts, not `List[Diagnosis]`** as
    Part G sketched (deliberate: callers need match metadata; noted as an
    interface deviation).

---

## 12. Claims to verify (what a critic should try to falsify)

- No call path lets analyzer output grow retrieval beyond k=12 or trigger
  more than one retrieval per round.
- A totally absent ChromaDB / embedding backend / Postgres yields byte-for-byte
  the old single-box behavior except for new no-op events
  (`experience_seeded {matches:0}`, `retrieval {chunks:0}` when a retriever
  exists but returns nothing — note: routes always pass a `RagRetriever`, so
  with ChromaDB down every round-0 emits `retrieval {chunks: 0}`… actually
  only round 0, since params never dirty without a retrieval_failure).
- Owner isolation: every new query (`swt_diagnoses`, feedback vectors,
  retrieval `where owner`) is owner-filtered; `record_gap` writes carry owner
  metadata.
- The engine never awaits a sync DB/RAG call on the event loop (store calls
  are sync SQLite — same as pre-existing behavior, unchanged; retrieval is
  thread-offloaded; embedding in `add_feedback` runs inside the sync route
  handler, and in `save_diagnosis` inside the async engine — **this one is a
  fair performance nit: a slow HTTP embed inside `save_diagnosis` blocks the
  event loop**).
- SSE JSON serializability of every yielded event (all primitives/dicts).
