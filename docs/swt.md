# SWT Loop — automated generator / critic / analyzer

SWT ("Save The World") is an Odysseus feature that closes the critique loop that
`Compare` leaves to a human. Where Compare runs models side by side for you to
judge, SWT runs an automated loop that judges and improves its own answer:

```
experience seed → [ retrieve → generate → critique → analyze → predict ] × rounds
```

It follows the ASI-EVOLVE `learn → design → experiment → analyze` cycle: the
analyzer turns every failed round into structured insight, that insight is
persisted to an **experience store**, and future runs on similar prompts read
it back — so the loop gets stronger across runs, not just within one. A
per-user **preference model** predicts whether *you* (not just the critic)
will accept the answer, and RLM-style recursive condensing keeps long
reference material from rotting the context.

## How a round works

0. **Retrieve** — top-k chunks for the current retrieval query are pulled from
   the knowledge base (ChromaDB, owner-scoped) and fed to the generator
   alongside any reference material you pasted. Skipped cleanly when ChromaDB
   is unavailable.
1. **Generator** answers the request, using the retrieved knowledge, condensed
   reference material, and the lessons the analyzer accumulated (including
   lessons seeded from prior runs).
2. **Critic** judges the answer and returns a structured verdict
   (`accepted`, `summary`, `issues`, `missing_context`).
3. **Analyzer** diagnoses *why* the critic objected, classifies the root cause,
   and emits a **typed prescription** the engine dispatches — each category has
   exactly one cure:

   | Category | Meaning | Dispatched cure |
   | --- | --- | --- |
   | `retrieval_failure` | The needed info existed but was not surfaced | Next round re-retrieves: wider top-k and/or a rewritten query |
   | `knowledge_gap` | The info was never captured at all | The gap is written back into the knowledge base so future runs retrieve it |
   | `orchestration_failure` | Wrong prompt/model for the task | Prompt hint, or switch the generator model mid-loop |

4. **Preference model** predicts whether *you* will accept the answer, from
   the history of what you have accepted and rejected.

The loop **converges** when the critic accepts **and** the predicted
satisfaction clears the threshold, or it returns the best answer once it runs
out of rounds.

## Learning across runs (the experience store)

Every diagnosis is persisted as a first-class row (category, rationale, typed
prescription, the prompt it came from). When a new loop starts, the engine
queries the store for prior diagnoses on similar prompts — semantically, when
an embedding backend is configured — and seeds their prompt hints and
retrieval adjustments before round 0. A prompt that needed wider retrieval
last week starts wide this week; a knowledge gap recorded last week is
retrievable this week.

Two storage backends sit behind one `ExperienceStore` interface:

- **SQLite** (`data/swt.db`) — the zero-config default; nothing in Odysseus's
  main schema is touched.
- **PostgreSQL** — set `SWT_DATABASE_URL=postgresql://…` and SWT keeps its
  tables in a dedicated `swt_main` schema of the shared database, so a
  distributed deployment has one experience store for every node. Requires
  `psycopg`; if the driver or the database is unavailable, SWT logs it and
  falls back to SQLite.

## The preference model

A per-user model of what you accept, learned from your Accept / Reject
feedback. (The idea of predicting a user's reaction from history is loosely
motivated by response-prediction work like TRIBE v2 — motivation only; the
mechanism is embeddings and interaction statistics, nothing neural.)

- With an embedding backend configured (Odysseus's standard
  `get_embedding_client()`: `EMBEDDING_URL` or local FastEmbed), answers are
  embedded and scored by a **prototype margin** — cosine distance to your
  accepted-answer centroid minus your rejected-answer centroid.
- Cheap **interaction features** (answer length, code-block use, hedging
  density) nudge the score toward whichever class the answer's *shape*
  resembles, because how an answer is shaped predicts acceptance too.
- Without embeddings it degrades to a similarity-weighted lexical vote — the
  `basis` field in the prediction (`embedding` / `lexical` / `cold-start`)
  always tells you which mode produced the score.
- It stays neutral (cold-start) until it has at least 3 feedback examples,
  deferring to the critic.

## Per-role endpoints (distributed deployments)

Every role — **generator, critic, analyzer, librarian** — takes a full
Odysseus model spec, and specs support `model@endpoint` (e.g.
`hermes3:8b@gpu-node`). That routes each role to a different configured
endpoint, which is how one loop spans multiple machines: generator on the GPU
box, critic on a second node, analyzer/librarian wherever there is headroom.
Bare model names keep everything on one box; nothing extra to configure.

The **librarian** role carries the context work (recursive condensing); it
defaults to the analyzer's model when unset.

## Recursive context (RLM)

Long reference material causes "context rot". When you paste more than the
budget, SWT condenses it RLM-style — split into chunks, extract only what is
relevant to the request, fold recursively — using your own local models.
Retrieval composes with this: retrieve → condense → generate. If a Recursive
Language Model package (`rlm`) is installed it is reported by the status
endpoint; the native path is what the loop uses so behavior is deterministic
and offline-testable.

## Using it

Open **SWT Loop** from the sidebar tools or the icon rail (the ↻ loop icon).
Enter a request, pick a **generator** and **critic** model (and optionally a
separate **analyzer** model — it defaults to the critic), optionally paste
reference material, set the round budget, and **Run loop**. Each round streams
in live. Accept or reject the final answer to train the preference model.

Models are whatever you have configured in Odysseus (local Ollama, an API
provider, etc.) — SWT routes every call through the same resolver and endpoints
the rest of the app uses.

## API

| Method & path | Purpose |
| --- | --- |
| `POST /api/swt/run` | Stream a loop run as Server-Sent Events |
| `POST /api/swt/feedback` | Record accept/reject (trains the preference model) |
| `GET /api/swt/history` | Recent loops for the current user |
| `GET /api/swt/history/{loop_id}` | One full loop record |
| `GET /api/swt/cognitive/stats` | Feedback counts |
| `GET /api/swt/status` | Capability probe (retrieval, embeddings, store backend, RLM) |

`/api/swt/run` accepts `librarian_model`, `use_retrieval`, `retrieval_k`, and
`use_experience` in addition to the original fields (`use_cognitive_model` is
the wire name for the preference-model toggle), and streams `data: {json}`
events of type `loop_start`, `experience_seeded`, `context_condensing`,
`context_condensed`, `round_start`, `retrieval`, `generation`, `critique`,
`diagnosis`, `satisfaction`, `retrieval_adjusted`, `knowledge_gap_recorded`,
`model_switch`, `error`, and `loop_complete` (which carries the full result).

## Storage

By default SWT keeps its own SQLite database at `data/swt.db` — no migration
of Odysseus's main schema. With `SWT_DATABASE_URL` set, it uses the shared
PostgreSQL database's `swt_main` schema instead (see above). Either backend
stores loop results, accept/reject feedback (with answer embeddings when
available), and every analyzer diagnosis.

## Security notes

- **No code execution.** SWT only generates and displays *text*. It never runs
  model-generated code and never invokes Odysseus tools/shell from a loop
  result. The external Recursive Language Model packages work by executing
  model-written Python in a REPL; SWT deliberately does **not** use that path —
  it only checks whether the package is importable and runs its own condenser.
- **Owner-scoped.** Every endpoint requires an authenticated user. Loops,
  history, diagnoses, feedback, and knowledge-base retrieval are filtered by
  owner, and model resolution goes through Odysseus's owner-scoped resolver, so
  one user can never read another's loops or use another's endpoint/API keys.
  There is no raw-URL input, so the model field cannot be turned into an SSRF
  vector (`model@endpoint` only selects among endpoints the owner already
  configured).
- **Output escaping.** Model names and all streamed content are HTML-escaped for
  both text and attribute contexts before rendering, so a poisoned/rogue model
  server on the LAN cannot inject markup or attributes into the browser.
- **Bounded work.** Request size (prompt/context), round count, retrieval top-k
  (hard cap 12, even under analyzer widening), and the recursive condenser's
  chunk fan-out are all capped, so a single request cannot spiral into an
  unbounded number of model calls on a low-power node.
- **Prompt injection.** Reference material, retrieved chunks, and the request
  are untrusted input to the models. A crafted context could talk the critic
  into accepting a bad answer, but this does not cross a privilege boundary —
  the worst case is a lower-quality answer, never code execution or data
  access. Note that `knowledge_gap` write-back stores analyzer-authored notes
  in the knowledge base; they are tagged `source: swt` so they can be audited
  or purged.
- **Local data.** The experience store holds your prompts, answers, diagnoses,
  and accept/reject feedback in plaintext. Keep it out of Git (the repo already
  ignores `data/`) and treat it like the rest of your private data.

## Design notes

- The engine (`src/swt/`) never imports the LLM stack directly; it talks to
  small protocol seams — `ModelAdapter`, `Retriever`, `KnowledgeWriter`,
  `ExperienceStore` — so the whole loop is unit-tested offline with scripted
  fakes (`tests/test_swt_engine.py`).
- Every stage degrades gracefully: an unparseable critic verdict is treated as
  "needs work", the analyzer falls back to a keyword heuristic, retrieval and
  the experience store fail soft to empty results, a failed context condense
  falls back to the raw context, and a missing embedding backend drops the
  preference model to lexical mode. The loop never stalls on a bad model
  response or a missing service.
- **Honest caveat:** the loop's control flow is fully tested offline; whether
  the analyzer's diagnoses are *accurate* against real models — and whether the
  loop improves answers enough to justify the extra latency/cost — has not
  been benchmarked yet.
