# SWT Loop — automated generator / critic / analyzer

SWT ("Save The World") is an Odysseus feature that closes the critique loop that
`Compare` leaves to a human. Where Compare runs models side by side for you to
judge, SWT runs a small, automated **micro-loop** that judges and improves its
own answer:

```
generator → critic → analyzer → (cognitive model) → converge or repeat
```

It is a local, hardware-agnostic implementation of the ASI-EVOLVE
`learn → design → experiment → analyze` loop, with a TRIBE v2-inspired
user-cognition layer and RLM-style recursive context handling.

## How a round works

1. **Generator** answers the request, using any condensed reference material and
   the lessons the analyzer accumulated in previous rounds.
2. **Critic** judges the answer and returns a structured verdict
   (`accepted`, `summary`, `issues`, `missing_context`).
3. **Analyzer** — the keystone — diagnoses *why* the critic objected and
   classifies the root cause into one of three categories, then prescribes one
   concrete adjustment for the next round:

   | Category | Meaning | Adjustment |
   | --- | --- | --- |
   | `retrieval_failure` | The needed info was in the context but unused | Re-scan / widen retrieval |
   | `knowledge_gap` | The info was never provided | Tell the generator to state assumptions |
   | `orchestration_failure` | Wrong prompt/model for the task | Adjust the prompt, or switch the generator model |

4. **Cognitive model** predicts whether *you* (not just the critic) will accept
   the answer, from the history of what you have accepted and rejected.

The loop **converges** when the critic accepts **and** the cognitive model's
predicted satisfaction clears the threshold, or it returns the best answer once
it runs out of rounds.

## The cognitive model (TRIBE v2, applied locally)

TRIBE v2 predicts a brain's response to a stimulus with zero task-specific
training. Applied here — deliberately and with bounded scope — it predicts *your*
response to an answer from your interaction history. It is a proxy
user-cognition layer, not a claim to model a brain.

- It scores a candidate answer by similarity to past **accepted** vs.
  **rejected** answers (lexical overlap by default; a semantic embedding
  function can be injected).
- It stays neutral (cold-start) until it has at least a few feedback examples,
  deferring to the critic.
- You train it with the **Accept / Reject** buttons on the final answer.

## Recursive context (RLM)

Long reference material causes "context rot". When you paste more than the
budget, SWT condenses it RLM-style — split into chunks, extract only what is
relevant to the request, fold recursively — using your own local models. If a
Recursive Language Model package (`rlm`) is installed it is reported by the
status endpoint; the native path is what the loop uses so behavior is
deterministic and offline-testable.

## Using it

Open **SWT Loop** from the sidebar tools or the icon rail (the ↻ loop icon).
Enter a request, pick a **generator** and **critic** model (and optionally a
separate **analyzer** model — it defaults to the critic), optionally paste
reference material, set the round budget, and **Run loop**. Each round streams in
live. Accept or reject the final answer to train the cognitive model.

Models are whatever you have configured in Odysseus (local Ollama, an API
provider, etc.) — SWT routes every call through the same resolver and endpoints
the rest of the app uses.

## API

| Method & path | Purpose |
| --- | --- |
| `POST /api/swt/run` | Stream a loop run as Server-Sent Events |
| `POST /api/swt/feedback` | Record accept/reject (trains the cognitive model) |
| `GET /api/swt/history` | Recent loops for the current user |
| `GET /api/swt/history/{loop_id}` | One full loop record |
| `GET /api/swt/cognitive/stats` | Feedback counts |
| `GET /api/swt/status` | Capability probe (RLM availability, categories) |

`/api/swt/run` streams `data: {json}` events of type `loop_start`,
`context_condensing`, `context_condensed`, `round_start`, `generation`,
`critique`, `diagnosis`, `satisfaction`, `model_switch`, `error`, and
`loop_complete` (which carries the full result).

## Storage

SWT keeps its own SQLite database at `data/swt.db` — it does **not** add tables
to Odysseus's main schema, so there is no migration and no risk to existing
data. It stores each loop result and the accept/reject feedback that the
cognitive model learns from.

## Security notes

- **No code execution.** SWT only generates and displays *text*. It never runs
  model-generated code and never invokes Odysseus tools/shell from a loop
  result. The external Recursive Language Model packages work by executing
  model-written Python in a REPL; SWT deliberately does **not** use that path —
  it only checks whether the package is importable and runs its own condenser.
- **Owner-scoped.** Every endpoint requires an authenticated user. Loops,
  history, and cognitive-model feedback are filtered by owner, and model
  resolution goes through Odysseus's owner-scoped resolver, so one user can
  never read another's loops or use another's endpoint/API keys. There is no
  raw-URL input, so the model field cannot be turned into an SSRF vector.
- **Output escaping.** Model names and all streamed content are HTML-escaped for
  both text and attribute contexts before rendering, so a poisoned/rogue model
  server on the LAN cannot inject markup or attributes into the browser.
- **Bounded work.** Request size (prompt/context), round count, and the
  recursive condenser's chunk fan-out are all capped, so a single request cannot
  spiral into an unbounded number of model calls on a low-power node.
- **Prompt injection.** Reference material and the request are untrusted input to
  the models. A crafted context could talk the critic into accepting a bad
  answer, but this does not cross a privilege boundary — the worst case is a
  lower-quality answer, never code execution or data access. Treat pasted
  context the same way you treat any untrusted document.
- **Local data.** `data/swt.db` stores your prompts, answers, and accept/reject
  feedback in plaintext SQLite. Keep it out of Git (the repo already ignores
  `data/`) and treat it like the rest of your private data.

## Design notes

- The engine (`src/swt/`) never imports the LLM stack directly; it talks to a
  small `ModelAdapter`, so the whole loop is unit-tested offline with a scripted
  fake (`tests/test_swt_engine.py`).
- Every stage degrades gracefully: an unparseable critic verdict is treated as
  "needs work", the analyzer falls back to a keyword heuristic, and a failed
  context condense falls back to the raw context. The loop never stalls on a bad
  model response.
