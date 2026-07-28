# Odysseus Session Handoff & Setup Manual

**Branch:** `claude/refinement-n1ibom` &nbsp;·&nbsp; **PR:** [#1](https://github.com/BrickeUpLinuxNewb/odysseus/pull/1) &nbsp;·&nbsp; **Fork:** `BrickeUpLinuxNewb/odysseus`
**Date:** 2026-07-12

This document has two parts:

- **Part 1 — What was built this session** (the SWT Loop refinement and the Project Inari eval harness).
- **Part 2 — A full setup manual** for running Odysseus on Windows 11, then Fedora, then Linux Mint.

---

# PART 1 — What was built this session

Three commits landed on `claude/refinement-n1ibom`, all in PR #1:

| Commit | What |
|---|---|
| `b0426fd` | SWT Loop refinement (retrieval, experience store, cross-run learning, preference model) |
| `f544fda` | SWT refinement implementation report (for critique review) |
| `af3f34c` | Project Inari eval harness (the measurement keystone) |

## 1.1 SWT Loop refinement

The SWT ("Save The World") loop is Odysseus's automated **generator → critic → analyzer** loop. It started as a working-but-shallow single-box demo; this session deepened it into the connected system it was designed as. Seven work items:

**D1 — Retrieval wired in.** New `src/swt/retrieval.py` adds a `Retriever` / `KnowledgeWriter` seam with a `RagRetriever` over the app's shared VectorRAG/ChromaDB instance. Top-k knowledge-base chunks now feed the generator alongside pasted context, and a `retrieval_failure` diagnosis actually re-runs retrieval next round with a widened `k` and/or a rewritten query (hard-capped at k=12).

**D2 — Pluggable experience store.** Persistence sits behind an `ExperienceStore` protocol. SQLite (`data/swt.db`) stays the zero-config default; a new `PostgresExperienceStore` (selected by `SWT_DATABASE_URL`, lazy `psycopg`) keeps its tables in a dedicated `swt_main` schema of the shared database. Diagnoses are stored as first-class rows with prompt embeddings, not buried in a JSON blob.

**D3 — Cross-run learning.** A new run queries prior diagnoses on similar prompts (cosine over stored vectors, lexical fallback) and seeds their hints and retrieval adjustments before round 0. `knowledge_gap` diagnoses write the gap back into the knowledge base (tagged `source: swt`) so a future run retrieves it.

**D4 — Preference model.** The oversold "TRIBE v2 cognitive model" was renamed and rebuilt honestly as `PreferenceModel` (`src/swt/preference.py`). It uses real embeddings — a prototype margin between accepted/rejected answer centroids — plus interaction-shape features (length, code blocks, hedging), with the lexical vote and cold-start deferral kept as fallbacks. `CognitiveModel` remains as a deprecated alias.

**D5 — Per-role endpoints.** Generator / critic / analyzer / a new **librarian** role each take a full model spec, so `model@endpoint` routes roles to different nodes in a distributed deployment. Bare names keep single-box behavior. This reused the existing resolver rather than adding a new SSRF/credential surface.

**D6 — Typed prescriptions.** The analyzer's loose `next_prompt_hint` / `retrieval_adjustment` / `switch_generator_to` strings became a typed `Prescription` dataclass the engine dispatches on — one concrete cure per diagnosis category.

**D7 — Polish.** Docs rewritten with an honest "not yet benchmarked" caveat, overselling comments removed, routes expose the new toggles, `/status` reports capabilities. Tests grew from 12 to 23, still fully offline via fakes. The full repo suite (4,468 tests) stays green.

## 1.2 The critique report

A standalone document (`SWT_REFINEMENT_REPORT.md`) captures the full change with a section of **14 known limitations** and a section of **falsifiable claims** — written specifically to be fed to a critique loop.

## 1.3 Project Inari eval harness — the measurement keystone

A test rig for non-deterministic model output. The ruler every efficiency (Track A) and reliability (Track B) change gets measured against before it is believed. New `evals/` package with a CLI: `python -m evals {run, compare, mine}`.

Design decisions baked in:

- **Programmatic scoring only** — seven checker kinds (choice / exact / contains / contains_all / regex / numeric / json). No LLM judge, because on this cluster the judge would be the same rubber it is measuring. Free-form generation quality is out of scope for v1.
- **Variance is part of the answer** — every task runs N times; the summary reports `mean ± std`, and `compare` refuses to call a delta real unless it clears **2× the pooled noise floor**. It also flags per-task regressions a flat mean would hide.
- **Quality/speed recorded together** — per-call latency (p50/p95) and throughput (true tokens/sec when the backend reports usage). VRAM stays operator-recorded per config.
- **Four targets** — `selftest` (answers from the rubric itself: the task-file linter), `model:SPEC` (Odysseus resolver path), `http:URL::MODEL` (raw llama.cpp/vLLM endpoint — the quantization-sweep path), and `swt:GEN::CRIT` (the whole loop as a black box, making the "does looping help?" question a two-command experiment).
- **`mine`** pulls real prompts from the SWT store as `kind: todo` candidates; the loader refuses to run them until a rubric is written — mining supplies workload, never the grading standard.

Starter set: 30 tasks (10 routing, 10 extraction, 5 JSON, 5 closed QA), selftest-linted to 1.000. 41 offline tests, no model required.

### Demonstration run (in-sandbox, simulated models)

With no real LLM in the build environment, a simulated "q4" (weak) and "q5" (better) model server demonstrated the full workflow:

```
selftest lint             score 1.000 ± 0.000   (all rubrics passable)
sim-q4 baseline           score 0.667 ± 0.000
sim-q4 rerun              score 0.644 ± 0.019
sim-q5 candidate          score 0.922 ± 0.051

compare q4-baseline vs q4-rerun   delta -0.022  ->  WITHIN-NOISE  (correctly refused)
compare q4-baseline vs q5         delta +0.256  ->  BETTER + flagged one regression
```

The q4/q5 numbers are from a simulator written to demo the harness; they say nothing about real quantizations. The first real number comes from running against your models on Node 1.

---

# PART 2 — Setup Manual: Running Odysseus

> **Which version to run.** Your fork's default branch is `dev`. The branch with this session's work (SWT refinement + eval harness) is `claude/refinement-n1ibom`. The commands below clone `dev`; to get this session's work, check out the feature branch after cloning (shown at the end of each OS section).
>
> **Two ways to run everywhere:** **Docker** (simplest, most isolated — recommended for a first run) or **Native** (a Python virtual environment — better for local GPU model serving and development). Each OS section covers both.
>
> **Requirements common to all:** Python 3.11 or newer (for native installs), Git, and about 5 GB free disk for the app plus dependencies (models are extra and optional).

---

## 2.1 Windows 11

### A. Install the prerequisites

1. **Git for Windows** — download from <https://git-scm.com/download/win>, run the installer, accept defaults. This also gives you `bash.exe`, which Odysseus's Cookbook and shell tools use. Verify in a new PowerShell window:
   ```powershell
   git --version
   ```
2. **Python 3.11+** — download from <https://www.python.org/downloads/windows/>. **On the first installer screen, tick "Add python.exe to PATH."** Verify:
   ```powershell
   py --version
   ```
   If that prints 3.11 or higher, you're set. If it points at an older version, note which 3.11+ you installed — you'll pass it explicitly below (e.g. `py -3.11`).
3. *(Optional, for a local model on Windows)* **Ollama** — download from <https://ollama.com/download>. After installing, you'll point Odysseus at `http://localhost:11434/v1` inside Settings.

### B. Option 1 — Docker (simplest)

1. Install **Docker Desktop** from <https://www.docker.com/products/docker-desktop/>. Launch it once and let it finish starting (whale icon steady in the tray).
2. Open **PowerShell** and run:
   ```powershell
   git clone https://github.com/BrickeUpLinuxNewb/odysseus.git
   cd odysseus
   copy .env.example .env
   docker compose up -d --build
   ```
   The first build takes several minutes.
3. Open **<http://localhost:7000>** in your browser once the containers are healthy.
4. Get your first-login admin password:
   ```powershell
   docker compose logs odysseus | Select-String -Pattern "password"
   ```
   Log in as `admin` with that password, then change it in **Settings**.

To stop it later: `docker compose down` (from the `odysseus` folder). To start again: `docker compose up -d`.

### C. Option 2 — Native (one-command launcher)

From the folder where you want the code:

```powershell
git clone https://github.com/BrickeUpLinuxNewb/odysseus.git
cd odysseus
powershell -ExecutionPolicy Bypass -File .\launch-windows.ps1
```

The launcher creates the virtual environment, installs dependencies, runs first-time setup, and starts the server. It's safe to re-run. When it finishes, open **<http://localhost:7000>** and log in with the admin password printed in the terminal.

**If you prefer to do it by hand** (or the launcher fails), run each step:

```powershell
git clone https://github.com/BrickeUpLinuxNewb/odysseus.git
cd odysseus
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python setup.py
python -m uvicorn app:app --host 127.0.0.1 --port 7000
```

(If `python` inside the venv points somewhere odd, use `py -3.12` or another installed 3.11+ version at the `venv` step.)

### D. Windows notes

- **To reach it from your phone** over a trusted LAN or Tailscale, the native launcher needs a flag (editing `.env` alone is not enough on native Windows):
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\launch-windows.ps1 -BindHost 0.0.0.0
  ```
  Keep `AUTH_ENABLED=true`; never expose the port straight to the public internet.
- **If login won't disable even with `AUTH_ENABLED=false`:** you edited `.env` in Notepad and it saved a UTF-8 BOM. Re-save `.env` as **UTF-8 without BOM** (VS Code: *Save with Encoding → UTF-8*).
- **Local GPU serving** of vLLM/SGLang needs Linux/WSL2. On plain Windows, use Ollama for a local model and point Odysseus at it in Settings.

### E. Get this session's branch (optional)

After cloning, before running:

```powershell
cd odysseus
git fetch origin claude/refinement-n1ibom
git checkout claude/refinement-n1ibom
```

Then run the eval harness lint to confirm it works:

```powershell
python -m evals run --tasks evals/tasks/core.jsonl --target selftest --label lint
```

You should see `score 1.000 ± 0.000`.

---

## 2.2 Fedora (KDE Plasma / Workstation)

### A. Install the prerequisites

Open **Konsole** (or any terminal):

```bash
sudo dnf install -y git python3 python3-pip python3-virtualenv tmux
```

Verify Python is 3.11+ (Fedora 37+ ships 3.11 or newer):

```bash
python3 --version
```

`tmux` is needed for Cookbook's background model downloads/serves. If you'll run the Docker path instead, you still want `git`.

### B. Option 1 — Docker (simplest)

1. Install Docker Engine + Compose plugin:
   ```bash
   sudo dnf install -y dnf-plugins-core
   sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
   sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
   sudo systemctl enable --now docker
   ```
2. *(Recommended)* Let your user run Docker without `sudo`:
   ```bash
   sudo usermod -aG docker $USER
   ```
   **Log out and back in** for the group change to take effect.
3. Clone and start:
   ```bash
   git clone https://github.com/BrickeUpLinuxNewb/odysseus.git
   cd odysseus
   cp .env.example .env
   docker compose up -d --build
   ```
4. Open **<http://localhost:7000>**. Get the admin password:
   ```bash
   docker compose logs odysseus | grep -i password
   ```

> **SELinux note (Fedora-specific):** Fedora enforces SELinux by default. The bundled `docker-compose.yml` handles volume labels, but if you ever see permission-denied on the `./data` or `./logs` mounts, relabel them with:
> ```bash
> sudo chcon -Rt svirt_sandbox_file_t ./data ./logs
> ```

### C. Option 2 — Native (virtual environment)

```bash
git clone https://github.com/BrickeUpLinuxNewb/odysseus.git
cd odysseus
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py
python -m uvicorn app:app --host 127.0.0.1 --port 7000
```

Open **<http://localhost:7000>** and log in with the admin password printed in the terminal. To run it again later, from the `odysseus` folder: `source venv/bin/activate` then the `uvicorn` line.

### D. Fedora notes

- **NVIDIA GPU:** install the driver from RPM Fusion (`akmod-nvidia`) plus the CUDA toolkit; for Docker GPU passthrough run `scripts/check-docker-gpu.sh` in the repo. **AMD GPU (ROCm):** run `scripts/check-docker-amd-gpu.sh` and follow its reported `.env` values. CPU-only works fine for the core app — GPU only matters for local model *serving*.
- **Firewall:** to reach Odysseus from another device, bind to `0.0.0.0` (below) and open the port: `sudo firewall-cmd --add-port=7000/tcp` (add `--permanent` to persist).

### E. Get this session's branch (optional)

```bash
cd odysseus
git fetch origin claude/refinement-n1ibom
git checkout claude/refinement-n1ibom
python -m evals run --tasks evals/tasks/core.jsonl --target selftest --label lint
```

Expect `score 1.000 ± 0.000`.

---

## 2.3 Linux Mint (Ubuntu-based)

### A. Install the prerequisites

Open a terminal:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip tmux
python3 --version
```

Linux Mint 21.x ships Python 3.10; **22.x ships 3.12**. Odysseus needs **3.11+**. Check the version printed above:

- If it's **3.11 or higher**, continue.
- If it's **3.10** (Mint 21.x), add a newer Python via the deadsnakes PPA:
  ```bash
  sudo apt install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install -y python3.11 python3.11-venv
  ```
  Then use `python3.11` wherever the steps say `python3`.

### B. Option 1 — Docker (simplest)

1. Install Docker using Docker's official convenience script (works cleanly on Mint's Ubuntu base):
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   ```
2. Let your user run Docker without `sudo`, then re-login:
   ```bash
   sudo usermod -aG docker $USER
   ```
   **Log out and back in.**
3. Clone and start:
   ```bash
   git clone https://github.com/BrickeUpLinuxNewb/odysseus.git
   cd odysseus
   cp .env.example .env
   docker compose up -d --build
   ```
4. Open **<http://localhost:7000>**. Admin password:
   ```bash
   docker compose logs odysseus | grep -i password
   ```

### C. Option 2 — Native (virtual environment)

Using the Python that satisfied the 3.11+ check in step A (`python3`, or `python3.11` if you added it):

```bash
git clone https://github.com/BrickeUpLinuxNewb/odysseus.git
cd odysseus
python3.11 -m venv venv     # or: python3 -m venv venv  (if python3 is already 3.11+)
source venv/bin/activate
pip install -r requirements.txt
python setup.py
python -m uvicorn app:app --host 127.0.0.1 --port 7000
```

Open **<http://localhost:7000>** and log in with the admin password from the terminal.

### D. Linux Mint notes

- **Firewall (ufw):** Mint ships `ufw` (often inactive). If active and you want LAN access after binding to `0.0.0.0`: `sudo ufw allow 7000/tcp`.
- **GPU:** Mint uses Ubuntu's driver manager — install NVIDIA drivers via *Driver Manager*, or ROCm for AMD, only if you intend to serve models locally. The core app is CPU-fine.
- Everything else (Docker overlays, Ollama endpoints) follows the same steps as Fedora and the main setup guide.

### E. Get this session's branch (optional)

```bash
cd odysseus
git fetch origin claude/refinement-n1ibom
git checkout claude/refinement-n1ibom
python -m evals run --tasks evals/tasks/core.jsonl --target selftest --label lint
```

Expect `score 1.000 ± 0.000`.

---

## 2.4 First-login checklist (all platforms)

1. Open **<http://localhost:7000>** (Docker/native Linux/Windows) — note macOS uses `7860`.
2. Log in as **`admin`** with the temporary password from the terminal / `docker compose logs`.
3. **Change the admin password** in Settings immediately.
4. In **Settings → Models**, add your model provider: a local Ollama endpoint (`http://localhost:11434/v1`), an OpenAI-compatible server, or an API key.
5. Review **`data/auth.json`**: disable open signup unless you want it, and keep only your own account as admin.
6. Keep `AUTH_ENABLED=true`. Do not expose the raw port to the public internet — use a reverse proxy or a private network (Tailscale) with HTTPS if you need remote access.

## 2.5 Common ports reference

| Port | Service |
|---|---|
| `7000` | Odysseus web UI (Docker/native Linux/Windows) |
| `7860` | Odysseus web UI (macOS start script) |
| `8080` | SearXNG (bundled search) |
| `8100` | ChromaDB (vector memory) |
| `11434` | Ollama (local models) |

## 2.6 If something goes wrong

- **Port 7000 already in use:** set `APP_PORT=7001` in `.env` (Docker) or pass `--port 7001` to the `uvicorn` command (native), then reopen on the new port.
- **`chromadb-client` conflict** (native): `./venv/bin/pip uninstall chromadb-client -y && ./venv/bin/pip install --force-reinstall chromadb`.
- **Check container health** (Docker): `docker compose ps` and `docker compose logs --tail=120 odysseus`.
- **Full troubleshooting and configuration** live in `docs/setup.md` inside the repo.

---

*Generated for `BrickeUpLinuxNewb/odysseus`, branch `claude/refinement-n1ibom`, PR #1.*
