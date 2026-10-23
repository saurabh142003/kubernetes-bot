# Deployment Guide: Docker + Render

How to build the project as Docker images and deploy it on Render, in a way that avoids the
dependency problems seen during local setup.

**Contents**

1. [What is verified and what is not](#1-what-is-verified-and-what-is-not)
2. [Why Docker, and what it fixes](#2-why-docker-and-what-it-fixes)
3. [What goes into the images](#3-what-goes-into-the-images)
4. [Files to add to the repo](#4-files-to-add-to-the-repo)
5. [Build and test locally with Docker](#5-build-and-test-locally-with-docker)
6. [Load data into Qdrant (before deploying)](#6-load-data-into-qdrant-before-deploying)
7. [Deploy on Render](#7-deploy-on-render)
8. [Environment variables](#8-environment-variables)
9. [After the deploy: checks](#9-after-the-deploy-checks)
10. [Things to know](#10-things-to-know)
11. [Updating dependencies later](#11-updating-dependencies-later)

---

## 1. What is verified and what is not

| Item | Status |
|---|---|
| Backend dependency set ([requirements-prod.txt](requirements-prod.txt) + [constraints.txt](constraints.txt)) installs in a **clean Python 3.12 venv** with no conflicts (`pip check` clean) | Verified (on Windows) |
| In that clean venv: `app.main` imports, guardrails block off-topic, jailbreak and sensitive prompts, a technical question passes, FlashRank imports | Verified |
| Size of that environment | about 600 MB (no torch) |
| `docker build` of the Dockerfiles below | **Not run yet.** Docker Desktop was not running when this was written |
| Install on Linux (the Docker image) with the same pinned versions | **Not run yet.** The pins came from a Windows install; Linux wheels for the same versions normally exist, but confirm with the first build |
| Memory use of the running backend on Render | **Not measured** |

Run the local Docker build in [section 5](#5-build-and-test-locally-with-docker) before pushing to Render.
If the build fails, the error will point at one specific package, which is much easier to fix than
a failure on a remote server.

---

## 2. Why Docker, and what it fixes

Problems hit during local setup, and how this setup prevents each one:

| Problem seen earlier | Cause | Prevention here |
|---|---|---|
| `ragas` and `nemoguardrails` failed to install | Python 3.14 has no prebuilt wheels for `scikit-network` and `annoy`, so pip tried to compile them and needed C++ Build Tools | The image uses **Python 3.12** (`python:3.12-slim-bookworm`), pinned in the Dockerfile |
| `pip` stuck for a long time on `google-*` versions | Dependency resolver backtracking | Versions are pinned with a **constraints file**, so the resolver has nothing to search |
| `transformers` could not import (`huggingface-hub` too old) and a `click` conflict | Packages disagreeing about shared dependencies | The constraints file records the combination that actually worked together |
| `ragas` import error | Newer `langchain-community` removed a module `ragas` needs | `ragas` is **not installed** in production (only evals use it) |
| RAM full during install | `torch` (via `sentence-transformers`) is multiple GB | `torch` and `sentence-transformers` are **left out** of the image |

Why `sentence-transformers` can be dropped: it is only a fallback in
[embeddings.py](app/services/retrieval/embeddings.py) that runs when Jina fails, and it produces
768-dim vectors, while the Qdrant collection uses 2048-dim vectors. A fallback embedding could not
search that collection anyway.

---

## 3. What goes into the images

Two images, one repo:

| Image | Runs | Dockerfile | Dependencies |
|---|---|---|---|
| **Backend** | FastAPI (`app.main:app`) | `Dockerfile` | `requirements-prod.txt` + `constraints.txt` |
| **UI** | Streamlit (`ui/app.py`) | `Dockerfile.ui` | `requirements-ui.txt` |

**Not in the backend image** (on purpose): `DATA/`, `processed_data/`, `ui/`, the ingestion parsers
(pypdf, pdfplumber, unstructured, python-docx, python-pptx, bs4), `ragas`, `deepeval`, `torch`,
`sentence-transformers`, the `.env` file and `.logfire/`.

The ingestion code under `app/ingestion/` is still copied (it is part of `app/`) but its parser
libraries are not installed. Nothing in the running API imports them. Ingestion is run from your
own machine ([section 6](#6-load-data-into-qdrant-before-deploying)).

---

## 4. Files to add to the repo

Already created:

* [requirements-prod.txt](requirements-prod.txt): the 16 backend packages
* [constraints.txt](constraints.txt): exact versions from the verified environment

Create the rest in the project root.

### `Dockerfile` (backend)

```dockerfile
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANGSMITH_TRACING=false

# OpenMP runtime used by onnxruntime (FlashRank / NeMo embeddings)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so this layer is cached until the requirements change
COPY requirements-prod.txt constraints.txt ./
RUN pip install --prefer-binary -r requirements-prod.txt -c constraints.txt

COPY app/ ./app/

# Download the local models once at build time, so the first request is not slow
RUN python -c "from flashrank import Ranker; Ranker(cache_dir='/tmp/flashrank')" \
 && python -c "from app.guardrails import initialize_rails; initialize_rails()"

EXPOSE 8000
# Render sets $PORT; fall back to 8000 for local runs
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

### `Dockerfile.ui` (Streamlit)

```dockerfile
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-ui.txt .
RUN pip install --prefer-binary -r requirements-ui.txt

COPY ui/ ./ui/

EXPOSE 8501
CMD ["sh", "-c", "streamlit run ui/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --server.fileWatcherType none"]
```

### `requirements-ui.txt`

```text
streamlit==1.66.0
requests==2.34.2
python-dotenv==1.2.4
logfire==5.2.0
```

These four are the only third-party imports in [ui/app.py](ui/app.py).

### `.dockerignore`

```text
# Secrets and local state
.env
.env.*
.logfire/
.git/
.claude/
.vscode/
.idea/

# Environments and caches
.venv/
.venv_old314/
__pycache__/
*.py[cod]
.deepeval/

# Not needed inside the images
DATA/
processed_data/
*.md
```

Do **not** ignore `ui/` here: the UI image needs it.

### `render.yaml` (optional Blueprint: creates both services in one go)

```yaml
services:
  - type: web
    name: rag-backend
    runtime: docker
    dockerfilePath: ./Dockerfile
    healthCheckPath: /
    autoDeploy: true
    plan: standard          # pick a plan with at least 2 GB RAM, see section 10
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: JINA_API_KEY
        sync: false
      - key: QDRANT_URL
        sync: false
      - key: QDRANT_API_KEY
        sync: false
      - key: LOGFIRE_TOKEN
        sync: false
      - key: PORTKEY_API_KEY
        sync: false
      - key: LANGSMITH_TRACING
        value: "false"

  - type: web
    name: rag-ui
    runtime: docker
    dockerfilePath: ./Dockerfile.ui
    autoDeploy: true
    envVars:
      - key: BACKEND_URL        # public URL of rag-backend, set after it is deployed
        sync: false
      - key: LOGFIRE_TOKEN
        sync: false
```

`sync: false` means Render asks you for the value in the dashboard instead of reading it from the file,
so no secret is ever written into the repo.

---

## 5. Build and test locally with Docker

Start **Docker Desktop** first and wait until it says the engine is running.

```powershell
# Backend
docker build -t rag-backend .
docker run --rm --env-file .env -p 8000:8000 rag-backend
```

In another terminal:

```powershell
curl http://localhost:8000/
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"q\":\"tell me a joke\",\"thread_id\":\"t1\"}"
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"q\":\"what is a Kubernetes pod?\",\"thread_id\":\"t1\"}"
```

Expected: the joke is blocked by guardrails, and the Kubernetes question returns an answer with sources.

```powershell
# UI (points at the backend running on your machine)
docker build -f Dockerfile.ui -t rag-ui .
docker run --rm -e BACKEND_URL=http://host.docker.internal:8000 -p 8501:8501 rag-ui
```

Open http://localhost:8501.

Notes:

* `--env-file` passes `.env` into the container. Docker does **not** strip quotes or spaces around `=`,
  so write lines as `KEY=value`, not `KEY = "value"`.
* If the **build** fails, the message names the package. Send it over and it can be fixed in
  `constraints.txt`.
* If the **run** fails with a missing-module error, a package is missing from `requirements-prod.txt`.

---

## 6. Load data into Qdrant (before deploying)

The images do not contain `DATA/` and do not run ingestion. Ingest once from your own machine, with
the full local environment (`.venv`, Python 3.12), against the same Qdrant Cloud cluster:

```powershell
python -m app.ingestion.processor DATA --wipe
```

* Use `--wipe` the first time and whenever the embedding model changes (vector size must match).
* Re-run (without `--wipe`) whenever you add documents.
* The deployed backend only **reads** from Qdrant.

---

## 7. Deploy on Render

### Before you push

1. `.gitignore` is in place ([.gitignore](.gitignore)). It excludes `.env`, `.logfire/` and the venvs.
2. Run `git status` and confirm **neither `.env` nor `.logfire/` is listed**.
3. Push everything (including the Dockerfiles and the requirement files) to one GitHub repo.

### Option A: Blueprint (both services at once)

1. Render dashboard: **New**, **Blueprint**, pick the repo. Render reads `render.yaml`.
2. Fill in every variable marked `sync: false` (table in [section 8](#8-environment-variables)).
3. Leave `BACKEND_URL` empty for now, deploy, then continue below.

### Option B: By hand (one service at a time)

1. **New**, **Web Service**, pick the repo.
2. **Language / Runtime:** Docker.
3. **Dockerfile Path:** `./Dockerfile` for the backend, `./Dockerfile.ui` for the UI.
4. **Health Check Path** (backend): `/`
5. Choose an instance type (see [section 10](#10-things-to-know) about memory).
6. Add the environment variables, then **Create Web Service**.

### Connect the UI to the backend

1. Wait for the backend to go live and copy its URL, for example `https://rag-backend.onrender.com`.
2. In the **UI** service, set `BACKEND_URL` to that URL (no trailing slash).
3. Redeploy the UI.

Render provides the `PORT` variable itself; both Dockerfiles already read it.

---

## 8. Environment variables

Set these in the Render dashboard (never in the repo).

### Backend (`rag-backend`)

| Variable | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | yes | Used by the gateway (direct mode) and the guardrail fallback check |
| `JINA_API_KEY` | yes | Query embeddings |
| `QDRANT_URL` (or `QDRANT_CLUSTER_ENDPOINT`) | yes | Qdrant Cloud URL, e.g. `https://xxxx.cloud.qdrant.io:6333` |
| `QDRANT_API_KEY` | yes | |
| `LOGFIRE_TOKEN` | yes | There is no `.logfire/` folder in the container, so the token is how Logfire authenticates |
| `PORTKEY_API_KEY` | if using Portkey | Also needs the Portkey providers with slugs `rag` and `brag` |
| `USE_GATEWAY` | optional | `false` skips Portkey and calls Groq directly |
| `DIRECT_GROQ_MODEL` | with `USE_GATEWAY=false` | Pick a model your Groq key can use (e.g. `openai/gpt-oss-120b`). `llama-3.3-70b-versatile` was not available on the key used during development |
| `GUARD_GROQ_MODEL` | optional | Guardrail fallback model. Default `qwen/qwen3.8-27b` |
| `LANGSMITH_TRACING` | optional | `false` stops the 401 warnings when there is no LangSmith key |

### UI (`rag-ui`)

| Variable | Required | Notes |
|---|---|---|
| `BACKEND_URL` | yes | Public URL of the backend |
| `LOGFIRE_TOKEN` | optional | The UI prints an error line at startup if it is missing, but still runs |

---

## 9. After the deploy: checks

Replace the URL with yours.

```powershell
curl https://rag-backend.onrender.com/
# {"message":"Enterprise LangGraph RAG API is live."}

curl -X POST https://rag-backend.onrender.com/query -H "Content-Type: application/json" -d "{\"q\":\"who is Virat Kohli\",\"thread_id\":\"t1\"}"
# status "Blocked by guardrails."

curl -X POST https://rag-backend.onrender.com/query -H "Content-Type: application/json" -d "{\"q\":\"what is a Kubernetes pod?\",\"thread_id\":\"t1\"}"
# status "Response generated." and non-empty "sources"
```

Then open the UI and ask one off-topic and one technical question. Check Logfire for the spans
(`🛡️ Guardrails Check`, `🧠 Planner Decision`, `🔍 Knowledge Retrieval`, `✍️ LLM Synthesis`).

If a technical question returns an answer with **empty sources**, look in the Render logs for
`Qdrant Search Failed`. That usually means a wrong `QDRANT_URL` or `QDRANT_API_KEY`.

---

## 10. Things to know

* **Memory (RAM).** NeMo, its embedding model, FlashRank and the LangChain stack all load into the
  backend. Usage was not measured. Start on a plan with at least 2 GB, watch the memory graph in
  Render's dashboard, and move down only if there is clear room. A very small instance risks
  being killed for running out of memory. Check Render's current plan table for exact sizes.
* **Cold starts.** Free or sleeping instances can take a while to wake up. The UI waits at most 60 seconds
  ([ui/app.py:92](ui/app.py#L92)), so the first message after a long idle period can time out. Pre-downloading
  the models at build time (the warm-up step in the Dockerfile) removes the biggest delay.
* **Conversation memory is in RAM.** LangGraph `MemorySaver` keeps history inside the running process, so
  a restart or redeploy clears all conversations. Run **one instance** only; with several, a user's
  messages would land on different instances with different memory.
* **Groq rate limits.** On the free tier, a model has a tokens-per-minute cap (8,000 was seen in testing).
  Each question uses the guardrail check, the planner and the responder, so several users at once can hit the limit.
  A paid Groq tier or the Portkey fallback target helps.
* **Guardrails fail open.** If Groq is unavailable, the small LLM check is skipped (a warning is logged)
  and the message passes. The embedding-based rails still work, as they do not call Groq.
* **No local embedding fallback in the image.** If Jina is down, query embedding raises an error and the API
  returns its generic apology message. Planned behaviour, since a fallback could not search the 2048-dim collection anyway.
* **Vector size.** Jina v4 is set to 2048 dimensions. If this ever changes, re-ingest with `--wipe`.
* **Platform-specific pins.** `constraints.txt` came from Windows, so it also lists a few Windows-only
  packages (for example `pywin32`, `colorama`). They are never installed on Linux, because a constraints file
  only pins versions. A few Linux-only extras (such as `uvloop`) are not pinned and use the latest compatible version.
* **Secrets.** Only the Render dashboard holds the keys. If a key ever appears in a commit, treat it as
  leaked and create a new one.

---

## 11. Updating dependencies later

Change the lockfile only from a **clean** environment, never from your everyday venv (which contains
evaluation and ingestion packages and different pins):

```powershell
py -3.12 -m venv .venv-prod
.\.venv-prod\Scripts\activate
pip install -r requirements-prod.txt
pip check
pip freeze | Out-File -Encoding utf8 constraints.txt
```

(In Windows PowerShell, `pip freeze > constraints.txt` writes UTF-16, which pip cannot read. Use `Out-File -Encoding utf8` as shown.)

Then test with `docker build` ([section 5](#5-build-and-test-locally-with-docker)) before pushing. Render rebuilds on every push
to the branch when `autoDeploy` is on.
