import time
import logfire
import requests
from app.config import settings

BATCH_SIZE = 32
_JINA_URL = "https://api.jina.ai/v1/embeddings"
_JINA_MODEL = "jina-embeddings-v4"
_JINA_DIM = 2048
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  # "jina" or "fallback"


# ── Jina client ────────────────────────────────────────────────────────────────

def _jina_embed(texts: list[str], task: str) -> list[list[float]]:
    """Call the Jina embeddings API. task: 'retrieval.passage' (docs) or 'retrieval.query'."""
    resp = requests.post(
        _JINA_URL,
        headers={
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": _JINA_MODEL,
            "task": task,
            "dimensions": _JINA_DIM,
            "input": texts,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_jina() -> bool:
    """Try one embed call to verify Jina is reachable."""
    if not settings.JINA_API_KEY:
        logfire.warning("JINA_API_KEY is not set. Will use sentence-transformers fallback.")
        return False
    try:
        _jina_embed(["probe"], "retrieval.query")
        logfire.info(f"Jina embeddings ready ({_JINA_MODEL}, {_JINA_DIM}-dim).")
        return True
    except Exception as e:
        logfire.warning(f"Jina probe failed: {e}. Will use sentence-transformers fallback.")
        return False


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _model_type is not None:
        return

    if _probe_jina():
        _model_type = "jina"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model."""
    _init()
    return _JINA_DIM if _model_type == "jina" else _FALLBACK_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "jina":
        # Exponential backoff: 1 s → 2 s → 4 s → 8 s (4 attempts total)
        for attempt in range(4):
            try:
                return _jina_embed(batch, "retrieval.passage")
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 429 and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Jina rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Jina embedding failed: {e}")
                    raise
        raise RuntimeError("Jina rate limit persisted after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "jina":
        return _jina_embed([query], "retrieval.query")[0]
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
