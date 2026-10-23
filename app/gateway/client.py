import os
from types import SimpleNamespace

import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# Production gateway config:
#   - Fallback: primary @rag/llama-3.3-70b-versatile → @brag/llama-3.1-8b-instant on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/llama-3.1-8b-instant"}},
    ]
}

# Temporary switch: USE_GATEWAY=false in .env bypasses Portkey and calls Groq directly
# (no Portkey key needed, no fallback/cache/retry). Default is the Portkey gateway.
USE_GATEWAY = os.getenv("USE_GATEWAY", "true").strip().lower() != "false"
DIRECT_GROQ_MODEL = os.getenv("DIRECT_GROQ_MODEL", settings.GROQ_MODEL)


class _DirectGroqCompletions:
    """Mimics portkey_client.chat.completions.create() but talks to Groq directly."""

    def __init__(self):
        from groq import Groq
        self._groq = Groq(api_key=settings.GROQ_API_KEY)

    def create(self, messages, temperature=0.1, **kwargs):
        return self._groq.chat.completions.create(
            model=DIRECT_GROQ_MODEL,
            messages=messages,
            temperature=temperature,
        )


if USE_GATEWAY:
    portkey_client = Portkey(
        api_key=settings.PORTKEY_API_KEY,
        config=GATEWAY_CONFIG
    )
else:
    logfire.warning("USE_GATEWAY=false: Portkey gateway bypassed, calling Groq directly.")
    portkey_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_DirectGroqCompletions())
    )


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    if not USE_GATEWAY:
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=DIRECT_GROQ_MODEL,
            temperature=0,
        )
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=GATEWAY_CONFIG,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"