"""Compatibility shim so `ragas` imports with langchain-community >= 0.4.

ragas 0.4.x does `from langchain_community.chat_models.vertexai import ChatVertexAI`
and `from langchain_community.llms import VertexAI`. Both modules were removed from
langchain-community, but the same classes live in `langchain-google-vertexai`.

Import this module BEFORE importing ragas:

    import app.ragas_compat  # noqa: F401
    from ragas import evaluate
"""
import importlib
import sys
import types


def _register(module_name: str, **attrs) -> None:
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
        return
    except ImportError:
        pass
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module


def apply() -> None:
    from langchain_google_vertexai import ChatVertexAI, VertexAI

    _register("langchain_community.chat_models.vertexai", ChatVertexAI=ChatVertexAI)
    _register("langchain_community.llms.vertexai", VertexAI=VertexAI)


apply()
