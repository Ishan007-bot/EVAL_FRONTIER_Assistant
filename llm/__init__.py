"""LLM backend factory.

`get_llm()` returns the active backend based on the BACKEND env var:
    ollama -> local dev (Ollama server)
    hf     -> HF Spaces deploy (transformers)
    gemini -> frontier model via Google Gemini API (comparison / demos)

Every backend exposes the same tiny interface:
    .name           -> str  (e.g. "ollama:qwen2.5:1.5b")
    .stream(msgs)   -> generator yielding response text tokens
    .chat(msgs)     -> str  (full response)

Backends are imported lazily so the local app never imports torch and
the deployed app never imports the ollama client.
"""
import os


def get_llm():
    backend = os.getenv("BACKEND", "ollama").lower()
    if backend == "ollama":
        from .ollama_backend import OllamaBackend
        return OllamaBackend()
    if backend == "hf":
        from .hf_backend import HFBackend
        return HFBackend()
    if backend == "gemini":
        from .gemini_backend import GeminiBackend
        return GeminiBackend()
    raise ValueError(f"Unknown BACKEND '{backend}'. Use 'ollama', 'hf', or 'gemini'.")
