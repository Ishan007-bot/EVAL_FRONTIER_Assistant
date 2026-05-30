"""Frontier-tier backend: Llama-3.3-70B via the Groq hosted API.

Used for the assistant comparison and as the LLM-as-judge during evaluation.
A large open-weight foundation model served via a hosted API (equivalent to
the DeepSeek example in the brief), chosen because Groq's free tier is
generous enough to run — and re-run — the full evaluation. Requires a free
GROQ_API_KEY from https://console.groq.com (no credit card needed).

Groq is OpenAI-compatible, so our {role, content} messages (system/user/
assistant) pass through unchanged. The `groq` package is imported lazily so
this module stays importable without it installed (e.g. on the HF Space).
"""
import os


class GroqBackend:
    def __init__(self):
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.api_key = os.getenv("GROQ_API_KEY") or None
        self._client = None

    @property
    def name(self) -> str:
        return f"groq:{self.model}"

    def _client_or_create(self):
        if self._client is None:
            from groq import Groq  # lazy import
            self._client = Groq(api_key=self.api_key)
        return self._client

    def stream(self, messages):
        client = self._client_or_create()
        stream = client.chat.completions.create(
            model=self.model, messages=messages, stream=True,
            temperature=0.7, max_tokens=1024,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def chat(self, messages) -> str:
        client = self._client_or_create()
        resp = client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=0.7, max_tokens=1024,
        )
        return resp.choices[0].message.content or ""
