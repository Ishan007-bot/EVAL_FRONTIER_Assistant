"""Frontier backend: Google Gemini via the google-genai SDK.

Used for the assistant comparison and as the LLM-as-judge during evaluation.
Selectable as BACKEND=gemini, so the same Streamlit app can run on the frontier
model for side-by-side demos. Requires a free GEMINI_API_KEY from
https://aistudio.google.com (no credit card needed for the free tier).

The `google.genai` package is imported lazily so this module stays importable
without it installed (e.g. on the HF Space, which only ships the OSS deps).
"""
import os


class GeminiBackend:
    def __init__(self):
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = os.getenv("GEMINI_API_KEY") or None
        self._client = None

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def _client_or_create(self):
        if self._client is None:
            from google import genai  # lazy import
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _convert(messages):
        """Gemini takes the system prompt separately, uses role 'model' for the
        assistant, and wraps text in parts."""
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system, contents

    def _config(self, system):
        from google.genai import types
        return types.GenerateContentConfig(system_instruction=system) if system else None

    def stream(self, messages):
        client = self._client_or_create()
        system, contents = self._convert(messages)
        config = self._config(system)
        for chunk in client.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        ):
            if chunk.text:
                yield chunk.text

    def chat(self, messages) -> str:
        client = self._client_or_create()
        system, contents = self._convert(messages)
        config = self._config(system)
        resp = client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        return resp.text or ""
