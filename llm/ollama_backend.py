"""Local OSS backend: talks to the Ollama server over HTTP.

Used when BACKEND=ollama (local development). The `ollama` package is
imported lazily so this module never forces an import on the deployed
HF Space (which uses transformers instead).
"""
import os


class OllamaBackend:
    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = None

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def _client_or_create(self):
        if self._client is None:
            import ollama  # lazy import — only loaded for the local backend
            self._client = ollama.Client(host=self.host)
        return self._client

    def stream(self, messages):
        """Yield response text token-by-token for a list of chat messages."""
        client = self._client_or_create()
        for chunk in client.chat(model=self.model, messages=messages, stream=True):
            token = chunk["message"]["content"]
            if token:
                yield token

    def chat(self, messages) -> str:
        """Return the full response as a single string (non-streaming use)."""
        return "".join(self.stream(messages))
