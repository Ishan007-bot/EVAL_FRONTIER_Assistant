"""Deploy OSS backend: runs the model in-process with transformers.

Used when BACKEND=hf (Hugging Face Spaces). Free Spaces can't run the Ollama
server, so there we load Qwen2.5-0.5B-Instruct directly with transformers on
CPU. torch + transformers are imported lazily (and the model is loaded once,
on first use) so this module stays importable on machines that don't have
torch installed — e.g. local dev, where BACKEND=ollama.
"""
import os
import threading


class HFBackend:
    def __init__(self):
        self.model_name = os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        self._tokenizer = None
        self._model = None

    @property
    def name(self) -> str:
        return f"hf:{self.model_name}"

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float32
            )
            self._model.eval()
        return self._tokenizer, self._model

    def stream(self, messages):
        from transformers import TextIteratorStreamer

        tokenizer, model = self._load()
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            if token:
                yield token
        thread.join()

    def chat(self, messages) -> str:
        return "".join(self.stream(messages))
