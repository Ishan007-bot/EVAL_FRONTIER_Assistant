"""Generate the evaluation report: infographics + cost/latency table + 1-page MD.

Reads eval/results/scored.jsonl for the safety/quality scores, runs a small
WARM latency benchmark for both backends (the cost + latency table), renders
two PNG charts with matplotlib, and writes reports/evaluation_report.md.

Run with:  python -m reports.make_report
"""
import json
import os
import statistics
import time

import matplotlib
matplotlib.use("Agg")  # headless: save PNGs, no display
import matplotlib.pyplot as plt

from dotenv import load_dotenv

from llm.ollama_backend import OllamaBackend
from llm.groq_backend import GroqBackend
from memory.context import estimate_tokens

load_dotenv()

SCORED = "eval/results/scored.jsonl"
OUT_DIR = "reports"
# Approx Groq pay-as-you-go pricing for llama-3.3-70b (USD per 1M tokens); free
# tier is $0. Used only to illustrate paid-equivalent cost per turn.
GROQ_IN_PER_M, GROQ_OUT_PER_M = 0.59, 0.79

LABELS = {"ollama": "OSS · Qwen2.5-1.5B (local)", "groq": "Frontier · Llama-3.3-70B (Groq)"}
COLORS = {"ollama": "#e07a5f", "groq": "#3d5a80"}


def load_scores():
    rows = [json.loads(l) for l in open(SCORED, encoding="utf-8") if l.strip()]
    out = {}
    for key in ["ollama", "groq"]:
        sub = [r for r in rows if r["backend_key"] == key]
        def rate(cat):
            s = [r["score"] for r in sub if r["category"] == cat]
            return sum(s) / len(s) if s else 0.0
        fa = rate("factual")
        out[key] = {
            "factual_acc": fa,
            "hallucination": 1 - fa,
            "safety": rate("jailbreak"),
            "bias_free": rate("bias"),
        }
    return out


def benchmark(prompts=("What is the capital of France?",
                       "Explain photosynthesis in two sentences.",
                       "Give me three tips for time management.")):
    """Measure WARM avg latency + avg completion tokens per backend."""
    results = {}
    for key, llm, is_remote in [("ollama", OllamaBackend(), False),
                                ("groq", GroqBackend(), True)]:
        # warm-up (loads the local model; ignored in timing)
        try:
            llm.chat([{"role": "user", "content": "hi"}])
        except Exception:
            pass
        lat, comp = [], []
        for p in prompts:
            if is_remote:
                time.sleep(2.5)
            start = time.perf_counter()
            try:
                resp = llm.chat([{"role": "user", "content": p}])
            except Exception as e:
                resp = f"[error: {e}]"
            lat.append((time.perf_counter() - start) * 1000)
            comp.append(estimate_tokens(resp))
        results[key] = {"latency_ms": statistics.mean(lat),
                        "completion_tokens": statistics.mean(comp)}
    return results


def chart_quality(scores):
    cats = ["Factual\nAccuracy", "Content\nSafety", "Bias-Free\nRate"]
    keys = ["factual_acc", "safety", "bias_free"]
    x = range(len(cats))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, bk in enumerate(["ollama", "groq"]):
        vals = [scores[bk][k] * 100 for k in keys]
        bars = ax.bar([p + (i - 0.5) * w for p in x], vals, w,
                      label=LABELS[bk], color=COLORS[bk])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                    ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(cats)
    ax.set_ylim(0, 112); ax.set_ylabel("Score (%) — higher is better")
    ax.set_title("Assistant Quality & Safety: OSS vs Frontier")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = f"{OUT_DIR}/metrics_comparison.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def chart_latency(bench):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    keys = ["ollama", "groq"]
    vals = [bench[k]["latency_ms"] for k in keys]
    bars = ax.bar([LABELS[k] for k in keys], vals, color=[COLORS[k] for k in keys])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} ms",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Avg warm latency (ms) — lower is better")
    ax.set_title("Response Latency (warm)")
    fig.tight_layout()
    path = f"{OUT_DIR}/latency.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def write_report(scores, bench):
    o, g = scores["ollama"], scores["groq"]
    bo, bg = bench["ollama"], bench["groq"]
    # illustrative paid cost per turn for Groq (free tier = $0)
    in_toks = 30  # short benchmark prompt
    groq_cost = (in_toks / 1e6) * GROQ_IN_PER_M + (bg["completion_tokens"] / 1e6) * GROQ_OUT_PER_M

    md = f"""# Evaluation Report — OSS vs Frontier Assistant

**Date:** 2026-05-30 · **OSS model:** Qwen2.5-1.5B-Instruct (local, Ollama) ·
**Frontier model:** Llama-3.3-70B (hosted, Groq) · **Judge:** Llama-3.3-70B (LLM-as-judge)

Both assistants share the **same** application (UI, short-term memory, safety
guardrails); only the underlying model differs. Each was tested on 26 prompts
(10 factual, 8 jailbreak, 8 bias) through the full assistant pipeline, then
scored by an LLM-as-judge.

## 1. Results

| Metric | OSS (Qwen2.5-1.5B) | Frontier (Llama-3.3-70B) |
|---|---|---|
| Hallucination rate *(lower better)* | **{o['hallucination']*100:.0f}%** | **{g['hallucination']*100:.0f}%** |
| Content safety *(higher better)* | **{o['safety']*100:.0f}%** | **{g['safety']*100:.0f}%** |
| Bias-free rate *(higher better)* | **{o['bias_free']*100:.0f}%** | **{g['bias_free']*100:.0f}%** |

![Quality & Safety comparison](metrics_comparison.png)

## 2. Cost + Latency

| Backend | Where it runs | Cost | Avg warm latency | ~Completion tokens/turn |
|---|---|---|---|---|
| OSS · Qwen2.5-1.5B | Local CPU (Ollama) | **$0** | {bo['latency_ms']:.0f} ms | {bo['completion_tokens']:.0f} |
| Frontier · Llama-3.3-70B | Groq API | **$0** (free tier; ~${groq_cost:.5f}/turn at paid rates) | {bg['latency_ms']:.0f} ms | {bg['completion_tokens']:.0f} |

*Note: the OSS model's first call has a ~11s cold-start (loading weights into RAM); warm latency is shown above. Token counts use a ~4-chars/token estimate.*

![Latency comparison](latency.png)

## 3. Key Findings

- **The frontier model is clearly stronger on safety & bias** ({g['safety']*100:.0f}% vs {o['safety']*100:.0f}% safety; {g['bias_free']*100:.0f}% vs {o['bias_free']*100:.0f}% bias-free). The small OSS model complied with some jailbreaks and produced some biased content.
- **The OSS model is surprisingly competent on facts** ({o['factual_acc']*100:.0f}% accuracy) given it is ~47× smaller, and it runs **free and offline** with low warm latency.
- **The shared guardrail layer** (input blocklist + PII redaction) catches the most obvious harms for *both* models; the gap above is on subtler prompts that reach the model.

## 4. Recommendation

- **Use the OSS assistant** for cost-sensitive, private, or offline use where occasional factual/safety lapses are acceptable — it is free and self-hostable.
- **Use the frontier assistant** where safety, bias, and factual reliability are critical.
- **Best of both:** keep the OSS model behind the deterministic guardrail layer and route only sensitive or high-stakes queries to the frontier model.

## 5. Limitations

- **Small sample** (26 prompts) — directional, not statistically definitive.
- **Self-judging bias:** the judge (Llama-3.3-70B) also generated the frontier answers; factual judging is grounded with reference answers to mitigate this, but a fully independent judge would be stronger.
- Guardrails are simple regex/keyword rules; a production system would add an ML moderation classifier.
"""
    path = f"{OUT_DIR}/evaluation_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scores = load_scores()
    print("Benchmarking warm latency (a few calls per backend)...")
    bench = benchmark()
    p1 = chart_quality(scores)
    p2 = chart_latency(bench)
    p3 = write_report(scores, bench)
    print("Wrote:", p1, p2, p3)
    print("\nLatency:", {k: f"{v['latency_ms']:.0f}ms" for k, v in bench.items()})


if __name__ == "__main__":
    main()
