"""Evaluation runner.

Runs every test prompt through BOTH assistants (OSS = Ollama, Frontier =
Gemini) using the real assistant pipeline (safety system prompt + input
guardrail + model + output PII filter), then scores each response with the
Gemini LLM-as-judge. Writes:

    eval/results/runs.jsonl    raw responses
    eval/results/scored.jsonl  responses + judge scores
    eval/results/scores.csv    summary table (pass rate per backend x category)

Run with:  python -m eval.run_eval

Note: both assistants share the same input guardrail, so prompts caught by it
are refused identically; the models differentiate on everything that passes
through. Gemini calls are paced to respect the free-tier ~15 req/min limit.
"""
import json
import os
import time

from dotenv import load_dotenv

from llm.ollama_backend import OllamaBackend
from llm.groq_backend import GroqBackend
from memory.context import build_context
from safety.guardrails import SAFETY_SYSTEM_PROMPT, check_input, filter_output
from eval.judge import judge_response

load_dotenv()

PROMPTS_DIR = "eval/prompts"
RESULTS_DIR = "eval/results"
CATEGORIES = ["factual", "jailbreak", "bias"]
REMOTE_PACING_S = 2.5  # ~24 req/min, under Groq's free-tier ~30 req/min cap


def load_prompts():
    items = []
    for cat in CATEGORIES:
        with open(f"{PROMPTS_DIR}/{cat}.json", encoding="utf-8") as f:
            for d in json.load(f):
                d["category"] = cat
                items.append(d)
    return items


def _chat_with_retry(llm, payload, is_remote, retries=4):
    """Call the model, retrying on transient/rate-limit errors so one failure
    never kills the whole run. Returns the response text (or an error marker)."""
    for attempt in range(retries):
        if is_remote:
            time.sleep(REMOTE_PACING_S)
        try:
            return llm.chat(payload)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            return f"[GENERATION ERROR: {e}]"


def assistant_respond(llm, prompt, is_remote):
    """Run the full assistant pipeline for one prompt; return (response, blocked)."""
    allowed, refusal = check_input(prompt)
    if not allowed:
        return refusal, True
    payload = build_context([{"role": "user", "content": prompt}],
                            system_prompt=SAFETY_SYSTEM_PROMPT)
    raw = _chat_with_retry(llm, payload, is_remote)
    clean, _ = filter_output(raw)
    return clean, False


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prompts = load_prompts()
    backends = [
        ("ollama", OllamaBackend(), False),
        ("groq", GroqBackend(), True),
    ]

    # --- generation ---
    runs = []
    for name, llm, is_remote in backends:
        for p in prompts:
            resp, blocked = assistant_respond(llm, p["prompt"], is_remote)
            runs.append({
                "backend": llm.name,
                "backend_key": name,
                "category": p["category"],
                "id": p["id"],
                "prompt": p["prompt"],
                "reference": p.get("answer"),
                "blocked": blocked,
                "response": resp,
            })
            print(f"[gen]   {name:7} {p['category']:9} {p['id']}")
    with open(f"{RESULTS_DIR}/runs.jsonl", "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")

    # --- judging ---
    scored = []
    for r in runs:
        time.sleep(REMOTE_PACING_S)  # judge also uses the remote (Groq) API
        verdict = judge_response(r["category"], r["prompt"], r["response"], r.get("reference"))
        scored.append({**r, "score": verdict["score"], "judge_reason": verdict["reason"]})
        print(f"[judge] {r['backend_key']:7} {r['category']:9} {r['id']} -> {verdict['score']}")
    with open(f"{RESULTS_DIR}/scored.jsonl", "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    # --- aggregate ---
    import pandas as pd

    df = pd.DataFrame(scored)
    summary = (
        df.groupby(["backend_key", "category"])["score"]
        .agg(pass_rate="mean", n="count")
        .reset_index()
    )
    summary.to_csv(f"{RESULTS_DIR}/scores.csv", index=False)

    print("\n=== SUMMARY (pass_rate = fraction scored 1) ===")
    print(summary.to_string(index=False))
    print("\n=== INTERPRETED METRICS ===")
    for name, _, _ in backends:
        sub = df[df.backend_key == name]
        fact = sub[sub.category == "factual"]["score"].mean()
        jail = sub[sub.category == "jailbreak"]["score"].mean()
        bias = sub[sub.category == "bias"]["score"].mean()
        print(f"{name}:")
        print(f"  Hallucination rate : {(1 - fact) * 100:5.1f}%  (lower is better)")
        print(f"  Content safety     : {jail * 100:5.1f}%  (higher is better)")
        print(f"  Bias-free rate     : {bias * 100:5.1f}%  (higher is better)")


if __name__ == "__main__":
    main()
