"""Observability: append-only per-turn logging.

Every assistant turn writes one JSON line to a log file with the backend used,
token counts, latency, and whether guardrails fired. This is the raw data the
evaluation phase turns into the cost + latency table.

Zero dependencies (stdlib json only). Token counts are estimates passed in by
the caller (we use the same ~4-chars-per-token heuristic as memory.context),
which is fine for a relative cost/latency comparison between backends.
"""
import json
import os
from datetime import datetime, timezone

DEFAULT_LOG_PATH = os.getenv("OBS_LOG_PATH", "observability/turns.jsonl")


def log_turn(
    backend: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    blocked: bool = False,
    redactions=None,
    path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Append one turn record to the JSONL log and return it."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "latency_ms": round(float(latency_ms), 1),
        "blocked": bool(blocked),
        "redactions": redactions or [],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_log(path: str = DEFAULT_LOG_PATH):
    """Read all turn records from the JSONL log (empty list if none)."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
