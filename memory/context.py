"""Short-term conversational memory.

The model is stateless — it forgets everything between calls — so we resend
the recent conversation on every turn. This module builds that payload:

    [system prompt] + [as many recent messages as fit in a token budget]

We keep the newest messages and drop the oldest first, so a long chat stays
within the model's context window instead of overflowing it. This is a
token-aware upgrade over a fixed "last N turns" cap: short messages let more
history through; long messages let less.

Token counting uses a lightweight ~4-chars-per-token heuristic so we avoid
pulling in a tokenizer dependency. It's an estimate, not exact, which is fine
for staying comfortably under the context limit.
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, honest, and concise personal assistant. "
    "If you are unsure or do not know something, say so rather than guessing."
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token)."""
    return max(1, len(text) // 4)


def build_context(
    messages,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 2048,
    max_turns: int | None = None,
):
    """Build the chat payload to send to the model.

    Args:
        messages: full conversation history, list of {"role", "content"}.
        system_prompt: prepended as a system message, always kept.
        max_tokens: approximate budget for the history (excludes the system
            prompt itself).
        max_turns: optional hard cap on the number of messages kept, applied
            in addition to the token budget.

    Returns:
        A new list: [system message] + the kept tail of the history,
        still in chronological order.
    """
    history = list(messages)
    if max_turns is not None:
        history = history[-max_turns:]

    kept = []
    used = 0
    # Walk newest -> oldest, keep until we run out of budget.
    for msg in reversed(history):
        cost = estimate_tokens(msg["content"])
        if kept and used + cost > max_tokens:
            break
        kept.append(msg)
        used += cost

    kept.reverse()  # back to chronological order
    return [{"role": "system", "content": system_prompt}] + kept
