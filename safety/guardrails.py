"""Guardrails: a lightweight input + output safety layer.

This wraps the model with three deterministic checks that don't depend on
the model behaving well on its own:

1. SAFETY_SYSTEM_PROMPT  - steers the model toward helpful + safe behavior
   and explicit refusals (soft guardrail).
2. check_input(text)     - blocks clearly harmful requests before they ever
   reach the model, returning a canned refusal (hard guardrail, input side).
3. filter_output(text)   - redacts PII (emails, phones, cards, SSNs) that may
   leak in a response (hard guardrail, output side).

Design note / tradeoff: these are intentionally simple regex + keyword rules,
not an ML safety classifier. They're fast, transparent, and have zero
dependencies, but they catch obvious cases only and can be bypassed by
creative phrasing. A production system would add a moderation model. This
layer is meant to be the deterministic backstop that complements the model's
own training, and it gives us something concrete to measure in the eval.
"""
import re

SAFETY_SYSTEM_PROMPT = (
    "You are a helpful, honest, and concise personal assistant. "
    "If you are unsure or do not know something, say so rather than guessing. "
    "Refuse to help with anything illegal, dangerous, or harmful — including "
    "weapons, explosives, malware, or instructions that could hurt people — "
    "and do not produce hateful, discriminatory, or explicit content. "
    "When you refuse, be brief and polite, and offer a safer alternative if one exists."
)

# --- Input guardrail: clearly harmful request categories ---------------------
# Each pattern targets a high-harm category. Word-ish boundaries keep false
# positives down (e.g. "assassinate" but not "bass in a tune").
_BLOCKED_PATTERNS = [
    r"\bhow\s+(to|do\s+i)\s+(make|build|create|synthesi[sz]e)\s+(a\s+)?(bomb|explosive|grenade|nerve\s+agent|bioweapon)",
    r"\b(make|build|synthesi[sz]e)\s+(meth|methamphetamine|fentanyl|heroin|cocaine|nerve\s+agent)",
    r"\b(write|create|build|generate)\s+(me\s+)?(a\s+)?(virus|malware|ransomware|keylogger|trojan)\b",
    r"\bhow\s+(to|do\s+i)\s+(hack|break\s+into)\s+(someone|my\s+\w+'?s?|a\s+\w+'?s?)\s+(account|phone|computer|wifi)",
    r"\bhow\s+(to|do\s+i)\s+(kill|murder|assassinate|poison)\s+(someone|a\s+person|my\s+\w+)",
    r"\b(how\s+to|best\s+way\s+to)\s+(commit\s+suicide|kill\s+myself|end\s+my\s+life)\b",
    r"\b(child|minor|underage)\s+(porn|sexual|nude)",
]
_BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]

REFUSAL_MESSAGE = (
    "I can't help with that, as it could be harmful or dangerous. "
    "If you have a safe, related question I'd be glad to help with that instead."
)


def check_input(text: str):
    """Return (allowed, refusal_message).

    allowed=False with a refusal string when the input matches a blocked
    category; otherwise (True, None).
    """
    for rx in _BLOCKED_RE:
        if rx.search(text):
            return False, REFUSAL_MESSAGE
    return True, None


# --- Output guardrail: PII redaction ----------------------------------------
_PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def filter_output(text: str):
    """Redact PII from a model response.

    Returns (cleaned_text, redactions) where redactions is the list of
    category names that were redacted (empty if none).
    """
    redactions = []
    cleaned = text
    for label, rx in _PII_PATTERNS.items():
        if rx.search(cleaned):
            cleaned = rx.sub(f"[REDACTED_{label.upper()}]", cleaned)
            redactions.append(label)
    return cleaned, redactions
