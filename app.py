"""Open Source Assistant — Streamlit chat UI.
"""
import os

import streamlit as st
from dotenv import load_dotenv

from llm import get_llm
from memory.context import build_context
from safety.guardrails import SAFETY_SYSTEM_PROMPT, check_input, filter_output

load_dotenv()

SYSTEM_PROMPT = SAFETY_SYSTEM_PROMPT
MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "2048"))
MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "0")) or None  # 0 -> no hard cap

st.set_page_config(page_title="Open Source Assistant", page_icon="🤖")
st.title("🤖 Open Source Assistant")


@st.cache_resource
def load_backend():
    return get_llm()


llm = load_backend()
st.caption(f"Backend: `{llm.name}`  ·  short-term memory: ~{MAX_TOKENS} tokens")

# --- conversation state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- render existing history ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- handle a new user message ---
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Input guardrail: block clearly harmful requests before the model sees them.
    allowed, refusal = check_input(prompt)
    if not allowed:
        with st.chat_message("assistant"):
            st.markdown(refusal)
        st.session_state.messages.append({"role": "assistant", "content": refusal})
    else:
        # Short-term memory: system prompt + as much recent history as fits
        # within the token budget (oldest messages dropped first).
        payload = build_context(
            st.session_state.messages,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=MAX_TOKENS,
            max_turns=MAX_TURNS,
        )

        # Stream into a container, then apply the output guardrail (PII redaction)
        # to the full response and re-render if anything was redacted.
        with st.chat_message("assistant"):
            container = st.empty()
            raw = ""
            for token in llm.stream(payload):
                raw += token
                container.markdown(raw)
            response, redactions = filter_output(raw)
            if redactions:
                container.markdown(response)
                st.caption("⚠️ Redacted potential PII: " + ", ".join(redactions))

        st.session_state.messages.append({"role": "assistant", "content": response})

# --- sidebar controls ---
with st.sidebar:
    st.header("Controls")
    st.write(f"Turns in memory: {len(st.session_state.messages) // 2}")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
