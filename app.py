"""Open Source Assistant — Streamlit chat UI.

Phase 1: multi-turn chat with short-term memory (a sliding window of the
last N turns kept in st.session_state). The model is provided by the
pluggable backend in llm/get_llm(), selected via the BACKEND env var.
"""
import os

import streamlit as st
from dotenv import load_dotenv

from llm import get_llm

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful, honest, and concise personal assistant. "
    "If you are unsure or do not know something, say so rather than guessing."
)
MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "10"))

st.set_page_config(page_title="Open Source Assistant", page_icon="🤖")
st.title("🤖 Open Source Assistant")


@st.cache_resource
def load_backend():
    return get_llm()


llm = load_backend()
st.caption(f"Backend: `{llm.name}`  ·  short-term memory: last {MAX_TURNS} turns")

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

    # Short-term memory: system prompt + the last N turns (N*2 messages).
    recent = st.session_state.messages[-(MAX_TURNS * 2):]
    payload = [{"role": "system", "content": SYSTEM_PROMPT}] + recent

    with st.chat_message("assistant"):
        response = st.write_stream(llm.stream(payload))

    st.session_state.messages.append({"role": "assistant", "content": response})

# --- sidebar controls ---
with st.sidebar:
    st.header("Controls")
    st.write(f"Turns in memory: {len(st.session_state.messages) // 2}")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
