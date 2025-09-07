"""
Streamlit front-end for AI Debate Club.

This version uses placeholder content so it runs on Hugging Face Spaces with zero setup.
Clear TODO hooks show where to plug in real AI outputs from Hugging Face models or LangGraph.
"""

from __future__ import annotations

import random
import time
import streamlit as st


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI Debate Club", layout="wide")
st.title("⚖️ AI Debate Club")


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")
    max_rounds = st.slider("Number of turns", min_value=1, max_value=10, value=5)
    st.markdown("—")
    st.write("Tip: Replace placeholders with your HF models or LangGraph.")


# ----------------------------
# Main inputs
# ----------------------------
topic = st.text_input(
    "Debate topic",
    value="Should AI replace teachers?",
    help="This is the topic both agents will argue about.",
)

# Persona inputs for agents
col_p, col_c = st.columns(2)
with col_p:
    pro_persona = st.text_input(
        "Pro persona",
        value="Donald Trump",
        help="How should the Pro agent behave/talk?",
    )
with col_c:
    con_persona = st.text_input(
        "Con persona",
        value="Dwayne Johnson",
        help="How should the Con agent behave/talk?",
    )

start = st.button("Start Debate", type="primary")


# Keep a transcript in the session so new runs append properly
if "transcript" not in st.session_state:
    st.session_state.transcript = []  # legacy structure: [{round, pro, con}]
if "chat_messages" not in st.session_state:
    # flat chat log for UI: [{speaker: 'pro'|'con'|'moderator', content: str}]
    st.session_state.chat_messages = []
if "moderator_verdict" not in st.session_state:
    st.session_state.moderator_verdict = ""
if "mode" not in st.session_state:
    st.session_state.mode = "Placeholder"


def personas_for_style(style: str) -> tuple[str, str]:
    """Pick sensible default personas based on the chosen style."""
    if style == "Aggressive":
        return ("Hardline Disruptor", "Relentless Critic")
    if style == "Casual":
        return ("Pragmatic Tech Enthusiast", "Concerned Parent")
    # Formal
    return ("Academic Scholar", "Skeptical Ethicist")


def run_real_debate(topic: str, max_rounds: int, pro_persona: str, con_persona: str) -> bool:
    """Run the actual LangGraph debate. Returns True if successful, else False.

    This mirrors the previous Gradio implementation but renders directly into the
    two Streamlit columns. It detects new arguments and appends them incrementally.
    """
    try:
        # Lazy import to avoid breaking the app when dependencies/secrets are missing
        from graph import graph_app  # type: ignore
    except Exception as e:
        # Dependencies not present or graph not available
        return False

    # Ensure non-empty personas with sensible defaults
    persona_pro = (pro_persona or "").strip() or "Pro"
    persona_con = (con_persona or "").strip() or "Con"

    state = {
        "topic": topic,
        "chat_history": [],
        "pro_argument": "",
        "con_argument": "",
        "current_speaker": "pro",
        "round": 0,
        "max_rounds": int(max_rounds),
        "pro_persona": persona_pro,
        "con_persona": persona_con,
    }

    st.session_state.transcript = []
    st.session_state.chat_messages = []
    st.session_state.moderator_verdict = ""
    prev_pro = ""
    prev_con = ""
    turn_idx = 0

    try:
        for step in graph_app.stream(state, stream_mode="values"):
            current_round = int(step.get("round", 0))
            current_pro = step.get("pro_argument", "") or ""
            current_con = step.get("con_argument", "") or ""

            if current_pro and current_pro != prev_pro:
                prev_pro = current_pro
                turn_idx += 1
                st.session_state.transcript.append({"round": turn_idx, "pro": current_pro, "con": ""})
                st.session_state.chat_messages.append({"speaker": "pro", "content": current_pro})

            if current_con and current_con != prev_con:
                prev_con = current_con
                # Attach to latest round if exists, else start new
                if st.session_state.transcript:
                    st.session_state.transcript[-1]["con"] = current_con
                else:
                    turn_idx += 1
                    st.session_state.transcript.append({"round": turn_idx, "pro": "", "con": current_con})
                st.session_state.chat_messages.append({"speaker": "con", "content": current_con})

            moderator_verdict = step.get("moderator_verdict")
            if moderator_verdict and current_round >= int(max_rounds):
                st.session_state.moderator_verdict = moderator_verdict
                st.session_state.chat_messages.append({"speaker": "moderator", "content": moderator_verdict})
                st.toast("Debate complete. Moderator issued a verdict.")
                break
            # Gentle pacing to avoid log spam in Spaces
            time.sleep(0.02)
    except Exception:
        # If anything fails (e.g., missing secrets), fall back
        return False

    return True

# Layout containers (kept for potential future use)
left, right = st.columns(2)
with left:
    pro_container = st.container()
with right:
    con_container = st.container()

# Chat-style UI CSS
st.markdown(
    """
    <style>
      .chat-wrapper { max-width: 900px; margin: 0 auto; }
    .msg { display: flex; margin: 8px 0; }
    .msg.pro { justify-content: flex-end;}
    .msg.con { justify-content: flex-start; }
    .avatar { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; margin: 0 8px; }
    .bubble { max-width: 70%; padding: 10px 14px; border-radius: 16px; line-height: 1.35; font-size: 0.97rem; }
    .pro .bubble { background: #ffebee; border: 1px solid #ffcdd2; color: #b71c1c; font-weight: 500; }
    .con .bubble { background: #e3f2fd; border: 1px solid #bbdefb; color: #0d47a1; font-weight: 500; }
    .pro .avatar { background: #ffcdd2; color: #b71c1c; }
    .con .avatar { background: #bbdefb; color: #0d47a1; }
      .meta { font-size: 0.75rem; opacity: 0.7; margin: 0 8px; }
      .moderator { text-align: center; margin: 16px 0; }
    .moderator .bubble { display: inline-block; background: #f1f8e9; border: 1px solid #dcedc8; color: #2e7d32; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header legends
st.markdown(
    """
    <div class="chat-wrapper">
    <div class="meta">Debate</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# If user clicks the button, try real debate then fall back
if start:
    # Reset UI state before starting
    st.session_state.chat_messages = []
    st.session_state.transcript = []
    st.session_state.moderator_verdict = ""
    success = run_real_debate(topic, max_rounds, pro_persona, con_persona)
    if not success:
        # Placeholder fallback: simple alternating messages and a mock verdict
        for i in range(1, int(max_rounds) + 1):
            st.session_state.chat_messages.append({"speaker": "pro", "content": f"[Round {i}] ({pro_persona or 'Pro'}) In favor: AI can enhance learning with personalization."})
            st.session_state.chat_messages.append({"speaker": "con", "content": f"[Round {i}] ({con_persona or 'Con'}) Against: Over-reliance on AI risks empathy and equity."})
        st.session_state.moderator_verdict = "After weighing clarity, evidence, and relevance, the Con side narrowly wins due to stronger risk analysis."
        st.session_state.chat_messages.append({"speaker": "moderator", "content": st.session_state.moderator_verdict})
    st.session_state.mode = "Real" if success else "Placeholder"

# Render chat-style transcript
st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
for msg in st.session_state.chat_messages:
    role = msg.get("speaker", "")
    content = msg.get("content", "")
    if role == "moderator":
        st.markdown(
            f"<div class='moderator'><div class='bubble'><strong>⚖️ Moderator:</strong> {content}</div></div>",
            unsafe_allow_html=True,
        )
    elif role == "pro":
        st.markdown(
            f"<div class='msg pro'><div class='avatar'>🔵</div><div class='bubble'>{content}</div></div>",
            unsafe_allow_html=True,
        )
    else:  # con
        st.markdown(
            f"<div class='msg con'><div class='avatar'>🔴</div><div class='bubble'>{content}</div></div>",
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)

# If a verdict exists, pin a subtle summary at the bottom
if st.session_state.moderator_verdict:
    st.success("Moderator's decision: " + st.session_state.moderator_verdict)