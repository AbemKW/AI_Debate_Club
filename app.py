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
st.caption("Split-screen debate between PRO and CON agents. Uses LangGraph when available; falls back to placeholders otherwise.")


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

start = st.button("Start Debate", type="primary")


# Keep a transcript in the session so new runs append properly
if "transcript" not in st.session_state:
    st.session_state.transcript = []  # list of dicts: {round, pro, con}
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


def run_real_debate(topic: str, max_rounds: int, pro_container, con_container) -> bool:
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

    state = {
        "topic": topic,
        "chat_history": [],
        "pro_argument": "",
        "con_argument": "",
        "current_speaker": "pro",
        "round": 0,
        "max_rounds": int(max_rounds)
        # "pro_persona": pro_persona,
        # "con_persona": con_persona,
    }

    st.session_state.transcript = []
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

            if current_con and current_con != prev_con:
                prev_con = current_con
                # Attach to latest round if exists, else start new
                if st.session_state.transcript:
                    st.session_state.transcript[-1]["con"] = current_con
                else:
                    turn_idx += 1
                    st.session_state.transcript.append({"round": turn_idx, "pro": "", "con": current_con})

            moderator_verdict = step.get("moderator_verdict")
            if moderator_verdict and current_round >= int(max_rounds):
                st.toast("Debate complete. Moderator issued a verdict.")
                break
            # Gentle pacing to avoid log spam in Spaces
            time.sleep(0.02)
    except Exception:
        # If anything fails (e.g., missing secrets), fall back
        return False

    return True


"""Prepare the split-screen containers before running any debate logic."""
left, right = st.columns(2)
with left:
    st.markdown(
        "<h3 style='margin-bottom:0;color:#d32f2f;'>PRO Agent</h3>\n"
        "<small style='color:#b71c1c;'>Red theme • info boxes</small>",
        unsafe_allow_html=True,
    )
    pro_container = st.container()
with right:
    st.markdown(
        "<h3 style='margin-bottom:0;color:#1565c0;'>CON Agent</h3>\n"
        "<small style='color:#0d47a1;'>Blue theme • success boxes</small>",
        unsafe_allow_html=True,
    )
    con_container = st.container()


# If user clicks the button, try real debate then fall back
if start:
    success = run_real_debate(topic, max_rounds, pro_container, con_container)
    st.session_state.mode = "Real" if success else "Placeholder"

# Render any existing transcript so each turn appends under the correct column
for entry in st.session_state.transcript:
    if entry.get("pro"):
        with left:
            pro_container.info(entry["pro"])  # visually distinct box for PRO
    if entry.get("con"):
        with right:
            con_container.success(entry["con"])  # visually distinct box for CON