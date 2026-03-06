"""
Streamlit front-end for AI Debate Club.
"""

from __future__ import annotations

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
    max_rounds = st.slider("Number of turns", min_value=1, max_value=10, value=3)
    st.markdown("—")
    st.write("Two AI agents will debate the topic, perform web research, and evaluate each other's claims.")


# ----------------------------
# Main inputs
# ----------------------------
topic = st.text_input(
    "Debate topic",
    value="Should AI replace teachers?",
    help="This is the topic both agents will argue about.",
)

col_p, col_c = st.columns(2)
with col_p:
    pro_persona = st.text_input("Pro persona", value="Donald Trump")
with col_c:
    con_persona = st.text_input("Con persona", value="Dwayne Johnson (The Rock)")

start = st.button("Start Debate", type="primary")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

def run_real_debate(
    topic: str,
    max_rounds: int,
    pro_persona: str,
    con_persona: str,
) -> bool:
    try:
        from graph import graph_app
    except Exception as e:
        st.error(f"Error importing graph: {e}")
        return False

    persona_pro = (pro_persona or "").strip() or "Pro"
    persona_con = (con_persona or "").strip() or "Con"

    state = {
        "topic": topic,
        "chat_history": [],
        "pro_argument": "",
        "con_argument": "",
        "pro_citations": "",
        "con_citations": "",
        "current_speaker": "pro",
        "round": 0,
        "max_rounds": int(max_rounds),
        "pro_persona": persona_pro,
        "con_persona": persona_con,
    }

    st.session_state.chat_messages = []
    
    prev_pro = ""
    prev_con = ""
    turn_idx = 0

    try:
        with st.spinner("Debate is in progress..."):
            for step in graph_app.stream(state, stream_mode="values"):
                current_round = int(step.get("round", 0))
                current_pro = step.get("pro_argument", "") or ""
                current_con = step.get("con_argument", "") or ""
                pro_cit = step.get("pro_citations", "") or ""
                con_cit = step.get("con_citations", "") or ""

                if current_pro and current_pro != prev_pro:
                    prev_pro = current_pro
                    turn_idx += 1
                    msg = {"speaker": "pro", "content": current_pro, "citations": pro_cit, "persona": persona_pro}
                    st.session_state.chat_messages.append(msg)
                    render_single_message(msg)

                if current_con and current_con != prev_con:
                    prev_con = current_con
                    msg = {"speaker": "con", "content": current_con, "citations": con_cit, "persona": persona_con}
                    st.session_state.chat_messages.append(msg)
                    render_single_message(msg)

                moderator_verdict = step.get("moderator_verdict")
                if moderator_verdict and current_round >= int(max_rounds):
                    msg = {"speaker": "moderator", "content": moderator_verdict}
                    st.session_state.chat_messages.append(msg)
                    render_single_message(msg)
                    st.toast("Debate complete!")
                    break
    except Exception as e:
        st.error(f"Debate failed: {e}")
        return False

    return True

def render_single_message(msg):
    role = msg.get("speaker", "")
    content = msg.get("content", "")
    citations = msg.get("citations", "")
    persona = msg.get("persona", "")
    
    if role == "moderator":
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(f"**Moderator Verdict:**\n\n{content}")
    elif role == "pro":
        with st.chat_message("user", avatar="🔵"):
            st.markdown(f"**{persona} (Pro):**")
            st.write(content)
            if citations:
                with st.expander("Research & Citations"):
                    st.markdown(citations)
    else:
        with st.chat_message("user", avatar="🔴"):
            st.markdown(f"**{persona} (Con):**")
            st.write(content)
            if citations:
                with st.expander("Research & Citations"):
                    st.markdown(citations)

chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_messages:
        render_single_message(msg)

if start:
    chat_container.empty()
    with chat_container:
        run_real_debate(topic, max_rounds, pro_persona, con_persona)
