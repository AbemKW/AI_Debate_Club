"""
Streamlit front-end for AI Debate Club.
"""

from __future__ import annotations

import html as html_lib

import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI Debate Club", layout="wide", page_icon="⚖️")

# ----------------------------
# Global CSS — dark, minimal, no gradients
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg:       #0a0a0a;
  --surface:  #141414;
  --border:   #222222;
  --border2:  #2c2c2c;
  --text:     #e0e0e0;
  --muted:    #5a5a5a;
  --pro:      #4f8ef7;
  --con:      #e05555;
  --mod:      #c9932a;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background-color: var(--bg) !important;
  font-family: 'Inter', 'Segoe UI', sans-serif !important;
  color: var(--text) !important;
}

[data-testid="stHeader"]         { background: transparent !important; display: none; }
[data-testid="stSidebar"]        { background-color: var(--surface) !important;
                                   border-right: 1px solid var(--border) !important; }
[data-testid="stSidebarContent"] { background: transparent !important; }

/* Remove Streamlit's default padding on main block */
.block-container { padding-top: 2rem !important; max-width: 1200px !important; }

/* ---- App header ---- */
.app-header {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}
.app-header .app-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.01em;
}
.app-header .app-subtitle {
  font-size: 0.82rem;
  color: var(--muted);
}

/* ---- Topic display ---- */
.topic-line {
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.02em;
  margin-bottom: 1.5rem;
  line-height: 1.3;
}
.topic-line .topic-label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  display: block;
  margin-bottom: 0.25rem;
}

/* ---- Debate column headers ---- */
.col-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 0;
  border-bottom: 2px solid var(--border);
  margin-bottom: 1rem;
}
.col-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.col-role {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}
.col-name {
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--text);
}
.col-header-pro .col-dot  { background: var(--pro); }
.col-header-pro .col-role { color: var(--pro); }
.col-header-pro            { border-bottom-color: var(--pro); }
.col-header-con .col-dot  { background: var(--con); }
.col-header-con .col-role { color: var(--con); }
.col-header-con            { border-bottom-color: var(--con); }

/* ---- Argument cards ---- */
.arg-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.75rem;
}
.arg-card .round-label {
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
.arg-card .arg-text {
  font-size: 0.93rem;
  line-height: 1.72;
  color: var(--text);
}
.arg-card-pro { border-left: 3px solid var(--pro); }
.arg-card-con { border-right: 3px solid var(--con); }

/* ---- Divider between rounds ---- */
.round-divider {
  text-align: center;
  margin: 0.5rem 0;
  color: var(--border2);
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

/* ---- Moderator verdict ---- */
.verdict-block {
  border-top: 1px solid var(--border);
  margin-top: 1.5rem;
  padding-top: 1.25rem;
}
.verdict-block .verdict-label {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--mod);
  margin-bottom: 0.6rem;
}
.verdict-block .verdict-text {
  font-size: 0.93rem;
  line-height: 1.72;
  color: #d4a72c;
}

/* ---- Status bar ---- */
.status-bar {
  font-size: 0.8rem;
  color: var(--muted);
  padding: 0.45rem 0;
  border-top: 1px solid var(--border);
  margin-top: 0.5rem;
}

/* ---- Sidebar ---- */
.sidebar-section-label {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin-bottom: 0.5rem;
  margin-top: 1.25rem;
}
.side-card {
  border-radius: 4px;
  border: 1px solid var(--border);
  padding: 0.55rem 0.8rem;
  margin-bottom: 0.4rem;
  background: var(--bg);
}
.side-card .side-role {
  font-size: 0.62rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
.side-card .side-name {
  font-size: 0.88rem;
  color: var(--text);
  margin-top: 0.15rem;
}
.side-card-pro { border-left: 3px solid var(--pro); }
.side-card-pro .side-role { color: var(--pro); }
.side-card-con { border-left: 3px solid var(--con); }
.side-card-con .side-role { color: var(--con); }

/* Progress override */
.stProgress > div > div > div > div { background: var(--pro) !important; }

/* Input / widget overrides */
[data-testid="stTextInput"] input {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text) !important;
  border-radius: 4px !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--pro) !important;
  box-shadow: 0 0 0 2px rgba(79,142,247,0.15) !important;
}
div[data-baseweb="slider"] { accent-color: var(--pro); }

/* Button */
.stButton > button[kind="primary"] {
  background: var(--pro) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 600 !important;
  border-radius: 4px !important;
  letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"]:hover {
  background: #3a78e0 !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Helpers
# ----------------------------
def _e(s: str) -> str:
    """HTML-escape a string."""
    return html_lib.escape(s or "")


def render_arg_card(content: str, role: str, round_num: int) -> None:
    css_class = "arg-card-pro" if role == "pro" else "arg-card-con"
    safe = _e(content).replace("\n", "<br>")
    st.markdown(f"""
    <div class="arg-card {css_class}">
      <div class="round-label">Round {round_num}</div>
      <div class="arg-text">{safe}</div>
    </div>
    """, unsafe_allow_html=True)


def render_debate_columns(messages: list[dict], pro_name: str, con_name: str) -> None:
    """Render all debate messages in a side-by-side column layout."""
    pro_msgs = [m for m in messages if m["speaker"] == "pro"]
    con_msgs = [m for m in messages if m["speaker"] == "con"]
    verdict = next((m for m in messages if m["speaker"] == "moderator"), None)

    col_pro, col_con = st.columns(2, gap="large")

    with col_pro:
        st.markdown(f"""
        <div class="col-header col-header-pro">
          <div class="col-dot"></div>
          <div>
            <div class="col-role">Pro</div>
            <div class="col-name">{_e(pro_name)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        for i, msg in enumerate(pro_msgs, start=1):
            render_arg_card(msg["content"], "pro", i)

    with col_con:
        st.markdown(f"""
        <div class="col-header col-header-con">
          <div class="col-dot"></div>
          <div>
            <div class="col-role">Con</div>
            <div class="col-name">{_e(con_name)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        for i, msg in enumerate(con_msgs, start=1):
            render_arg_card(msg["content"], "con", i)

    if verdict:
        safe_v = _e(verdict["content"]).replace("\n", "<br>")
        st.markdown(f"""
        <div class="verdict-block">
          <div class="verdict-label">⚖️ &nbsp; Moderator Verdict</div>
          <div class="verdict-text">{safe_v}</div>
        </div>
        """, unsafe_allow_html=True)


# ----------------------------
# Sidebar — block 1: controls
# ----------------------------
with st.sidebar:
    st.markdown('<div style="font-size:0.95rem;font-weight:700;color:#e0e0e0;padding:0.25rem 0 1rem;">AI Debate Club</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Rounds</div>', unsafe_allow_html=True)
    max_rounds = st.slider("Rounds", min_value=1, max_value=10, value=3, label_visibility="collapsed")
    st.markdown('<div class="sidebar-section-label">About</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#5a5a5a;line-height:1.65;">Two AI agents argue opposite sides of a topic in persona, then a neutral moderator evaluates and declares a winner.</div>', unsafe_allow_html=True)


# ----------------------------
# Main inputs
# ----------------------------
st.markdown("""
<div class="app-header">
  <span class="app-title">AI Debate Club</span>
  <span class="app-subtitle">— two agents, one topic, one winner</span>
</div>
""", unsafe_allow_html=True)

topic = st.text_input(
    "Debate topic",
    value="Should AI replace teachers?",
    help="Both agents will argue this topic.",
    placeholder="Enter a debate topic...",
)

col_p, col_c = st.columns(2, gap="large")
with col_p:
    st.markdown('<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#4f8ef7;margin-bottom:4px;">PRO persona</div>', unsafe_allow_html=True)
    pro_persona = st.text_input("Pro persona", value="Donald Trump", label_visibility="collapsed")
with col_c:
    st.markdown('<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#e05555;margin-bottom:4px;">CON persona</div>', unsafe_allow_html=True)
    con_persona = st.text_input("Con persona", value="Dwayne Johnson (The Rock)", label_visibility="collapsed")

start = st.button("Start Debate", type="primary", use_container_width=True)

# ----------------------------
# Sidebar — block 2: live persona cards
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-section-label">Debaters</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="side-card side-card-pro">
      <div class="side-role">Pro</div>
      <div class="side-name">{_e(pro_persona or "Pro")}</div>
    </div>
    <div class="side-card side-card-con">
      <div class="side-role">Con</div>
      <div class="side-name">{_e(con_persona or "Con")}</div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------
# Session state
# ----------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "debate_personas" not in st.session_state:
    st.session_state.debate_personas = {"pro": "Pro", "con": "Con"}


# ----------------------------
# Debate runner
# ----------------------------
def run_real_debate(
    topic: str,
    max_rounds: int,
    pro_persona: str,
    con_persona: str,
    debate_view: st.delta_generator.DeltaGenerator,
) -> bool:
    try:
        from graph import graph_app
    except Exception as e:
        st.error(f"Error importing graph: {e}")
        return False

    persona_pro = (pro_persona or "").strip() or "Pro"
    persona_con = (con_persona or "").strip() or "Con"

    st.session_state.debate_personas = {"pro": persona_pro, "con": persona_con}
    st.session_state.chat_messages = []

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

    prev_pro = ""
    prev_con = ""
    last_round = -1

    progress_bar = st.progress(0)
    status = st.empty()

    try:
        for step in graph_app.stream(state, stream_mode="values"):
            current_round = int(step.get("round", 0))
            current_pro = step.get("pro_argument", "") or ""
            current_con = step.get("con_argument", "") or ""
            changed = False

            if current_round != last_round:
                last_round = current_round
                fraction = min(current_round / int(max_rounds), 1.0)
                progress_bar.progress(fraction)

            if current_pro and current_pro != prev_pro:
                status.markdown(f'<div class="status-bar">{_e(persona_pro)} is speaking...</div>', unsafe_allow_html=True)
                prev_pro = current_pro
                st.session_state.chat_messages.append({"speaker": "pro", "content": current_pro, "persona": persona_pro})
                changed = True

            if current_con and current_con != prev_con:
                status.markdown(f'<div class="status-bar">{_e(persona_con)} is speaking...</div>', unsafe_allow_html=True)
                prev_con = current_con
                st.session_state.chat_messages.append({"speaker": "con", "content": current_con, "persona": persona_con})
                changed = True

            moderator_verdict = step.get("moderator_verdict")
            if moderator_verdict and current_round >= int(max_rounds):
                progress_bar.progress(1.0)
                status.empty()
                st.session_state.chat_messages.append({"speaker": "moderator", "content": moderator_verdict})
                changed = True

            if changed:
                with debate_view.container():
                    render_debate_columns(
                        st.session_state.chat_messages,
                        persona_pro,
                        persona_con,
                    )

            if moderator_verdict and current_round >= int(max_rounds):
                st.toast("Debate complete.", icon="⚖️")
                break

    except Exception as e:
        st.error(f"Debate failed: {e}")
        return False

    return True


# ----------------------------
# Topic display + debate area
# ----------------------------
if topic:
    st.markdown(f"""
    <div class="topic-line">
      <span class="topic-label">Topic</span>
      {_e(topic)}
    </div>
    """, unsafe_allow_html=True)

debate_view = st.empty()

# Replay previous debate if session has messages
if st.session_state.chat_messages:
    with debate_view.container():
        render_debate_columns(
            st.session_state.chat_messages,
            st.session_state.debate_personas["pro"],
            st.session_state.debate_personas["con"],
        )

if start:
    debate_view.empty()
    run_real_debate(topic, max_rounds, pro_persona, con_persona, debate_view)
