"""
Streamlit front-end for AI Debate Club.
"""

from __future__ import annotations

import html as html_lib
from langchain_core.messages import AIMessageChunk

import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI Debate Club", layout="wide", page_icon="⚖️")

# ----------------------------
# Global CSS + Auto-scroll JS
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg:      #0a0a0a;
  --surface: #141414;
  --border:  #222222;
  --border2: #2c2c2c;
  --text:    #e0e0e0;
  --muted:   #5a5a5a;
  --pro:     #4f8ef7;
  --con:     #e05555;
  --mod:     #c9932a;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background-color: var(--bg) !important;
  font-family: 'Inter', 'Segoe UI', sans-serif !important;
  color: var(--text) !important;
}

[data-testid="stHeader"]         { display: none !important; }
[data-testid="stSidebar"]        { background-color: var(--surface) !important;
                                   border-right: 1px solid var(--border) !important; }
[data-testid="stSidebarContent"] { background: transparent !important; }

.block-container { padding-top: 2rem !important; max-width: 860px !important; }

/* ── App header ── */
.app-header {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.9rem;
  margin-bottom: 1.5rem;
}
.app-header .app-title    { font-size: 1rem; font-weight: 700; color: var(--text); }
.app-header .app-subtitle { font-size: 0.8rem; color: var(--muted); }

/* ── Topic ── */
.topic-line {
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.02em;
  margin-bottom: 1.5rem;
  line-height: 1.3;
}
.topic-label {
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  display: block;
  margin-bottom: 0.2rem;
}

/* ── Round divider ── */
.round-divider {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 1.25rem 0 0.75rem;
  color: var(--muted);
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}
.round-divider::before, .round-divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Chat bubbles ── */
.chat-row {
  display: flex;
  margin-bottom: 0.75rem;
}
.chat-row-pro { justify-content: flex-start; }
.chat-row-con { justify-content: flex-end; }

.bubble {
  max-width: 72%;
  padding: 0.85rem 1.1rem;
  background: var(--surface);
  border: 1px solid var(--border);
}
.bubble-pro {
  border-left: 2px solid var(--pro);
  border-radius: 3px 10px 10px 10px;
}
.bubble-con {
  border-right: 2px solid var(--con);
  border-radius: 10px 3px 10px 10px;
}

.bubble-name {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  margin-bottom: 0.45rem;
}
.bubble-name-pro { color: var(--pro); }
.bubble-name-con { color: var(--con); }

.bubble-text {
  font-size: 0.92rem;
  line-height: 1.72;
  color: var(--text);
}

.cursor { opacity: 0.6; }

/* ── Verdict ── */
.verdict-wrap {
  margin-top: 1.75rem;
  padding: 1.1rem 1.2rem;
  background: var(--surface);
  border: 1px solid var(--border2);
  border-top: 2px solid var(--mod);
  border-radius: 4px;
}
.verdict-label {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--mod);
  margin-bottom: 0.55rem;
}
.verdict-text {
  font-size: 0.92rem;
  line-height: 1.72;
  color: #d4a72c;
}

/* ── Status line ── */
.status-line {
  font-size: 0.78rem;
  color: var(--muted);
  margin: 0.4rem 0;
  padding-left: 2px;
}

/* ── Sidebar ── */
.sb-label {
  font-size: 0.62rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin-top: 1.2rem;
  margin-bottom: 0.4rem;
}
.sb-card {
  border-radius: 4px;
  border: 1px solid var(--border);
  padding: 0.5rem 0.75rem;
  margin-bottom: 0.35rem;
  background: var(--bg);
}
.sb-card .sb-role { font-size: 0.6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; }
.sb-card .sb-name { font-size: 0.85rem; color: var(--text); margin-top: 0.1rem; }
.sb-card-pro { border-left: 3px solid var(--pro); }
.sb-card-pro .sb-role { color: var(--pro); }
.sb-card-con { border-left: 3px solid var(--con); }
.sb-card-con .sb-role { color: var(--con); }

/* ── Widget overrides ── */
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
.stButton > button[kind="primary"] {
  background: var(--pro) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 600 !important;
  border-radius: 4px !important;
}
.stButton > button[kind="primary"]:hover { background: #3a78e0 !important; }
.stProgress > div > div > div > div { background: var(--pro) !important; }
</style>

<script>
(function() {
  const target = document.querySelector('.stMain [role="main"]');
  if (!target) return;
  
  const observer = new MutationObserver(() => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  });
  
  observer.observe(target, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)


# ----------------------------
# Helpers
# ----------------------------
def _e(s: str) -> str:
    return html_lib.escape(s or "")


def bubble_html(content: str, role: str, persona: str, streaming: bool = False) -> str:
    safe = _e(content).replace("\n", "<br>")
    cursor = '<span class="cursor">▌</span>' if streaming else ""
    if role == "pro":
        return (
            f'<div class="chat-row chat-row-pro">'
            f'<div class="bubble bubble-pro">'
            f'<div class="bubble-name bubble-name-pro">{_e(persona)}</div>'
            f'<div class="bubble-text">{safe}{cursor}</div>'
            f'</div></div>'
        )
    elif role == "con":
        return (
            f'<div class="chat-row chat-row-con">'
            f'<div class="bubble bubble-con">'
            f'<div class="bubble-name bubble-name-con">{_e(persona)}</div>'
            f'<div class="bubble-text">{safe}{cursor}</div>'
            f'</div></div>'
        )
    elif role == "moderator":
        return (
            f'<div class="verdict-wrap">'
            f'<div class="verdict-label">⚖️ &nbsp; Moderator Verdict</div>'
            f'<div class="verdict-text">{safe}{cursor}</div>'
            f'</div>'
        )
    return ""


def round_divider_html(n: int) -> str:
    return f'<div class="round-divider">Round {n}</div>'


def render_history(messages: list[dict]) -> None:
    """Replay stored messages (non-streaming)."""
    last_round = 0
    for msg in messages:
        r = msg.get("round", 0)
        if msg["speaker"] != "moderator" and r != last_round:
            st.markdown(round_divider_html(r), unsafe_allow_html=True)
            last_round = r
        st.markdown(bubble_html(msg["content"], msg["speaker"], msg.get("persona", "")), unsafe_allow_html=True)


# ----------------------------
# Sidebar — block 1
# ----------------------------
with st.sidebar:
    st.markdown('<div style="font-size:0.9rem;font-weight:700;color:#e0e0e0;padding:0.25rem 0 0.75rem;">AI Debate Club</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Rounds</div>', unsafe_allow_html=True)
    max_rounds = st.slider("Rounds", min_value=1, max_value=10, value=3, label_visibility="collapsed")
    st.markdown('<div class="sb-label">About</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.75rem;color:#5a5a5a;line-height:1.6;">Two AI agents debate in persona, then a neutral moderator evaluates and declares a winner.</div>', unsafe_allow_html=True)


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
    st.markdown('<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#4f8ef7;margin-bottom:4px;">PRO persona</div>', unsafe_allow_html=True)
    pro_persona = st.text_input("Pro persona", value="Donald Trump", label_visibility="collapsed")
with col_c:
    st.markdown('<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#e05555;margin-bottom:4px;">CON persona</div>', unsafe_allow_html=True)
    con_persona = st.text_input("Con persona", value="Dwayne Johnson (The Rock)", label_visibility="collapsed")

start = st.button("Start Debate", type="primary", use_container_width=True)

# ----------------------------
# Sidebar — block 2: live persona cards
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sb-label">Debaters</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-card sb-card-pro">
      <div class="sb-role">Pro</div>
      <div class="sb-name">{_e(pro_persona or "Pro")}</div>
    </div>
    <div class="sb-card sb-card-con">
      <div class="sb-role">Con</div>
      <div class="sb-name">{_e(con_persona or "Con")}</div>
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
# Debate runner — streams tokens live
# ----------------------------
def run_real_debate(topic: str, max_rounds: int, pro_persona: str, con_persona: str) -> bool:
    try:
        from graph import graph_app
    except Exception as e:
        st.error(f"Error importing graph: {e}")
        return False

    persona_pro = (pro_persona or "").strip() or "Pro"
    persona_con = (con_persona or "").strip() or "Con"

    st.session_state.chat_messages = []
    st.session_state.debate_personas = {"pro": persona_pro, "con": persona_con}

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

    progress_bar = st.progress(0)
    status      = st.empty()

    current_node  = None
    current_text  = ""
    current_ph    = None   # st.empty() for the active streaming bubble
    pro_turn      = 0
    con_turn      = 0

    def finish_turn():
        """Persist the completed turn to session state and finalize its placeholder."""
        nonlocal current_text, current_ph
        if not current_node or not current_text:
            return
        role = current_node
        persona = persona_pro if role == "pro" else persona_con if role == "con" else ""
        round_num = pro_turn if role == "pro" else con_turn
        st.session_state.chat_messages.append({
            "speaker": role,
            "content": current_text,
            "persona": persona,
            "round": round_num,
        })
        if current_ph:
            current_ph.markdown(bubble_html(current_text, role, persona, streaming=False), unsafe_allow_html=True)

    try:
        for chunk, metadata in graph_app.stream(state, stream_mode="messages"):
            node = metadata.get("langgraph_node", "")
            if node not in ("pro", "con", "moderator"):
                continue
            if not isinstance(chunk, AIMessageChunk):
                continue
            token = chunk.content
            if not isinstance(token, str) or not token:
                continue

            # ── Node transition ──────────────────────────────────────────
            if node != current_node:
                finish_turn()

                current_node = node
                current_text = ""

                if node == "pro":
                    pro_turn += 1
                    # Round divider before each new pro turn (except first)
                    st.markdown(round_divider_html(pro_turn), unsafe_allow_html=True)
                    progress_bar.progress(max((con_turn) / max_rounds, 0))
                    status.markdown(f'<div class="status-line">{_e(persona_pro)} is speaking...</div>', unsafe_allow_html=True)

                elif node == "con":
                    con_turn += 1
                    status.markdown(f'<div class="status-line">{_e(persona_con)} is speaking...</div>', unsafe_allow_html=True)

                elif node == "moderator":
                    progress_bar.progress(1.0)
                    status.markdown('<div class="status-line">Moderator is deliberating...</div>', unsafe_allow_html=True)
                    st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)

                current_ph = st.empty()

            # ── Stream token into placeholder ────────────────────────────
            current_text += token
            persona = persona_pro if current_node == "pro" else persona_con if current_node == "con" else ""
            current_ph.markdown(
                bubble_html(current_text, current_node, persona, streaming=True),
                unsafe_allow_html=True,
            )

        # Finalize the last turn
        finish_turn()
        status.empty()
        st.toast("Debate complete.", icon="⚖️")

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

# Replay previous session
if st.session_state.chat_messages:
    render_history(
        st.session_state.chat_messages,
    )

if start:
    run_real_debate(topic, max_rounds, pro_persona, con_persona)
