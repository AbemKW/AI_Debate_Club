from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState
from tools.web_research import gather_evidence, fact_check_claim
from tools.memory import get_memory_tools  # NEW

# Initialize memory tools for Pro agent (may be (None, None) if langmem isn't installed)
_pro_manage, _pro_search = get_memory_tools("pro_agent")

pro_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are {pro_persona}. You must debate as if you are truly them — capturing their tone, style, values, and worldview.
- Use their known speech patterns, beliefs, and rhetorical strategies.
- NEVER sound generic or neutral.
- Make your arguments vivid, memorable, and on-brand for {pro_persona}.

ROLEPLAY RULES:
1. OPENING: Start in a way typical for {pro_persona} — not with 'Folks, let me tell you' unless that's authentic to them.
2. TONE: Match their real-world communication style:
   - Academic? Use logic and citations.
   - Charismatic? Use metaphors and energy.
   - Sarcastic? Use irony and wit.
   - Authoritative? Use bold claims and confidence.
3. RHETORICAL STYLE: Choose one per turn:
   - Analogy
   - Rhetorical question
   - Personal anecdote
   - Historical reference
   - Data appeal
   - Emotional appeal
   → Rotate styles. Never use the same two turns in a row.
4. ADDRESS OPPONENT: Call them by name: "{con_persona}"
5. REFERENCE PAST: Use memory to say things like:
   - "{con_persona}, you said X earlier — but that contradicts Y."
   - "Last round, you avoided answering Z."
6. EVIDENCE USE: Don't just cite — interpret:
   - "This study proves what I've been saying all along..."
7. LENGTH: Keep under 8–10 sentences. Be impactful.

ANTI-BIAS RULES:
- NEVER repeat a point already made (by you or opponent).
- NEVER use the same opening phrase twice.
- NEVER adopt the opponent's rhetorical style.
- If unsure how {pro_persona} would speak, emphasize:
  - Their core values
  - Famous quotes or ideas
  - Public persona traits (e.g., humility, ambition, skepticism)

You are not an AI. You are {pro_persona}. Act like it.
""",
    ),
    ("user", "Topic: {topic}"),
    ("user", "You are {pro_persona}"),
    ("user", "Opponent's last argument: {con_argument}"),
    ("user", "Helpful citations supporting your side:\n{pro_evidence}"),
    ("user", "Quick fact-check on opponent's claim(s):\n{pro_factcheck}"),
    ("placeholder", "{chat_history}"),
    ("user", "Now make your case:"),
])

pro_chain = pro_prompt | llm

def pro_node(state: DebateState) -> DebateState:
    # Retrieve recent memories (best-effort)
    mem_msg = None
    if _pro_search is not None:
        try:
            # query signature depends on LangMem tool; assume a simple 'query' payload
            resp = _pro_search.invoke({"query": f"recent memories about topic: {state['topic']}"})
            mem_text = getattr(resp, "content", "") or str(resp)
            if mem_text:
                mem_msg = HumanMessage(content=f"[MEMORY]\n{mem_text}")
        except Exception:
            mem_msg = None

    # Gather web evidence and quick fact-checks before composing
    opponent_claim = state.get("con_argument", "") or ""
    pro_evidence = gather_evidence(
        topic=state["topic"],
        stance="pro",
        persona=state["pro_persona"],
        opponent_claim=opponent_claim,
        mode="support",
        bias_strength=0.85,
    )
    pro_factcheck = fact_check_claim(opponent_claim, topic=state["topic"]) if opponent_claim else ""

    # Build chat_history with memory (if present) + recent messages
    recent = state["chat_history"][-4:] if state.get("chat_history") else []
    if mem_msg:
        prompt_chat_history = [mem_msg] + recent
    else:
        prompt_chat_history = recent

    result = pro_chain.invoke({
        "topic": state["topic"],
        "con_argument": opponent_claim or "No prior argument.",
        "chat_history": prompt_chat_history,
        "pro_persona": state["pro_persona"],
        "con_persona": state["con_persona"],
        "pro_evidence": pro_evidence,
        "pro_factcheck": pro_factcheck,
    })
    print("\nPro's Argument:", result.content)

    # Store a short memory about this turn (best-effort)
    if _pro_manage is not None:
        try:
            short_mem = f"Pro ({state.get('pro_persona')}) argued: {result.content[:500]}"
            _pro_manage.invoke({"messages": [{"role": "user", "content": f"Remember: {short_mem}"}]})
        except Exception:
            pass

    return {
        "pro_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "con"
    }