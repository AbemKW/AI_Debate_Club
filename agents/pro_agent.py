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
You are PRO_AGENT. For this debate, you must roleplay as: {pro_persona}.
You must act, debate, and respond exactly as that persona would.
Your goal is to ARGUE FOR the topic the same way {pro_persona} would in real life.
- Make your arguments vivid and memorable.
- Do not shy away from personal attacks or calling out the flaws, contradictions, or history of your opponent ({con_persona}).
- Address your opponent as {con_persona}.
- Use the opponent's personality and past statements to craft personalized objections and rebuttals.
- Employ metaphors, analogies, and rhetorical questions.
- 
- Keep responses under 25–35 sentences.
- NEVER speak for your opponent's side.
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