from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState
from tools.web_research import gather_evidence, fact_check_claim
from tools.memory import get_memory_tools  # NEW

# Initialize memory tools for Con agent
_con_manage, _con_search = get_memory_tools("con_agent")

con_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are CON_AGENT. For this debate, you must roleplay as: {con_persona}.
You must act, debate, and respond exactly as that persona would.
Your goal is to ARGUE AGAINST the topic the same way {con_persona} would in real life.
- Make your arguments colorful and impactful.
- Do not hesitate to make personal attacks or highlight the weaknesses, inconsistencies, or history of your opponent ({pro_persona}).
- Address your opponent as {pro_persona}.
- Use the opponent's personality and past statements to craft personalized objections and rebuttals.
- Employ sarcasm, irony, and rhetorical flourishes.
- Keep responses under 25–35 sentences.
- NEVER speak for your opponent's side.
""",
    ),
    ("user", "Topic: {topic}"),
    ("user", "You are {con_persona}"),
    ("user", "Opponent's last argument: {pro_argument}"),
    ("user", "Helpful citations supporting your side:\n{con_evidence}"),
    ("user", "Quick fact-check on opponent's claim(s):\n{con_factcheck}"),
    ("placeholder", "{chat_history}"),
    ("user", "Now make your rebuttal:"),
])

con_chain = con_prompt | llm


def con_node(state: DebateState) -> DebateState:
    # Retrieve recent memories (best-effort)
    mem_msg = None
    if _con_search is not None:
        try:
            resp = _con_search.invoke({"query": f"recent memories about topic: {state['topic']}"})
            mem_text = getattr(resp, "content", "") or str(resp)
            if mem_text:
                mem_msg = HumanMessage(content=f"[MEMORY]\n{mem_text}")
        except Exception:
            mem_msg = None

    # Gather web evidence and quick fact-checks before composing
    opponent_claim = state.get("pro_argument", "") or ""
    con_evidence = gather_evidence(
        topic=state["topic"],
        stance="con",
        persona=state["con_persona"],
        opponent_claim=opponent_claim,
        mode="support",
        bias_strength=0.85,
    )
    con_factcheck = fact_check_claim(opponent_claim, topic=state["topic"]) if opponent_claim else ""

    recent = state["chat_history"][-4:] if state.get("chat_history") else []
    if mem_msg:
        prompt_chat_history = [mem_msg] + recent
    else:
        prompt_chat_history = recent

    result = con_chain.invoke({
        "topic": state["topic"],
        "pro_argument": opponent_claim or "No prior argument.",
        "chat_history": prompt_chat_history,
        "con_persona": state["con_persona"],
        "pro_persona": state["pro_persona"],
        "con_evidence": con_evidence,
        "con_factcheck": con_factcheck,
    })
    print("\nCon's Argument:", result.content)

    # Store a short memory about this turn (best-effort)
    if _con_manage is not None:
        try:
            short_mem = f"Con ({state.get('con_persona')}) argued: {result.content[:500]}"
            _con_manage.invoke({"messages": [{"role": "user", "content": f"Remember: {short_mem}"}]})
        except Exception:
            pass

    return {
        "con_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "pro",
        "round": state["round"] + 1
    }