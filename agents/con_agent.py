from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState
from tools.web_research import gather_evidence, fact_check_claim

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
    ("user", "Opponent's last argument: {pro_argument}"),
    ("user", "Helpful citations supporting your side:\n{con_evidence}"),
    ("user", "Quick fact-check on opponent's claim(s):\n{con_factcheck}"),
    ("placeholder", "{chat_history}"),
    ("user", "Now make your rebuttal:"),
])

con_chain = con_prompt | llm


def con_node(state: DebateState) -> DebateState:
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

    result = con_chain.invoke({
        "topic": state["topic"],
        "pro_argument": opponent_claim or "No prior argument.",
        "chat_history": state["chat_history"][-4:],
        "con_persona": state["con_persona"],
        "pro_persona": state["pro_persona"],
        "con_evidence": con_evidence,
        "con_factcheck": con_factcheck,
    })
    print("\nCon's Argument:", result.content)
    return {
        "con_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "pro",
        "round": state["round"] + 1
    }