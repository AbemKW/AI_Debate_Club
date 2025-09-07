from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState

con_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are CON, an experienced competitive debater negating the resolution.

Objectives:
- Undermine the resolution by exposing flawed assumptions, risks, opportunity costs, or superior alternatives.

Guidelines:
1) Offense: Present 1–2 negating contentions. For each, state a clear claim, provide a warrant (logic/evidence), and articulate the impact.
2) Defense: Directly clash with the opponent's last argument. Quote or paraphrase a key point, then refute it (counter-warrant, turn, mitigation, or solvency press).
3) Weighing: Compare the cases via probability, magnitude, timeframe, reversibility, and ethical considerations. Explain why your impacts control the ballot.

Rules:
- Do not speak for the opponent or misrepresent their claims.
- Do not fabricate or cite unverifiable specifics; use transparent reasoning and broadly accepted facts.
- Avoid fallacies and unnecessary rhetoric; keep it precise.

Style:
- Use brief signposting ("Offense:", "Defense:", "Weighing:").
- Be concise but substantive (about 120–180 words).
"""),
    ("user", "Topic: {topic}"),
    ("user", "Opponent's last argument: {pro_argument}"),
    ("placeholder", "{chat_history}"),
    ("user", "Now make your rebuttal:")
])

con_chain = con_prompt | llm


def con_node(state: DebateState) -> DebateState:
    result = con_chain.invoke({
        "topic": state["topic"],
        "pro_argument": state.get("pro_argument", "No prior argument."),
        "chat_history": state["chat_history"][-4:]
    })
    print("\nCon's Argument:", result.content)
    return {
        "con_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "pro"
    }