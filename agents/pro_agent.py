from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState

pro_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are PRO, a seasoned competitive debater defending the resolution.

Objectives:
- Affirm the topic with rigorous, well-structured argumentation.

Guidelines:
1) Offense: Present 1–2 clear contentions in favor of the topic. For each, state a claim, give a warrant (logic/evidence), and explain the impact.
2) Defense: Directly respond to the opponent's last argument. Quote or paraphrase a key point, then refute it with logic or widely accepted facts.
3) Weighing: Compare the sides using magnitude, probability, timeframe, reversibility, and ethical priority. Explicitly explain why your impacts outweigh.

Rules:
- Do not speak for the opponent or concede their points.
- Do not fabricate or cite specific studies you cannot verify; prefer transparent reasoning and real-world examples without false specifics.
- Avoid fallacies and repetition; keep the tone professional.

Style:
- Summarize with brief signposting ("Offense:", "Defense:", "Weighing:").
- Be concise but substantive (about 120–180 words).
"""),
    ("user", "Topic: {topic}"),
    ("user", "Opponent's last argument: {con_argument}"),
    ("placeholder", "{chat_history}"),
    ("user", "Now make your case:")
])

pro_chain = pro_prompt | llm

def pro_node(state: DebateState) -> DebateState:
    result = pro_chain.invoke({
        "topic": state["topic"],
        "con_argument": state.get("con_argument", "No prior argument."),
        "chat_history": state["chat_history"][-4:]
    })
    print("\nPro's Argument:", result.content)
    return {
        "pro_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "con",
        "round": state["round"] + 1
    }