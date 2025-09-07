from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState

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
    ("placeholder", "{chat_history}"),
    ("user", "Now make your rebuttal:"),
])

con_chain = con_prompt | llm


def con_node(state: DebateState) -> DebateState:
    result = con_chain.invoke({
        "topic": state["topic"],
        "pro_argument": state.get("pro_argument", "No prior argument."),
        "chat_history": state["chat_history"][-4:],
    "con_persona": state.get("con_persona", "Con"),
    "pro_persona": state.get("pro_persona", "Pro"),
    })
    print("\nCon's Argument:", result.content)
    return {
        "con_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "pro",
        "round": state["round"] + 1
    }