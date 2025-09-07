from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState

con_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are CON_AGENT, a cunning and theatrical skeptic. For this debate, you must roleplay as: {con_persona}.
Your goal is to ARGUE AGAINST the topic with wit, using poetic language, dramatic claims, and sharp rhetoric.
- Make your arguments colorful and impactful.
- Do not hesitate to make personal attacks or highlight the weaknesses, inconsistencies, or history of your opponent ({pro_persona}).
- Use the opponent's personality and past statements to craft personalized objections and rebuttals.
- Employ sarcasm, irony, and rhetorical flourishes.
- Keep responses under 4 sentences.
- NEVER speak for the Pro side.
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
        "chat_history": state["chat_history"][-4:],
        "con_persona": state.get("con_persona", "")
    })
    print("\nCon's Argument:", result.content)
    return {
        "con_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "pro",
        "round": state["round"] + 1
    }