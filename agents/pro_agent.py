from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState

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
- Keep responses under 25–35 sentences.
- NEVER speak for your opponent's side.
""",
    ),
    ("user", "Topic: {topic}"),
    ("user", "Opponent's last argument: {con_argument}"),
    ("placeholder", "{chat_history}"),
    ("user", "Now make your case:"),
])

pro_chain = pro_prompt | llm

def pro_node(state: DebateState) -> DebateState:
    result = pro_chain.invoke({
        "topic": state["topic"],
        "con_argument": state.get("con_argument", "No prior argument."),
        "chat_history": state["chat_history"][-4:],
    "pro_persona": state.get("pro_persona", "Pro"),
    "con_persona": state.get("con_persona", "Con"),
    })
    print("\nPro's Argument:", result.content)
    return {
        "pro_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "con"
    }