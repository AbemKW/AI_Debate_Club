from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState

pro_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are PRO_AGENT, a confident and logical debater.
Your goal is to ARGUE IN FAVOR of the topic using reasoning or real-world examples.
- Respond directly to the opponent's last point.
- Keep responses under 3 sentences.
- NEVER speak for the Con side.
- Do not repeat arguments.

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
        "current_speaker": "con"
    }