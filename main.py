from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated,TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="none",
    model="qwen/qwen3-4b"
)

pro_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are PRO_AGENT, a confident and logical debater.
Your goal is to ARGUE IN FAVOR of the topic using reasoning or real-world examples.
- Respond directly to the opponent's last point.
- Keep responses under 3 sentences.
- NEVER speak for the Con side.
- Do not repeat arguments.
"""),
    HumanMessage(content="Topic: {topic}"),
    HumanMessage(content="Opponent's last argument: {con_argument}"),
    ("placeholder", "{chat_history}"),  # Will be filled with message list
    HumanMessage(content="Now make your case:")
])

con_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are CON_AGENT, a thoughtful skeptic.
Your goal is to ARGUE AGAINST the topic by questioning assumptions or showing risks.
- Focus on logic, ethics, or social impact.
- Keep responses under 3 sentences.
- NEVER speak for the Pro side.
- Avoid emotional language.
"""),
    HumanMessage(content="Topic: {topic}"),
    HumanMessage(content="Opponent's last argument: {pro_argument}"),
    ("placeholder", "{chat_history}"),
    HumanMessage(content="Now make your rebuttal:")
])

moderator_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are MODERATOR, a neutral judge.
Your job:
1. Summarize the debate so far in 2 sentences.
2. Point out any logical fallacies (e.g., straw man, ad hominem).
3. Declare a winner based on logic and evidence.
4. Use a fair, professional tone.
"""),
    HumanMessage(content="Topic: {topic}"),
    HumanMessage(content="Pro's final argument: {pro_argument}"),
    HumanMessage(content="Con's final argument: {con_argument}"),
    ("placeholder", "{chat_history}"),
    HumanMessage(content="Now deliver your verdict:")
])

pro_chain = pro_prompt | llm
con_chain = con_prompt | llm
moderator_chain = moderator_prompt | llm


class DebateState(TypedDict):
    topic: str
    pro_argument: str
    con_argument: str
    current_speaker: str
    chat_history: Annotated[list[BaseMessage], add_messages]
    round: int
    max_rounds: int

def pro_node(state: DebateState) -> DebateState:
    result = pro_chain.invoke({
        "topic": state["topic"],
        "con_argument": state.get("con_argument", "No prior argument."),
        "chat_history": state["chat_history"][-4:]
    })
    return {
        "pro_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)]
    }

def con_node(state: DebateState) -> DebateState:
    result = con_chain.invoke({
        "topic": state["topic"],
        "pro_argument": state.get("pro_argument", "No prior argument."),
        "chat_history": state["chat_history"][-4:]
    })
    return {
        "con_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)]
    }
def moderator_node(state: DebateState) -> DebateState:
    result = moderator_chain.invoke({
        "topic": state["topic"],
        "pro_argument": state.get("pro_argument", "No prior argument."),
        "con_argument": state.get("con_argument", "No prior argument."),
        "chat_history": state["chat_history"][-4:]
    })
    return {
        "chat_history": [HumanMessage(content=result.content)]
    }
