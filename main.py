from langgraph.graph import StateGraph, END
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
    ("user", "Topic: {topic}"),  # ✅ Now {topic} will be filled
    ("user", "Opponent's last argument: {con_argument}"),
    ("placeholder", "{chat_history}"),
    ("user", "Now make your case:")
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
    ("user", "Topic: {topic}"),
    ("user", "Opponent's last argument: {pro_argument}"),
    ("placeholder", "{chat_history}"),
    ("user", "Now make your rebuttal:")
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
    ("user", "Topic: {topic}"),
    ("user", "Pro's final argument: {pro_argument}"),
    ("user", "Con's final argument: {con_argument}"),
    ("placeholder", "{chat_history}"),
    ("user", "Now deliver your verdict:")
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
    print("Pro's Argument:\n", result.content)
    return {
        "pro_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "con",
        "round": state["round"] + 1
    }

def con_node(state: DebateState) -> DebateState:
    result = con_chain.invoke({
        "topic": state["topic"],
        "pro_argument": state.get("pro_argument", "No prior argument."),
        "chat_history": state["chat_history"][-4:]
    })
    print("Con's Argument:\n", result.content)
    return {
        "con_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "pro",
        "round": state["round"] + 1
    }
def moderator_node(state: DebateState) -> DebateState:
    result = moderator_chain.invoke({
        "topic": state["topic"],
        "pro_argument": state.get("pro_argument", "No prior argument."),
        "con_argument": state.get("con_argument", "No prior argument."),
        "chat_history": state["chat_history"][-4:]
    })
    print("Moderator's Verdict:\n", result.content)
    return {
        "chat_history": [HumanMessage(content=result.content)]
    }

graph = StateGraph(DebateState)
graph.add_node("pro", pro_node)
graph.add_node("con", con_node)
graph.add_node("moderator", moderator_node)

graph.set_entry_point("pro")

def route_speaker(state):
    if(state["round"] > state["max_rounds"]):
        return "moderator"
    if(state["current_speaker"] == "pro"):
        return "pro"
    elif state["current_speaker"] == "con":
        return "con"
    else:
        return "moderator"

# Use conditional edge from a "router" step
graph.add_conditional_edges(
    source="pro",  # After pro speaks
    path=route_speaker
)
graph.add_conditional_edges(
    source="con",  # After con speaks
    path=route_speaker
)
graph.add_conditional_edges(
    source="moderator",
    path=lambda x: END  # Moderator ends the flow
)

app = graph.compile()

result = app.invoke({
    "topic": "The impact of AI on society",
    "chat_history": [],
    "pro_argument": "",
    "con_argument": "",
    "current_speaker": "pro",
    "round": 0,
    "max_rounds": 3
})
