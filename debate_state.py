from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class DebateState(TypedDict):
    topic: str
    pro_argument: str
    con_argument: str
    current_speaker: str
    chat_history: Annotated[list[BaseMessage], add_messages]
    round: int
    max_rounds: int
    moderator_verdict: str
