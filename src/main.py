from langgraph.graph import StateGraph, END
from agents.pro_agent import pro_node
from agents.con_agent import con_node
from agents.moderator_agent import moderator_node
from debate_state import DebateState


graph = StateGraph(DebateState)
graph.add_node("pro", pro_node)
graph.add_node("con", con_node)
graph.add_node("moderator", moderator_node)

graph.set_entry_point("pro")

def route_speaker(state):
    if(state["round"] >= state["max_rounds"]):
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