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
    if state["round"] >= state["max_rounds"]:
        return "moderator"
    if state["current_speaker"] == "pro":
        return "pro"
    elif state["current_speaker"] == "con":
        return "con"
    else:
        return "moderator"

graph.add_conditional_edges("pro", route_speaker)
graph.add_conditional_edges("con", route_speaker)
graph.add_conditional_edges("moderator", lambda x: END)

graph_app = graph.compile()
