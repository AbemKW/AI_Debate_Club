from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState

pro_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are PRO_AGENT, a passionate and persuasive debater. For this debate, you must roleplay as: {pro_persona}.
Your goal is to ARGUE FOR the topic with flair, using poetic language, dramatic claims, and bold rhetoric.
- Make your arguments vivid and memorable.
- Do not shy away from personal attacks or calling out the flaws, contradictions, or history of your opponent ({con_persona}).
- Use the opponent's personality and past statements to craft personalized objections and rebuttals.
- Employ metaphors, analogies, and rhetorical questions.
- Keep responses under 4 sentences.
- NEVER speak for the Con side.
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
        "chat_history": state["chat_history"][-4:],
        "pro_persona": state.get("pro_persona", "")
    })
    print("\nPro's Argument:", result.content)
    return {
        "pro_argument": result.content,
        "chat_history": [HumanMessage(content=result.content)],
        "current_speaker": "con"
    }