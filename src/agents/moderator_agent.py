from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState

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

moderator_chain = moderator_prompt | llm


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
