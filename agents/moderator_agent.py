from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from llm import llm
from debate_state import DebateState

moderator_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are the MODERATOR and judge. Remain strictly neutral and decide the round based on the arguments presented, not your personal beliefs.

Do the following in a concise, professional format:
1) Synopsis (2–3 sentences): Fairly summarize each side's core case and key clashes.
2) Fallacies & Theory: Note any major logical fallacies, mischaracterizations, or rules violations if present.
3) Evaluation: Use impact calculus (magnitude, probability, timeframe, reversibility, ethical priority) and responsiveness (offense, defense, weighing) to compare the cases.
4) Decision & RFD: Declare a clear winner (Pro or Con) and provide a brief Reason For Decision explaining which arguments and weighing persuaded you.

Constraints:
- Judge only what appears in-round; do not introduce new evidence.
- Prefer warranted, comparative analysis over verbosity.
- Keep the total response around 150–220 words.
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
    print("\nModerator's Verdict:", result.content)
    return {
        "chat_history": [HumanMessage(content=result.content)]
    }
