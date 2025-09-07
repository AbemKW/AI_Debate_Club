import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gradio as gr
from graph import graph_app
from llm import health_check


def set_round(max_rounds):
    try:
        # Avoid shadowing the built-in 'round'
        num_rounds = int(max_rounds) + 1
        return num_rounds
    except ValueError:
        return 3  # Default value if conversion fails

def run_debate(topic, max_rounds):
    num_rounds = set_round(max_rounds)
    state = {
        "topic": topic,
        "chat_history": [],
        "pro_argument": "",
        "con_argument": "",
        "current_speaker": "pro",
        "round": 1,
        "max_rounds": num_rounds
    }
    debate_text = f"🎯 **Debate Topic:** {topic}\n"
    debate_text += f"📊 **Number of Rounds:** {num_rounds}\n"
    debate_text += "=" * 50 + "\n\n"
    
    # Track previous state to detect changes
    prev_pro_arg = ""
    prev_con_arg = ""
    prev_round = 0
    
    # Use app.stream instead of app.invoke
    try:
        for step in graph_app.stream(state, stream_mode="values"):
            # step is a dict with the current state
            current_round = step.get("round", 0)
            current_pro = step.get("pro_argument", "")
            current_con = step.get("con_argument", "")

            # Check if Pro made a new argument
            if current_pro and current_pro != prev_pro_arg:
                debate_text += f"🔵 **Pro's Argument (Round {current_round}):**\n{current_pro}\n\n"
                prev_pro_arg = current_pro
                yield debate_text

            # Check if Con made a new argument  
            if current_con and current_con != prev_con_arg:
                debate_text += f"🔴 **Con's Argument (Round {current_round}):**\n{current_con}\n\n"
                prev_con_arg = current_con
                yield debate_text

            # Check if we've reached the end (moderator verdict)
            chat_history = step.get("chat_history", [])
            if chat_history and current_round >= num_rounds and len(chat_history) > 0:
                # This should be the moderator's verdict
                last_message = chat_history
                if hasattr(last_message, 'content'):
                    debate_text += f"⚖️ **Moderator's Verdict:**\n{last_message.content}\n\n"
                    debate_text += "=" * 50 + "\n🏁 **Debate Complete!**"
                    yield debate_text
                    break
    except Exception as e:
        # Also print to Space logs
        print("LLM/graph error:", repr(e))
        debate_text += "\n❌ LLM error while running the debate. Check Space logs.\n"
        debate_text += f"Details: {e}"
        yield debate_text
    yield debate_text

with gr.Blocks() as demo:
    gr.Markdown("# AI Debate Club")
    ok = health_check()
    status_color = "green" if ok else "red"
    gr.Markdown(
        f"<div style='border-left:4px solid {status_color}; padding:8px;'>LLM health: {ok}</div>",
        elem_id="health",
        visible=True,
    )
    topic = gr.Textbox(label="Debate Topic", value="The impact of AI on society")
    max_rounds = gr.Number(label="Number of Rounds", value=3, precision=0)
    output = gr.Textbox(label="Debate Result", interactive=True)
    btn = gr.Button("Start Debate")
    btn.click(run_debate, inputs=[topic, max_rounds], outputs=output)

demo.launch()