
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gradio as gr
from src.main import app

def run_debate(topic, max_rounds):
    state = {
        "topic": topic,
        "chat_history": [],
        "pro_argument": "",
        "con_argument": "",
        "current_speaker": "pro",
        "round": 0,
        "max_rounds": int(max_rounds)
    }
    result = app.invoke(state)
    return str(result)

with gr.Blocks() as demo:
    gr.Markdown("# AI Debate Club")
    topic = gr.Textbox(label="Debate Topic", value="The impact of AI on society")
    max_rounds = gr.Number(label="Number of Rounds", value=3)
    output = gr.Textbox(label="Debate Result")
    btn = gr.Button("Start Debate")
    btn.click(run_debate, inputs=[topic, max_rounds], outputs=output)

demo.launch()
