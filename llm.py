import os
from langchain_openai import ChatOpenAI

"""
Chat model configured to use Hugging Face's OpenAI-compatible Inference Router.
Set your HF token in the Space settings (Repository secrets) as HF_TOKEN.
Optionally override the model via HF_CHAT_MODEL env var.
"""

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = os.environ.get(
    "HF_CHAT_MODEL",
    "Qwen/Qwen2.5-7B-Instruct",  # Fallback to a widely available instruct model
)

llm = ChatOpenAI(
    base_url=HF_ROUTER_BASE_URL,
    api_key=os.environ.get("HF_TOKEN", ""),
    model=DEFAULT_MODEL,
    streaming=True,
    temperature=0.7,
)