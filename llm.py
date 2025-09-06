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

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    # Make missing token obvious at startup in Space logs
    raise RuntimeError(
        "HF_TOKEN is not set. Add it in your Hugging Face Space: Settings -> Repository secrets -> HF_TOKEN."
    )

llm = ChatOpenAI(
    base_url=HF_ROUTER_BASE_URL,
    api_key=HF_TOKEN,
    model=DEFAULT_MODEL,
    streaming=True,
    temperature=0.7,
)

def health_check() -> tuple[bool, str]:
    """Run a tiny test call to verify router + token + model work.

    Returns (ok, message).
    """
    try:
        # Use the minimal LC chat interface
        resp = llm.invoke("ping")
        txt = getattr(resp, "content", None) or str(resp)
        return True, f"LLM OK (model={DEFAULT_MODEL}). Sample: {txt[:80]}"  # limit output
    except Exception as e:
        return False, f"LLM ERROR (model={DEFAULT_MODEL}): {e}"