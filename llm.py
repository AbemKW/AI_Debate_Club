import os
from langchain_openai import ChatOpenAI

"""
Chat model configured to use Hugging Face's OpenAI-compatible Inference Router.
Set your HF token in the Space settings (Repository secrets) as HF_TOKEN.
Optionally override the model via HF_CHAT_MODEL env var.
"""
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    # Make missing token obvious at startup in Space logs
    raise RuntimeError(
        "HF_TOKEN is not set. Add it in your Hugging Face Space: Settings -> Repository secrets -> HF_TOKEN."
    )

llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
        model="deepseek-ai/DeepSeek-V3-0324:together",
        streaming=True,
        temperature=0.7,
    )

def health_check() -> bool:
    """Run a tiny test call to verify router + token + model work.

    Returns (ok, message).
    """
    try:
        # Use the minimal LC chat interface
        resp = llm.invoke("ping")
        txt = getattr(resp, "content", None) or str(resp)
        return True
    except Exception as e:
        return False
