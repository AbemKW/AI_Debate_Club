import os
from langchain_groq import ChatGroq

HF_TOKEN = os.environ.get("groq_key")
if not HF_TOKEN:
    # Make missing token obvious at startup in Space logs
    raise RuntimeError(
        "HF_TOKEN is not set. Add it in your Hugging Face Space: Settings -> Repository secrets -> HF_TOKEN."
    )

llm = ChatGroq(
        api_key=HF_TOKEN,
        model="groq/compound",
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
