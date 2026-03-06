import os
from langchain_groq import ChatGroq

# Prefer the explicit HF_TOKEN env var, fall back to older "groq_key" name
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("groq_key")
if not HF_TOKEN:
    # Make missing token obvious at startup in Space logs
    raise RuntimeError(
        "HF_TOKEN (or groq_key) is not set. Add it in your Hugging Face Space: Settings -> Repository secrets -> HF_TOKEN."
    )

def get_llm() -> ChatGroq:
    """Helper to create the LLM instance. Can be extended to support multiple models."""
    return ChatGroq(
        api_key=HF_TOKEN,
        model="qwen/qwen3-32b",
        streaming=True,
        temperature=0.7,
    )

def health_check() -> bool:
    """Run a tiny test call to verify router + token + model work.

    Returns (ok, message).
    """
    try:
        # Use the minimal LC chat interface via a fresh client to avoid
        # relying on any external module-level state.
        client = get_llm()
        resp = client.invoke("ping")
        txt = getattr(resp, "content", None) or str(resp)
        return True
    except Exception:
        return False


# Export a module-level `llm` for callers that expect it (legacy compatibility)
llm = get_llm()
