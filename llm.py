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
FALLBACK_MODELS = [
    # A few commonly available chat-instruct models on the router
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
    "google/gemma-2-9b-it",
]

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

# On import, attempt an automatic fallback if the configured model isn't supported
try:
    ok, msg = health_check()
    disable_fallback = os.environ.get("HF_DISABLE_AUTO_FALLBACK", "0") == "1"
    # Only attempt fallback when it's a model support problem and fallback isn't disabled
    if (not ok) and (not disable_fallback) and (
        "model_not_supported" in msg
        or "not supported by any provider" in msg
        or "Invalid model" in msg
    ):
        for cand in FALLBACK_MODELS:
            try:
                test_llm = ChatOpenAI(
                    base_url=HF_ROUTER_BASE_URL,
                    api_key=HF_TOKEN,
                    model=cand,
                    streaming=True,
                    temperature=0.7,
                )
                resp = test_llm.invoke("ping")
                _ = getattr(resp, "content", None) or str(resp)
                llm = test_llm  # switch
                DEFAULT_MODEL = cand  # update for health banner
                print(f"[LLM] Fallback succeeded; using model: {cand}")
                break
            except Exception as fe:
                print(f"[LLM] Fallback candidate failed: {cand} -> {fe}")
        else:
            print("[LLM] No fallback models succeeded; staying with configured model.")
except Exception as ie:
    # Don't crash import due to fallback logic; health banner will show the error
    print(f"[LLM] Initialization check error: {ie}")