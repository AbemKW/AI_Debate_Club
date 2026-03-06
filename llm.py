import os

def get_llm():
    """Helper to create the LLM instance. Defaults to OpenAI, falls back to Groq."""
    
    # Try OpenAI first
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=True
        )

    # Try Groq as fallback
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("groq_key")
    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=groq_key,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            streaming=True
        )
        
    raise RuntimeError(
        "No API key found. Please set OPENAI_API_KEY or GROQ_API_KEY."
    )

def health_check() -> bool:
    """Run a tiny test call to verify router + token + model work."""
    try:
        client = get_llm()
        resp = client.invoke("ping")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

# Export a module-level `llm` for callers that expect it
try:
    llm = get_llm()
except Exception:
    llm = None
