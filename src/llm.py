from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="none",
    model="qwen/qwen3-4b"
)