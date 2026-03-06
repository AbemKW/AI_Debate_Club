"""
Lightweight wrapper to initialize LangMem memory tools (manage/search) per agent.
Falls back to (None, None) if langmem or LangGraph store isn't available.
"""

from typing import Tuple, Optional

try:
    from langmem import create_manage_memory_tool, create_search_memory_tool
    from langgraph.store.memory import InMemoryStore
except Exception:
    create_manage_memory_tool = None
    create_search_memory_tool = None
    InMemoryStore = None


def get_memory_tools(namespace: str) -> Tuple[Optional[object], Optional[object]]:
    """
    Return (manage_tool, search_tool) for the given namespace.
    If tools are unavailable, return (None, None).
    """
    if create_manage_memory_tool is None or create_search_memory_tool is None or InMemoryStore is None:
        return None, None
    try:
        manage = create_manage_memory_tool(namespace=(f"mem-{namespace}",))
        search = create_search_memory_tool(namespace=(f"mem-{namespace}",))
        return manage, search
    except Exception:
        return None, None