"""
Lightweight web research helpers for agents.

- search_web: query DuckDuckGo for top results with snippets
- build_citations: format results into a compact, numbered citation list
- generate_queries: use LLM to propose targeted queries (falls back to heuristics)
- gather_evidence: assemble citations relevant to a topic and stance/persona
- fact_check_claim: search to support/refute a claim and summarize citations

All functions are resilient to missing internet or packages; they degrade to
empty evidence strings so agents can still run.
"""
from __future__ import annotations

from typing import List, Dict, Any

try:
    from duckduckgo_search import DDGS  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    DDGS = None  # type: ignore

from llm import llm
import json


def search_web(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    """Search the web and return a list of results with title, href, and snippet.

    Falls back to empty list if DDG is unavailable.
    """
    results: List[Dict[str, Any]] = []
    if not query or max_results <= 0:
        return results
    if DDGS is None:
        return results
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, safesearch="moderate"):
                # r keys: title, href, body (snippet)
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
                if len(results) >= max_results:
                    break
    except Exception:
        # Network or API failure -> return best-effort empty
        return []
    return results


def build_citations(results: List[Dict[str, Any]]) -> str:
    """Format results into a numbered citation block.

    Example:
    [1] Source Title — snippet (url)
    [2] ...
    """
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snip = (r.get("snippet") or "").strip()
        # Keep lines concise for prompt budgets
        if len(snip) > 220:
            snip = snip[:217] + "..."
        lines.append(f"[{i}] {title} — {snip} ({url})")
    return "\n".join(lines)


def _json_list_guard(txt: str, fallback: List[str]) -> List[str]:
    import json
    try:
        data = json.loads(txt)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data[:6]
    except Exception:
        pass
    return fallback


def generate_queries(
    topic: str,
    stance: str,
    persona: str,
    opponent_claim: str | None = None,
    mode: str = "support",
) -> List[str]:
    """Ask the LLM for 3-5 sharp search queries tailored to stance/persona.

    mode: "support" to cherry-pick favorable evidence; "attack" to seek
    critiques and fact checks against opponent_claim.

    Falls back to simple templates if LLM/tooling fails.
    """
    if mode not in {"support", "attack"}:
        mode = "support"
    if mode == "support":
        base_fallback = [
            f"benefits of {topic}",
            f"{topic} positive outcomes meta-analysis",
            f"case studies supporting {topic}",
        ]
    else:
        base_fallback = [
            f"fact check {opponent_claim or topic}",
            f"criticisms of {opponent_claim or topic}",
            f"{opponent_claim or topic} debunked controversy",
        ]
    try:
        intent = (
            "support your stance by cherry-picking favorable sources"
            if mode == "support"
            else "undermine the opponent by finding credible refutations and weaknesses"
        )
        prompt = (
            "You generate 3-5 web search queries optimized to " + intent + ".\n"
            "Bias is intentional: prefer phrasing that increases the chance of alignment with the goal.\n"
            f"Topic: {topic}\nStance: {stance}\nPersona: {persona}\n"
            f"Opponent claim: {opponent_claim or 'n/a'}\n"
            "Important: Return ONLY a JSON array of query strings."
        )
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", "") or str(resp)
        return _json_list_guard(text, base_fallback)
    except Exception:
        return base_fallback


def _score_results_bias(results: List[Dict[str, Any]], topic: str, stance: str, opponent_claim: str | None, mode: str) -> List[int]:
    """Ask the LLM to score each result for rhetorical usefulness (0-100)."""
    if not results:
        return []
    try:
        lines = []
        for i, r in enumerate(results, 1):
            t = (r.get("title") or "").strip()
            sn = (r.get("snippet") or "").strip()
            url = (r.get("url") or "").strip()
            if len(sn) > 320:
                sn = sn[:317] + "..."
            lines.append(f"{i}. {t} — {sn} ({url})")
        objective = (
            "Maximize support for the stance by preferring clearly favorable evidence."
            if mode == "support"
            else "Maximize ability to attack the opponent by preferring refutations/contradictions of their claim."
        )
        prompt = (
            "You are a debate researcher. Score each item 0-100 for how useful it is to your rhetorical goal.\n"
            f"Topic: {topic}\nStance: {stance}\nOpponent claim: {opponent_claim or 'n/a'}\nGoal: {objective}\n"
            "Items:\n" + "\n".join(lines) + "\n"
            "Return ONLY a JSON array of integers (0-100), one per item, same order."
        )
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", "") or str(resp)
        scores = _json_list_guard(text, [50] * len(results))
        # Convert to ints in 0..100
        clean: List[int] = []
        for s in scores:
            try:
                v = int(float(s))
            except Exception:
                v = 50
            v = max(0, min(100, v))
            clean.append(v)
        # If sizes mismatch, pad
        while len(clean) < len(results):
            clean.append(50)
        return clean[: len(results)]
    except Exception:
        return [50] * len(results)


def _re_rank_by_bias(results: List[Dict[str, Any]], topic: str, stance: str, opponent_claim: str | None, mode: str, bias_strength: float) -> List[Dict[str, Any]]:
    scores = _score_results_bias(results, topic, stance, opponent_claim, mode)
    paired = list(zip(results, scores))
    # Sort by score desc
    paired.sort(key=lambda x: x[1], reverse=True)
    # Optionally threshold by bias_strength
    threshold = max(0, min(100, int(bias_strength * 100)))
    filtered = [r for (r, s) in paired if s >= threshold]
    # If filtering removed everything, fall back to top-k
    if not filtered:
        filtered = [r for (r, _s) in paired]
    return filtered


def gather_evidence(
    topic: str,
    stance: str,
    persona: str,
    opponent_claim: str | None = None,
    max_total: int = 8,
    mode: str = "support",
    bias_strength: float = 0.75,
) -> str:
    """Run multiple searches and return a compact, biased citations block.

    mode: "support" cherry-picks favorable evidence. "attack" surfaces items that
    undermine opponent claims. bias_strength in [0,1] controls aggressiveness of
    filtering; higher values skew more strongly.
    """
    try:
        queries = generate_queries(topic, stance, persona, opponent_claim, mode=mode)
        seen = set()
        merged: List[Dict[str, Any]] = []
        for q in queries:
            for r in search_web(q, max_results=4):
                url = (r.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                merged.append(r)
                if len(merged) >= max_total * 2:  # over-collect, then filter
                    break
            if len(merged) >= max_total * 2:
                break
        # Bias-aware re-ranking/filtering
        biased = _re_rank_by_bias(merged, topic, stance, opponent_claim, mode, bias_strength)
        return build_citations(biased[:max_total])
    except Exception:
        return ""


def fact_check_claim(claim: str, topic: str | None = None) -> str:
    """Search to quickly check a claim; return citations list.

    Strategy: query with keywords like fact check / site filters to nudge credible sources.
    """
    if not claim:
        return ""
    heuristics = [
        f"fact check {claim}",
        f"{claim} site:.gov",  # encourage official data
        f"{claim} site:.edu",  # academic sources
    ]
    if topic:
        heuristics.append(f"{topic} claim verification")
    try:
        results: List[Dict[str, Any]] = []
        seen = set()
        for q in heuristics:
            for r in search_web(q, max_results=4):
                url = (r.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                results.append(r)
                if len(results) >= 10:
                    break
            if len(results) >= 10:
                break
        return build_citations(results)
    except Exception:
        return ""
