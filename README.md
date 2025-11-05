---
title: AI Debate Club
sdk: streamlit
emoji: ⚡
colorFrom: red
colorTo: yellow
---

[YouTube Demo]()
# AI Debate Club

> **Two AI agents. One topic. A full debate — with research, memory, and personality.**

A multi-agent AI system where two roleplaying agents argue user-defined topics, supported by web research, persistent memory, and expressive persona-driven reasoning — all orchestrated via LangGraph and powered by Groq.

[Watch the 3-part YouTube series]([https://youtube.com/playlist?list=...](https://www.youtube.com/watch?v=rlxmLQTNWlU&list=PLiuwYwnHlubxCD4fE5I-VUtkTbc9CRL7Y))  
[Try the live demo on Hugging Face]([https://huggingface.co/spaces/abemkibatu101/ai-debate-club])

##  Features

**Multi-Agent Orchestration**  
- Pro, Con, and Moderator agents take turns using **LangGraph** state machines and conditional routing.

**Persona-Driven Debates**  
- Assign any personas (e.g., "Socrates vs Elon Musk", "Trump vs The Rock") — agents adapt tone, style, and rhetoric dynamically.

**Web Research with Bias Simulation**  
- Each agent runs **biased web searches** to find supporting evidence and fact-check opponents using DuckDuckGo.
- Search queries are LLM-generated for maximum relevance.

**Persistent Memory**  
- Agents remember past arguments using **LangMem**.
- Avoid repetition, build on previous points, and call out contradictions.

**Rhetorical Diversity Engine**  
- Forces rotation of rhetorical styles (analogy, data, anecdote, emotion) to prevent monotony.
- Ensures unique voice across rounds.

**Streamlit UI**  
- Interactive interface showing real-time debate flow, citations, and agent roles.
- Easy for non-technical users to engage.

---

## Tech Stack

| Layer | Technology |
|------|------------|
| **Orchestration** | LangGraph, LangChain |
| **LLM Backend** | Groq (Llama-3), or local Qwen3-4B via LM Studio |
| **UI** | Streamlit |
| **Research** | DuckDuckGo Search, `langchain-huggingface` |
| **Memory** | LangMem (InMemoryStore) |
| **Deployment** | Hugging Face Spaces |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourname/ai-debate-club.git
cd ai-debate-club
