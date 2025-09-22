---
title: AI Debate Club
sdk: streamlit
emoji: ⚡
colorFrom: red
colorTo: yellow
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/rlxmLQTNWlU?si=CIVTk3FYPJbO76wb" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

# AI Debate Club: Multi-Agent AI Debate Platform

An interactive Streamlit application that simulates intelligent debates between AI agents using advanced language models and orchestration frameworks. Experience the future of AI discourse where autonomous agents engage in structured, turn-based debates on complex topics.

## Features

- **Multi-Agent Architecture**: Pro and Con agents debate autonomously with a neutral Moderator overseeing the discussion
- **LangGraph Integration**: Built on LangGraph for robust agent orchestration and state management
- **Real-time Streaming**: Watch debates unfold in real-time with smooth text streaming
- **Context-Aware Responses**: Agents maintain conversation history and build upon previous arguments
- **Customizable Topics**: Input any debate topic and watch the AI agents formulate compelling arguments
- **Turn-Based Moderation**: Ensures fair, structured debate with proper turn-taking and summary generation

## How It Works

1. **Input Debate Topic**: Enter any controversial or thought-provoking topic
2. **Agent Initialization**: Pro and Con agents are instantiated with opposing viewpoints
3. **Structured Debate**: Moderator ensures alternating turns and maintains debate coherence
4. **Intelligent Responses**: Each agent generates contextually relevant arguments using advanced language models
5. **Debate Summary**: Moderator provides final analysis and key takeaways

## Technical Stack

- **Frontend**: Streamlit for interactive web interface
- **Backend**: Python with LangGraph for agent orchestration
- **AI Models**: Integration-ready for Hugging Face models and OpenAI GPT series
- **State Management**: Robust conversation memory and context tracking

## Getting Started

This Space runs a Streamlit app from `app.py`. The current implementation includes:

- Complete UI framework with debate visualization
- Agent architecture with placeholder responses
- Moderation logic for turn management

## Notes

- Currently uses placeholder text for PRO/CON outputs to demonstrate the interface
- Supports Hugging Face Inference API or direct LangGraph pipeline connections
- Easily extensible for additional debate formats (panels, cross-examinations, etc.)

Explore the power of multi-agent AI systems in structured debate scenarios! 🧠⚖️🤖
