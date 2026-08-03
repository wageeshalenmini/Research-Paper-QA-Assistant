# Research Paper Q&A Assistant

This project demonstrates two different approaches to building a research question-answering system using large language models (LLMs) and retrieval-augmented generation (RAG):

1. A single-agent architecture (ReAct pattern)
2. A multi-agent architecture with specialized search, summarization, and critique agents

Both versions use [LangGraph](https://github.com/logspace-ai/langchain-extensions#%EF%B8%8F-langgraph) for graph-based control flow and [Ollama](https://ollama.com/) as the local LLM inference engine, running on top of [LangChain](https://langchain.com/).

## Setup

1. Install [Ollama](https://ollama.com/download) and add it to your PATH
2. Pull the `llama3.1` model: `ollama pull llama3.1` (a few GB download)
3. Install Python dependencies: `pip install langchain-ollama arxiv langgraph langchain`

## Usage

To run the single-agent version:
```bash
python single_agent.py
```

To run the multi-agent version:
```bash
python multi_agent.py
```

Both scripts run in an interactive loop. Type a research question at the prompt and wait for the assistant to search arXiv, read relevant abstracts, and synthesize an answer. Type `exit` or `quit` to end the program.

## Architecture Comparison

### Single Agent (ReAct)

The single-agent version uses a ReAct-style (Reason+Act) loop:
- One agent iterates between reasoning about the question and calling `search_arxiv` / `get_paper_abstract` tools until it has enough information to answer
- The agent has access to all information (its own search queries, raw results, past reasoning) at every step
- Simple to implement, but no specialization or self-checking

### Multi-Agent Pipeline

The multi-agent version splits the task across specialized agents:
- **Search Agent:** Uses tools to find relevant papers and read abstracts
- **Summarizer Agent:** Synthesizes key claims and citations from the raw search results
- **Critic Agent:** Fact-checks the summary against the source abstracts, sends it back for revision if any claims are unsupported
- Agents are loosely coupled — each one only sees the final output of the previous stage, not the full reasoning chain
- Supports specialization (search vs. summarization vs. critique) and iterative refinement
- More complex to implement, but more scalable and self-correcting

## Key Learnings

- **Fact-checking matters:** The critic agent catches real errors and exaggerations that the single-agent version misses. This is the key value-add of multi-agent architectures.
- **Pipeline structure shapes communication:** What each agent sees and doesn't see (e.g. the summarizer getting only the search agent's final output, not its full message history) meaningfully affects the final result. Agent communication interfaces are a key design surface.
- **Local models are cheaper but flakier:** Running on local Ollama models avoids API costs, but requires much more explicit prompting to match the reliability of hosted APIs like Claude. Error analysis and model comparison are essential parts of the development loop.

## Future Directions

- **Add a citation quality score:** Have the critic assign a numeric score for how well-supported the summary is, surface that to the user, and track it across queries to measure output quality.
- **Experiment with other pipelines:** Try a blackboard architecture where multiple agents contribute to a shared knowledge store, or a tree structure with multiple summarizer agents that specialize in different domains.
- **Scale to larger models:** Run on `qwen2.5:14b` or `llama3.1:70b` to improve reliability and coherence, at the cost of slower runtime.

