"""
Single-agent research Q&A assistant, hardened with deterministic scaffolding.

This is still single-agent: exactly one node (`agent_node`) has an LLM deciding
what happens next in a loop. Every other node is plain code - a regex check,
a string comparison, a counter - not a second decision-maker. That's what
keeps this out of multi-agent territory while still fixing the three most
common failure modes of a bare ReAct loop:

  1. Unbounded tool loops        -> step_count cap, forces a final answer
  2. Hallucinated citations      -> grounding_check_node (regex, no LLM)
  3. Weak search recall          -> normalize_query_node rewrites vague
                                     questions into a tighter search query
                                     before the agent ever starts
"""

import re
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from arxiv_tools import ALL_TOOLS

MAX_AGENT_STEPS = 6         # agent<->tools round trips before forcing a final answer
MAX_GROUNDING_RETRIES = 2   # bounded retries if the answer cites unverified papers

ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    normalized_query: str
    final_answer: str
    step_count: int
    retry_count: int
    grounding_ok: bool
    grounding_issues: list


llm = ChatOllama(model="llama3.1", temperature=0).bind_tools(ALL_TOOLS)
plain_llm = ChatOllama(model="llama3.1", temperature=0)  # no tools - used for
                                                           # normalization and
                                                           # forced finalization


# ------------------------------------------------------------- normalize query

NORMALIZE_PROMPT = """Rewrite the user's research question into a short, keyword-dense
search query suitable for arXiv search. Don't answer the question - just rewrite it as
a better search query. Return only the rewritten query, nothing else.

Original question: {question}"""


def normalize_query_node(state: State):
    response = plain_llm.invoke(
        [HumanMessage(content=NORMALIZE_PROMPT.format(question=state["question"]))]
    )
    normalized = response.content.strip()
    seed = HumanMessage(
        content=f"{state['question']}\n\n(Suggested search focus: {normalized})"
    )
    return {"normalized_query": normalized, "messages": [seed]}


# --------------------------------------------------------------------- agent

SYSTEM_PROMPT = """You are a research assistant. When asked a research question:
1. Use search_arxiv to find relevant candidate papers.
2. Use get_paper_abstract on the most relevant 2-4 papers to read their full abstracts.
3. Write a synthesized answer that cites specific papers by arXiv ID and title.
Do not state a claim about a paper without having actually read its abstract via the tools.
If the search results are thin or off-topic, say so rather than filling gaps from general knowledge."""


def agent_node(state: State):
    step = state.get("step_count", 0) + 1
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    if step > MAX_AGENT_STEPS:
        # Safety valve: stop calling tools, force a final answer from whatever
        # evidence has been gathered so far.
        forced = messages + [HumanMessage(
            content="You've reached the research step limit. Write your final "
                     "synthesized answer now using only what you've already found - "
                     "do not request any more tools."
        )]
        response = plain_llm.invoke(forced)
        return {"messages": [response], "step_count": step}

    response = llm.invoke(messages)
    return {"messages": [response], "step_count": step}


def route_agent(state: State):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "grounding_check"


# ------------------------------------------------------------- grounding check

def extract_arxiv_ids(text: str) -> set:
    return set(ARXIV_ID_RE.findall(text))


def grounding_check_node(state: State):
    """Deterministic check, no LLM call: every arXiv ID cited in the final answer
    must actually appear in a ToolMessage the agent generated - i.e. an abstract
    it actually retrieved, not one it's guessing about."""
    answer_text = state["messages"][-1].content
    cited_ids = extract_arxiv_ids(answer_text)

    tool_text = "\n".join(m.content for m in state["messages"] if isinstance(m, ToolMessage))
    available_ids = extract_arxiv_ids(tool_text)

    unverified = cited_ids - available_ids
    retry_count = state.get("retry_count", 0)

    if unverified and retry_count < MAX_GROUNDING_RETRIES:
        note = HumanMessage(content=(
            f"Grounding check failed: you cited {', '.join(sorted(unverified))} but no "
            "abstract for that ID was actually retrieved via tools. Only cite papers "
            "you've actually read abstracts for - fetch the abstract first, or remove "
            "the claim."
        ))
        return {
            "grounding_ok": False,
            "grounding_issues": sorted(unverified),
            "retry_count": retry_count + 1,
            "messages": [note],
        }

    return {
        "grounding_ok": True,
        "grounding_issues": sorted(unverified),
        "final_answer": answer_text,
    }


def route_grounding(state: State):
    return "format_answer" if state.get("grounding_ok") else "agent"


# ----------------------------------------------------------------- formatting

def format_answer_node(state: State):
    """Deterministic: build the references list from what was actually verified,
    rather than trusting the model's own formatting of its citations."""
    answer = state.get("final_answer", "").strip()
    issues = state.get("grounding_issues", [])

    tool_text = "\n".join(m.content for m in state["messages"] if isinstance(m, ToolMessage))
    verified_ids = extract_arxiv_ids(answer) & extract_arxiv_ids(tool_text)

    parts = [answer]
    if verified_ids:
        parts.append("\n---\nVerified references (abstract actually retrieved): "
                      + ", ".join(sorted(verified_ids)))
    if issues:
        parts.append("\nNote: these citations could not be verified against retrieved "
                      "abstracts and may be inaccurate: " + ", ".join(issues))

    return {"final_answer": "\n".join(parts)}


# ------------------------------------------------------------------------ graph

graph = StateGraph(State)
graph.add_node("normalize_query", normalize_query_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(ALL_TOOLS))
graph.add_node("grounding_check", grounding_check_node)
graph.add_node("format_answer", format_answer_node)

graph.add_edge(START, "normalize_query")
graph.add_edge("normalize_query", "agent")
graph.add_conditional_edges("agent", route_agent, {"tools": "tools", "grounding_check": "grounding_check"})
graph.add_edge("tools", "agent")
graph.add_conditional_edges("grounding_check", route_grounding, {"format_answer": "format_answer", "agent": "agent"})
graph.add_edge("format_answer", END)

app = graph.compile()


if __name__ == "__main__":
    print("Research Q&A Assistant (single agent, reliability-hardened). Type 'exit' to quit.\n")
    while True:
        question = input("Ask a research question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        initial_state = {
            "messages": [],
            "question": question,
            "normalized_query": "",
            "final_answer": "",
            "step_count": 0,
            "retry_count": 0,
            "grounding_ok": False,
            "grounding_issues": [],
        }

        result = app.invoke(initial_state)

        print("\n" + result["final_answer"] + "\n")
        if result.get("grounding_issues"):
            print(f" (flagged {len(result['grounding_issues'])} unverified citation(s) "
                  f"after {result.get('retry_count', 0)} retry attempt(s))")
        print(f" Research steps used: {result.get('step_count', 0)}/{MAX_AGENT_STEPS}\n")