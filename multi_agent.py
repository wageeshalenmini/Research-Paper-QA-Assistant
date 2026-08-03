from typing import Annotated
from typing_extensions import TypedDict
 
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
 
from arxiv_tools import ALL_TOOLS
 
MAX_REVISIONS = 2
 
 
class State(TypedDict):
    messages: Annotated[list, add_messages]   # search agent's own scratchpad
    question: str
    papers: str          # collected search results + abstracts, handed off between agents
    summary: str
    critique: str
    approved: bool
    revision_count: int
 
 
search_llm = ChatOllama(model="llama3.1", temperature=0).bind_tools(ALL_TOOLS)
plain_llm = ChatOllama(model="llama3.1", temperature=0)
 
 
# ---------------------------------------------------------------- search agent
 
SEARCH_SYSTEM_PROMPT = """You are a search agent. Your only job is to find and read papers
relevant to the user's question.
1. Call search_arxiv to find candidate papers.
2. Call get_paper_abstract on the 2-4 most relevant ones to read their full abstracts.
Once you've read enough abstracts to answer the question, stop calling tools and just
reply with a short note that you've gathered enough information."""
 
 
def search_agent_node(state: State):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SEARCH_SYSTEM_PROMPT)] + messages
    return {"messages": [search_llm.invoke(messages)]}
 
 
def route_search(state: State):
    """Custom routing instead of the prebuilt tools_condition, because we need
    to send control to collect_papers (not END) once the search agent stops
    calling tools."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "collect_papers"
 
 
def collect_papers_node(state: State):
    """Hand-off step: pull every ToolMessage (search results + abstracts) out
    of the search agent's private message history and put them in a plain
    'papers' state field. This is the actual moment of agent-to-agent
    communication - the summarizer never sees the search agent's reasoning,
    only this distilled fact sheet."""
    collected = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
    return {"papers": "\n\n".join(collected)}
 
 
# ------------------------------------------------------------- summarizer agent
 
SUMMARIZER_PROMPT = """You are a summarizer agent. Given a research question and raw paper
data (search results and abstracts), write a synthesized answer that cites specific papers
by arXiv ID and title. Only make claims directly supported by the abstract text provided -
do not add outside knowledge.
 
Question: {question}
 
Paper data:
{papers}
{critique_note}"""
 
 
def summarizer_node(state: State):
    critique_note = ""
    if state.get("critique"):
        critique_note = (
            "\n\nA previous draft was reviewed and flagged with this feedback - "
            f"revise your answer to address it:\n{state['critique']}"
        )
    prompt = SUMMARIZER_PROMPT.format(
        question=state["question"], papers=state["papers"], critique_note=critique_note
    )
    response = plain_llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content}
 
 
# ------------------------------------------------------------------ critic agent
 
CRITIC_PROMPT = """You are a critic agent fact-checking a research summary against the
source abstracts it claims to be based on.
 
Question: {question}
 
Source paper data:
{papers}
 
Draft summary to check:
{summary}
 
Check every claim in the draft summary against the source paper data above. If every claim
is supported, respond with exactly:
APPROVED
 
If any claim is unsupported, exaggerated, or not actually present in the source data,
respond with:
REVISE: <specific feedback on what to fix>"""
 
 
def critic_node(state: State):
    prompt = CRITIC_PROMPT.format(
        question=state["question"], papers=state["papers"], summary=state["summary"]
    )
    response = plain_llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    if text.upper().startswith("APPROVED"):
        return {"approved": True, "critique": ""}
    feedback = text.split("REVISE:", 1)[-1].strip() if "REVISE:" in text else text
    return {
        "approved": False,
        "critique": feedback,
        "revision_count": state.get("revision_count", 0) + 1,
    }
 
 
def route_after_critic(state: State):
    if state.get("approved"):
        return "end"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "end"  # safety valve so a stubborn critic can't loop forever
    return "revise"
 
 
# ------------------------------------------------------------------------- graph
 
graph = StateGraph(State)
graph.add_node("search_agent", search_agent_node)
graph.add_node("tools", ToolNode(ALL_TOOLS))
graph.add_node("collect_papers", collect_papers_node)
graph.add_node("summarizer", summarizer_node)
graph.add_node("critic", critic_node)
 
graph.add_edge(START, "search_agent")
graph.add_conditional_edges("search_agent", route_search, {"tools": "tools", "collect_papers": "collect_papers"})
graph.add_edge("tools", "search_agent")
graph.add_edge("collect_papers", "summarizer")
graph.add_edge("summarizer", "critic")
graph.add_conditional_edges("critic", route_after_critic, {"end": END, "revise": "summarizer"})
 
app = graph.compile()
 
 
if __name__ == "__main__":
    print("Multi-agent Research Q&A Assistant. Type 'exit' to quit.\n")
    while True:
        question = input("Ask a research question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "papers": "",
            "summary": "",
            "critique": "",
            "approved": False,
            "revision_count": 0,
        }
        result = app.invoke(initial_state)
        print("\n--- Final Answer ---")
        print(result["summary"])
        print(f"\n(revisions made: {result.get('revision_count', 0)}, approved: {result.get('approved')})\n")