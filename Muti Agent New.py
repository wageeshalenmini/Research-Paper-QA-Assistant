"""
Multi-agent research Q&A assistant - fully local / free version (no API costs).

Why this shape: the earlier 4-way supervisor ("choose search_agent, summarizer_agent,
critic_agent, or finish, plus write a directive, in one shot") was too much for
llama3.1 to judge reliably - it kept re-dispatching search after evidence was
already collected. Rather than pay for a stronger routing model, this version
removes the central router entirely and replaces it with two much narrower,
single-question judgment points a small local model can actually get right:

  1. evidence_judge_node  - "is there enough evidence, or do we need more?"
     (binary - not "which of 4 agents should run")
  2. critic_agent_node    - "approved, revise, or needs more evidence?"
     (already proven reliable in earlier versions - unchanged)

Code then translates each answer into the next graph hop. This is a real
trade-off worth naming: there's no single LLM deciding among every possible
next step, so it's a lighter multi-agent design than the supervisor version -
but both judgment points are still genuine LLM decisions that change control
flow, and the search agent's own tool-calling loop is untouched. Fully local,
zero API cost.
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

MAX_SEARCH_STEPS = 6       # caps the search agent's own tool-calling loop
MAX_EVIDENCE_RETRIES = 2   # caps how many times evidence_judge can send it back to search
MAX_REVISIONS = 3          # caps how many times critic can send it back to summarizer


class State(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    papers: str
    summary: str
    critique: str
    verdict: str                # "", APPROVED, REVISE, NEEDS_EVIDENCE
    approved: bool
    revision_count: int
    citation_score: float
    score_history: list
    search_directive: str
    evidence_ok: bool
    evidence_retries: int
    search_step_count: int
    collected_msg_idx: int


search_llm = ChatOllama(model="llama3.1", temperature=0).bind_tools(ALL_TOOLS)
plain_llm = ChatOllama(model="llama3.1", temperature=0)


# ---------------------------------------------------------------- search agent

SEARCH_SYSTEM_PROMPT = """You are a search agent. Your only job is to find and read papers
relevant to the user's question.
1. Call search_arxiv to find candidate papers.
2. Call get_paper_abstract on the 2-4 most relevant ones to read their full abstracts.
If you were given a directive, focus your search on filling that specific gap rather
than repeating a general search.
Once you've read enough abstracts, stop calling tools and reply with a short note that
you've gathered enough information."""


def search_agent_node(state: State):
    step = state.get("search_step_count", 0) + 1
    history = state["messages"]
    directive = state.get("search_directive", "")

    working = history
    if not any(isinstance(m, SystemMessage) for m in working):
        working = [SystemMessage(content=SEARCH_SYSTEM_PROMPT)] + working
    if directive:
        working = working + [HumanMessage(content=f"Directive: {directive}")]

    if step > MAX_SEARCH_STEPS:
        # safety valve: stop calling tools, force a stopping note
        forced = working + [HumanMessage(
            content="You've reached the search step limit. Reply with a short note "
                     "that you're done searching - do not call any more tools."
        )]
        response = plain_llm.invoke(forced)
    else:
        response = search_llm.invoke(working)

    new_messages = [response]
    if directive:
        new_messages = [HumanMessage(content=f"Directive: {directive}")] + new_messages

    return {"messages": new_messages, "search_directive": "", "search_step_count": step}


def route_search(state: State):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "collect_papers"


STOPWORDS = {
    "what", "are", "is", "the", "a", "an", "in", "on", "of", "for", "to", "and", "or",
    "about", "research", "papers", "paper", "domain", "recent", "recently",
    "published", "past", "few", "years", "year", "that", "this", "these", "those",
    "with", "by", "from", "been", "have", "has", "had", "can", "could", "would",
    "should", "which", "who", "whom", "how", "why", "when", "where", "does", "do",
}


def extract_keywords(question: str) -> list:
    """Deterministic, not LLM-judged: pull the question's own content words as the
    relevance filter. llama3.1 proved unreliable at judging topical fit even when
    explicitly instructed to (see evidence_judge_node's history in this project) -
    keyword matching against the question's own terms is cheap, free, and for a
    domain word like 'telecommunication' considerably more reliable."""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", question.lower())
    keywords = [w for w in words if w not in STOPWORDS]
    return keywords or words  # fallback if everything got stripped


def collect_papers_node(state: State):
    """Hand-off step: the summarizer and critic never see the search agent's private
    tool-call reasoning, only this distilled fact sheet - now filtered for topical
    relevance to the question before it's ever handed off."""
    messages = state["messages"]
    start = state.get("collected_msg_idx", 0)
    new_chunks_raw = [m.content for m in messages[start:] if isinstance(m, ToolMessage)]

    keywords = extract_keywords(state["question"])
    new_chunks = [c for c in new_chunks_raw if any(k in c.lower() for k in keywords)]
    dropped = len(new_chunks_raw) - len(new_chunks)
    if dropped:
        print(f"[collect_papers] dropped {dropped} off-topic result(s) "
              f"(no match for: {', '.join(keywords)})")

    papers = state.get("papers", "")
    if new_chunks:
        addition = "\n\n".join(new_chunks)
        papers = f"{papers}\n\n{addition}".strip() if papers else addition

    return {"papers": papers, "collected_msg_idx": len(messages)}


# --------------------------------------------------------------- evidence judge

EVIDENCE_JUDGE_PROMPT = """You are checking whether enough research evidence has been
collected to answer a question well.

Question: {question}

Evidence collected so far:
{papers}

Is this enough to write a well-supported answer? Respond in exactly this format:
DECISION: ENOUGH or MORE
FOCUS: <if MORE, one short phrase describing what's missing - otherwise 'none'>"""


def evidence_judge_node(state: State):
    prompt = EVIDENCE_JUDGE_PROMPT.format(
        question=state["question"],
        papers=state.get("papers") or "(nothing collected yet)",
    )
    response = plain_llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    decision = "MORE"
    focus = ""
    for line in text.splitlines():
        if line.startswith("DECISION:"):
            d = line.split("DECISION:")[1].strip().upper()
            if d in ("ENOUGH", "MORE"):
                decision = d
        elif line.startswith("FOCUS:"):
            f = line.split("FOCUS:")[1].strip()
            focus = "" if f.lower() == "none" else f

    retries = state.get("evidence_retries", 0)

    if decision == "MORE" and retries < MAX_EVIDENCE_RETRIES:
        return {"evidence_ok": False, "search_directive": focus, "evidence_retries": retries + 1}

    # ENOUGH, or retries exhausted - proceed either way
    return {"evidence_ok": True}


def route_evidence(state: State):
    return "summarizer_agent" if state.get("evidence_ok") else "search_agent"


# ------------------------------------------------------------- summarizer agent

SUMMARIZER_PROMPT = """You are a summarizer agent. Given a research question and raw paper
data (search results and abstracts), write a synthesized answer that cites specific papers
by arXiv ID and title. Only make claims directly supported by the abstract text provided -
do not add outside knowledge.

Question: {question}

Paper data:
{papers}
{critique_note}"""


def summarizer_agent_node(state: State):
    critique_note = ""
    if state.get("critique"):
        critique_note = (
            "\n\nThe critic reviewed a previous draft and flagged this - "
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

Check every claim in the draft summary against the source paper data above.

Then assign a citation quality score between 0.0 and 1.0:
- 1.0: every claim is directly supported by a specific abstract
- 0.7-0.9: most claims supported, minor unsupported details
- 0.4-0.6: some claims supported but notable gaps or exaggerations
- 0.0-0.3: most claims are unsupported or cannot be verified

Then decide a verdict:
- APPROVED: claims are well supported, ready to return to the user.
- REVISE: the evidence is sufficient but the summary's wording, framing, or claim
  precision needs to change.
- NEEDS_EVIDENCE: the summary makes claims the current paper data can't support at all -
  more or different papers are needed, not just a rewrite.

Respond in exactly this format:
SCORE: <number between 0.0 and 1.0>
VERDICT: APPROVED or REVISE or NEEDS_EVIDENCE
FEEDBACK: <specific feedback if REVISE or NEEDS_EVIDENCE, or 'All claims supported' if APPROVED>"""


def critic_agent_node(state: State):
    prompt = CRITIC_PROMPT.format(
        question=state["question"], papers=state["papers"], summary=state["summary"]
    )
    response = plain_llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    score = 0.5
    verdict = "REVISE"
    feedback = ""
    for line in text.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = max(0.0, min(1.0, float(line.split("SCORE:")[1].strip())))
            except ValueError:
                pass
        elif line.startswith("VERDICT:"):
            v = line.split("VERDICT:")[1].strip().upper()
            if v in ("APPROVED", "REVISE", "NEEDS_EVIDENCE"):
                verdict = v
        elif line.startswith("FEEDBACK:"):
            feedback = line.split("FEEDBACK:")[1].strip()

    history = state.get("score_history", [])
    history.append({
        "question": state["question"][:60],
        "score": score,
        "revisions": state.get("revision_count", 0),
    })

    approved = verdict == "APPROVED"
    return {
        "approved": approved,
        "verdict": verdict,
        "critique": "" if approved else feedback,
        # if the critic thinks more evidence is needed, hand its own feedback
        # straight to search_agent as the directive - no extra LLM call needed
        "search_directive": feedback if (not approved and verdict == "NEEDS_EVIDENCE") else "",
        "revision_count": state.get("revision_count", 0) + (0 if approved else 1),
        "citation_score": score,
        "score_history": history,
    }


def route_after_critic(state: State):
    if state.get("approved"):
        return "end"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "end"
    if state.get("verdict") == "NEEDS_EVIDENCE":
        return "search_agent"
    return "summarizer_agent"  # REVISE


# ------------------------------------------------------------------------- graph

graph = StateGraph(State)
graph.add_node("search_agent", search_agent_node)
graph.add_node("tools", ToolNode(ALL_TOOLS))
graph.add_node("collect_papers", collect_papers_node)
graph.add_node("evidence_judge", evidence_judge_node)
graph.add_node("summarizer_agent", summarizer_agent_node)
graph.add_node("critic_agent", critic_agent_node)

graph.add_edge(START, "search_agent")
graph.add_conditional_edges("search_agent", route_search, {"tools": "tools", "collect_papers": "collect_papers"})
graph.add_edge("tools", "search_agent")
graph.add_edge("collect_papers", "evidence_judge")
graph.add_conditional_edges(
    "evidence_judge", route_evidence, {"search_agent": "search_agent", "summarizer_agent": "summarizer_agent"}
)
graph.add_edge("summarizer_agent", "critic_agent")
graph.add_conditional_edges(
    "critic_agent",
    route_after_critic,
    {"end": END, "summarizer_agent": "summarizer_agent", "search_agent": "search_agent"},
)

app = graph.compile()


if __name__ == "__main__":
    print("Multi-agent Research Q&A Assistant (local, free). Type 'exit' to quit.\n")
    score_history = []

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
            "verdict": "",
            "approved": False,
            "revision_count": 0,
            "citation_score": 0.0,
            "score_history": score_history,
            "search_directive": "",
            "evidence_ok": False,
            "evidence_retries": 0,
            "search_step_count": 0,
            "collected_msg_idx": 0,
        }

        result = app.invoke(initial_state)
        score_history = result.get("score_history", [])

        print("\n--- Final Answer ---")
        print(result["summary"] or "(no summary was produced - check papers length below)")

        score = result.get("citation_score", 0.0)
        approved = result.get("approved", False)
        revisions = result.get("revision_count", 0)
        papers_len = len(result.get("papers", ""))

        label = "Strong" if score >= 0.8 else "Moderate" if score >= 0.5 else "Weak"

        print(f"\n Citation Quality : {score:.2f}/1.00 ({label})")
        print(f" Verdict          : {'Approved' if approved else 'Needs revision'}")
        print(f" Revisions made   : {revisions}")
        print(f" Papers collected : {papers_len} chars")

        if len(score_history) > 1:
            avg = sum(e["score"] for e in score_history) / len(score_history)
            print(f" Session average  : {avg:.2f}/1.00 ({len(score_history)} queries)")
        print()