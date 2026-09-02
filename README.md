# Research Paper Q&A Assistant

Two LangGraph implementations of the same task - answering a research question by
searching arXiv, reading abstracts, and writing a cited summary - built against a
local Ollama `llama3.1` model. They differ in how much of the decision-making is
delegated to the LLM versus handled by deterministic code, and that difference is
the whole point: this repo is as much a record of *why* the architecture ended up
this way as it is working code.

| | `single_agent_reliable_graph.py` | `multi_agent_graph_local_free.py` |
|---|---|---|
| LLM roles | 1 (search) | 4 (search, evidence judge, summarizer, critic) |
| Routing decided by | Code, with one bounded retry loop | Code, translating two LLM judgment calls |
| Dependencies | `langchain-ollama`, `langgraph`, `arxiv_tools` | same |
| Cost | Free, fully local | Free, fully local |
| Best for | Fast, simple answers with citation grounding | Higher-quality answers via a critique/revise loop |

Both require **Ollama running locally with `llama3.1` pulled**, and your own
`arxiv_tools.py` providing `ALL_TOOLS` (expected to include something like
`search_arxiv` and `get_paper_abstract`) in the same directory.

```
pip install langchain-ollama langgraph
ollama pull llama3.1
python single_agent_reliable_graph.py
# or
python multi_agent_graph_local_free.py
```

---

## 1. `single_agent_reliable_graph.py`

**Architecture:** one real agent (a ReAct loop that calls search tools and decides
when it's done), wrapped in deterministic scaffolding that catches the failure
modes a bare agent loop is prone to. No other node makes a decision an LLM could
have made differently - every non-agent node is a regex, a string comparison, or a
counter.

```
normalize_query -> agent <-> tools -> grounding_check -> format_answer
                                            |
                                    (bounded retry back to agent)
```

- **`normalize_query_node`** - one plain LLM call that rewrites a vague question
  into a tighter arXiv search query before the agent starts.
- **`agent_node`** - the ReAct loop. Capped at `MAX_AGENT_STEPS` (default 6); past
  that it's force-finalized by invoking the model *without* tools bound, so it
  physically cannot keep calling tools.
- **`grounding_check_node`** - deterministic, no LLM: extracts every arXiv ID cited
  in the final answer via regex, compares it against IDs that actually appear in
  the tool call results, and flags anything cited but never actually retrieved.
  Bounded by `MAX_GROUNDING_RETRIES` (default 2).
- **`format_answer_node`** - deterministic: builds a references list from the
  intersection of cited and verified IDs, and appends a caveat for anything that
  couldn't be verified.

**Known limitation:** `ARXIV_ID_RE` only matches modern `YYMM.NNNNN`-style IDs. If
your `arxiv_tools` corpus includes old-style IDs (`hep-th/9901001`), extend the
regex or the grounding check will flag legitimate citations as unverified.

---

## 2. `multi_agent_graph_local_free.py`

**Architecture:** four LLM roles, each with a narrow job, orchestrated by code that
translates two focused LLM judgment calls into graph transitions. This is the
result of an earlier version failing: a single central "supervisor" node asked to
choose among 4 possible next agents *and* write a directive in one call turned out
to be unreliable against `llama3.1` - it would silently keep re-dispatching search
even after solid evidence had already been collected. Splitting that one overloaded
decision into two much narrower ones (see below) fixed it.

```
search_agent <-> tools -> collect_papers -> evidence_judge -+-> search_agent (retry)
                                                              +-> summarizer_agent -> critic_agent -+-> END
                                                                                                       +-> summarizer_agent (revise)
                                                                                                       +-> search_agent (needs evidence)
```

- **`search_agent_node`** - ReAct loop, same shape as the single-agent version.
  Capped at `MAX_SEARCH_STEPS` (default 6).
- **`collect_papers_node`** - hand-off boundary: the summarizer and critic never
  see the search agent's private tool-call reasoning, only this distilled fact
  sheet. Also runs a **deterministic keyword-relevance filter**: results that don't
  contain any content word from the question are dropped before they ever reach
  the rest of the pipeline. This exists because `evidence_judge` was tested asking
  the LLM to judge topical relevance directly, and it reliably failed to - it kept
  saying evidence was sufficient even when most of it was off-topic (scientometrics
  papers showing up for a telecommunications query). Keyword matching against the
  question's own terms turned out to be more reliable than LLM judgment for this
  specific check.
- **`evidence_judge_node`** - one binary LLM decision: `ENOUGH` or `MORE`, with an
  optional one-line `FOCUS` for what's missing. Deliberately narrow - this is the
  direct fix for the earlier 4-way supervisor being too much for the model to
  judge reliably. Bounded by `MAX_EVIDENCE_RETRIES` (default 2).
- **`summarizer_agent_node`** - single LLM call, writes the cited summary.
- **`critic_agent_node`** - single LLM call, fact-checks the summary against the
  collected evidence and returns a `SCORE` (0.0-1.0), a `VERDICT`
  (`APPROVED` / `REVISE` / `NEEDS_EVIDENCE`), and `FEEDBACK`. If the verdict is
  `NEEDS_EVIDENCE`, its own feedback text is passed straight to `search_agent` as
  the next directive - no extra LLM call needed to translate one into the other.
  Bounded by `MAX_REVISIONS` (default 3).

**Known limitations:**
- **Score and verdict are independent judgments, not derived from one another.**
  The critic can rate evidence 0.80 while still returning `REVISE` for an
  unrelated reason. A run that hits `MAX_REVISIONS` prints "Needs revision" even
  if the underlying answer is already solid - the cap stopped it, not disapproval.
  If you want a high score to auto-pass regardless of the verdict label, add a
  threshold check in `route_after_critic` (e.g. `score >= 0.75` routes to `end`
  even if `verdict == "REVISE"`).
- **The keyword filter is blunt.** It's a literal substring match against the
  question's own content words after stripping a small stopword list. A paper
  that's genuinely on-topic but uses different phrasing than the question (e.g.
  "cellular systems" for a question about "telecommunication") will be dropped as
  a false negative. This trades recall for precision deliberately, given that the
  LLM-judgment alternative was tested and found less reliable - but it's worth
  knowing the trade exists. A local embedding-similarity check (e.g.
  `sentence-transformers`) would catch paraphrases while staying free and local,
  at the cost of one more dependency.
- **Two LLM calls run at `temperature=0`** (`evidence_judge`, `critic`), so
  identical state produces identical output. This is useful for debugging - if a
  rerun produces byte-identical results, no LLM decision actually changed - but it
  also means don't expect variety across repeated runs of the same question.

---

## Tuning knobs

| Constant | File | Default | Effect |
|---|---|---|---|
| `MAX_AGENT_STEPS` | single-agent | 6 | Tool round-trips before forced finalization |
| `MAX_GROUNDING_RETRIES` | single-agent | 2 | Retries if citations can't be verified |
| `MAX_SEARCH_STEPS` | multi-agent | 6 | Tool round-trips per search dispatch |
| `MAX_EVIDENCE_RETRIES` | multi-agent | 2 | Retries if evidence judged insufficient |
| `MAX_REVISIONS` | multi-agent | 3 | Retries if critic returns REVISE/NEEDS_EVIDENCE |

Raising any of these trades local inference time for more attempts at a clean
result; none of them cost money since both implementations run entirely on local
Ollama.

## Design principle behind both files

Every node in both graphs falls into one of two categories: an LLM making a
decision that changes control flow (an agent), or code executing a fixed rule
(a workflow step). The guiding rule used throughout: **push a decision to
deterministic code whenever code can make it reliably, and reserve LLM calls for
judgments that genuinely need language understanding.** Both files are the result
of testing that boundary against `llama3.1` directly, rather than assuming where
it should sit - the keyword filter and the binary evidence check exist because
broader LLM judgment calls were tried first and found unreliable in practice.
