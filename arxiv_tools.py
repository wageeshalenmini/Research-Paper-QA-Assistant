import arxiv
from langchain_core.tools import tool
 
_client = arxiv.Client(delay_seconds=0.0)
 
 
@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers matching a topic or keyword query.
    Returns a numbered list of candidate papers with their arXiv ID, title,
    authors, and a short snippet of the abstract. Use this first to find
    candidate papers before reading any single one in detail."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = list(_client.results(search))
    if not results:
        return f"No papers found for query: {query}"
 
    lines = []
    for i, r in enumerate(results, start=1):
        arxiv_id = r.get_short_id()
        authors = ", ".join(a.name for a in r.authors[:3])
        if len(r.authors) > 3:
            authors += " et al."
        snippet = r.summary.replace("\n", " ")[:200]
        lines.append(
            f"{i}. [{arxiv_id}] \"{r.title}\" — {authors} ({r.published.year})\n"
            f"   abstract snippet: {snippet}..."
        )
    return "\n".join(lines)
 
 
@tool
def get_paper_abstract(arxiv_id: str) -> str:
    """Fetch the full abstract text for a specific arXiv paper by its ID
    (e.g. '2103.14030'). Use this to read a paper's full abstract before
    summarizing or citing it, rather than relying on the short snippet
    from search_arxiv."""
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(_client.results(search))
    if not results:
        return f"No paper found with ID {arxiv_id}"
    r = results[0]
    return (
        f"Title: {r.title}\n"
        f"Authors: {', '.join(a.name for a in r.authors)}\n"
        f"Published: {r.published.date()}\n"
        f"Full abstract: {r.summary}"
    )
 
 
ALL_TOOLS = [search_arxiv, get_paper_abstract]