from typing import Annotated
from typing_extensions import TypedDict
 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
 
from arxiv_tools import ALL_TOOLS
 
 
class State(TypedDict):
    messages: Annotated[list, add_messages]
 
 
SYSTEM_PROMPT = """You are a research assistant. When asked a research question:
1. Use search_arxiv to find relevant candidate papers.
2. Use get_paper_abstract on the most relevant 2-4 papers to read their full abstracts.
3. Write a synthesized answer that cites specific papers by arXiv ID and title.
Do not state a claim about a paper without having actually read its abstract via the tools.
If the search results are thin or off-topic, say so rather than filling gaps from general knowledge."""
 
llm = ChatOllama(model="llama3.1", temperature=0).bind_tools(ALL_TOOLS)
 
 
def agent_node(state: State):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    return {"messages": [llm.invoke(messages)]}
 
 
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(ALL_TOOLS))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
app = graph.compile()
 
 
if __name__ == "__main__":
    print("Research Q&A Assistant (local model). Type 'exit' to quit.\n")
    while True:
        question = input("Ask a research question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        result = app.invoke({"messages": [HumanMessage(content=question)]})
        print("\n" + result["messages"][-1].content + "\n")
 