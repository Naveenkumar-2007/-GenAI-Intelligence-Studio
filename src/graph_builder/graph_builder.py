"""Graph builder for multi-agent Agentic RAG workflow."""

from langgraph.graph import StateGraph, END

from src.state.agent_state import AgentState
from src.node.agentic_nodes import AgenticNodes


def _route_by_mode(state: dict) -> str:
    """Route to different paths based on mode."""
    mode = state.get("mode", "docs")
    if mode == "product":
        return "product"
    elif mode == "video":
        return "video"
    elif mode == "research":
        return "research"
    else:
        return "docs"


class GraphBuilder:
    """Builds and runs the multi-agent LangGraph workflow."""

    def __init__(self, retriever, llm):
        self.nodes = AgenticNodes(retriever, llm)
        self.graph = None

    def build(self):
        builder = StateGraph(AgentState)

        # Add all nodes
        builder.add_node("router", self.nodes.router_node)
        builder.add_node("memory_read", self.nodes.memory_read_node)
        builder.add_node("retriever", self.nodes.retriever_node)
        builder.add_node("tools", self.nodes.tools_node)
        builder.add_node("video_precontext", self.nodes.video_precontext_node)
        builder.add_node("video_chapters", self.nodes.video_chapter_node)
        builder.add_node("react_agent", self.nodes.react_agent_node)
        builder.add_node("product_builder", self.nodes.product_builder_node)
        builder.add_node("writer", self.nodes.writer_node)
        builder.add_node("memory_write", self.nodes.memory_write_node)
        
        # NEW: Research nodes
        builder.add_node("research_precontext", self.nodes.research_precontext_node)
        builder.add_node("research_agent", self.nodes.research_agent_node)

        # Entry point
        builder.set_entry_point("router")

        # CONDITIONAL ROUTING based on mode
        # Product mode bypasses RAG entirely
        # Research mode uses web search/scrape tools
        # Video and Docs modes go through RAG pipeline
        builder.add_conditional_edges(
            "router",
            _route_by_mode,
            {
                "docs": "memory_read",
                "video": "memory_read",
                "product": "product_builder",
                "research": "research_precontext",
            }
        )

        # Docs/Video path: memory → retriever → video nodes → tools → react_agent → writer
        builder.add_edge("memory_read", "retriever")
        builder.add_edge("retriever", "video_precontext")
        builder.add_edge("video_precontext", "video_chapters")
        builder.add_edge("video_chapters", "tools")
        builder.add_edge("tools", "react_agent")
        builder.add_edge("react_agent", "writer")

        # Product path: product_builder → writer
        builder.add_edge("product_builder", "writer")
        
        # Research path: research_precontext → research_agent → writer
        builder.add_edge("research_precontext", "research_agent")
        builder.add_edge("research_agent", "writer")

        # Common ending: writer → memory_write → END
        builder.add_edge("writer", "memory_write")
        builder.add_edge("memory_write", END)

        self.graph = builder.compile()
        return self.graph

    def run(self, question: str, user_id: str, mode: str, video_url: str = "") -> dict:
        """Run the graph with the given inputs."""
        if self.graph is None:
            self.build()
        
        init_state = {
            "question": question,
            "user_id": user_id,
            "mode": mode,
            "intent": None,
            "retrieved_docs": [],
            "tool_context": "",
            "intermediate_answer": "",
            "answer": "",
            "memory_snippet": None,
            "memory_to_save": None,
            "video_url": video_url,
            "video_chapters": [],
            # Research mode fields
            "research_links": [],
            "research_raw_contents": [],
            "research_plan": "",
        }
        result = self.graph.invoke(init_state)
        return result
