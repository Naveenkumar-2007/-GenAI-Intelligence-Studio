"""Multi-agent nodes for GenAI Intelligence Studio using LangGraph."""

from __future__ import annotations

import uuid
from typing import Optional
import json

from langchain_core.messages import HumanMessage

from src.state.agent_state import AgentState
from src.memory.memory_store import MemoryStore

# Import logging utilities
try:
    from src.utils.logger import (
        telemetry,
        log_react_step,
        log_mode_detection,
        react_logger,
        error_logger,
    )
    LOGGING_ENABLED = True
except ImportError:
    LOGGING_ENABLED = False


class AgenticNodes:
    """
    Node functions for the multi-agent workflow:

    - router_node          → classify intent (for docs mode)
    - memory_read_node     → load user memory
    - retriever_node       → RAG retrieval (docs mode)
    - tools_node           → pre-context based on mode/intent
    - react_agent_node     → ReAct agent (docs mode, Groq + tools)
    - product_builder_node → specialized MVP generator (product mode)
    - writer_node          → clean final answer + decide memory_to_save
    - memory_write_node    → persist memory
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.memory_store = MemoryStore()

    # ---------- 1) router ----------

    def router_node(self, state: AgentState) -> dict:
        """
        Basic intent classification for docs mode.
        """
        mode = state.get("mode", "docs")
        question = state.get("question", "")
        
        # Log mode detection
        if LOGGING_ENABLED:
            log_mode_detection(mode, question)
            telemetry.track_mode(mode)
        
        # Skip detailed routing for non-docs modes
        if mode in ("product", "video", "research"):
            return {}

        prompt = (
            "You are an intent classifier for a document assistant.\n"
            "Choose ONE label (exact string):\n"
            "- code\n"
            "- news\n"
            "- general\n\n"
            f"User query: {question}\n\n"
            "Return only the label."
        )
        resp = self.llm.invoke(prompt)
        label = (resp.content or "").strip().lower()
        if label not in {"code", "news", "general"}:
            label = "general"
        
        if LOGGING_ENABLED:
            react_logger.info(f"Intent classified: {label} for query: {question[:100]}")
        
        return {"intent": label}

    # ---------- 2) memory read ----------

    def memory_read_node(self, state: AgentState) -> dict:
        # Skip memory for video mode to avoid context pollution
        if state.get("mode") == "video":
            return {"memory_snippet": None}
        snippet = self.memory_store.get_memory(state.get("user_id", "default_user"))
        return {"memory_snippet": snippet or None}

    # ---------- 3) retriever (docs/video mode) ----------

    def retriever_node(self, state: AgentState) -> dict:
        mode = state.get("mode")
        if mode not in ("docs", "video"):
            return {}
        
        question = state.get("question", "")
        
        # For video mode, get more chunks to have fuller context
        if mode == "video":
            # Use a broader search to get more transcript content
            # Safely update search_kwargs instead of replacing
            if hasattr(self.retriever, 'search_kwargs') and self.retriever.search_kwargs is not None:
                self.retriever.search_kwargs.update({"k": 15})
            elif hasattr(self.retriever, 'search_kwargs'):
                self.retriever.search_kwargs = {"k": 15}
            docs = self.retriever.invoke(question)
        else:
            # For docs mode, use default k=4
            if hasattr(self.retriever, 'search_kwargs') and self.retriever.search_kwargs is not None:
                self.retriever.search_kwargs.update({"k": 4})
            elif hasattr(self.retriever, 'search_kwargs'):
                self.retriever.search_kwargs = {"k": 4}
            docs = self.retriever.invoke(question)
            
        return {"retrieved_docs": docs}

    # ---------- 4) tools context ----------

    def tools_node(self, state: AgentState) -> dict:
        extra = ""
        mode = state.get("mode", "docs")
        intent = state.get("intent")
        question = state.get("question", "")

        if mode == "docs":
            if intent == "code":
                prompt = (
                    "User is asking about code or technical concept in documents.\n"
                    "In 3 short bullets, frame what they probably want.\n\n"
                    f"Query:\n{question}"
                )
                resp = self.llm.invoke(prompt)
                extra = f"[PRE-CODE-CONTEXT]\n{resp.content}"
        elif mode == "product":
            prompt = (
                "User wants to build a product. In 3 bullets, guess:\n"
                "1) Target users\n2) Core value\n3) Risks.\n\n"
                f"Idea:\n{question}"
            )
            resp = self.llm.invoke(prompt)
            extra = f"[PRE-PRODUCT-CONTEXT]\n{resp.content}"

        if extra:
            existing = state.get("tool_context", "") or ""
            return {"tool_context": (existing + "\n" + extra).strip()}
        return {}

    # ---------- VIDEO PRE-CONTEXT ----------

    def video_precontext_node(self, state: AgentState) -> dict:
        if state.get("mode") != "video":
            return {}

        question = state.get("question", "")
        prompt = f"""
You are analyzing a YouTube lecture.

User question:
{question}

Generate in 3 bullets:
1. What part of the video might contain the answer?
2. What reasoning style is needed?
3. What to focus on from the transcript?
"""
        resp = self.llm.invoke(prompt)
        return {"tool_context": resp.content}

    # ---------- VIDEO CHAPTER GENERATOR ----------

    def video_chapter_node(self, state: AgentState) -> dict:
        if state.get("mode") != "video":
            return {}

        retrieved_docs = state.get("retrieved_docs", [])
        transcript = "\n".join(
            f"[{d.metadata.get('timestamp_start', 0)}s] {d.page_content[:200]}"
            for d in retrieved_docs[:10]
        )

        prompt = f"""
You are a lecture summarizer.

Here is part of the transcript:
{transcript}

Create 5–8 short chapter titles with timestamps.

Format:
[0m00s] Introduction
[5m10s] Main concept
...
"""

        resp = self.llm.invoke(prompt)
        chapters = resp.content.split("\n")
        return {"video_chapters": chapters}

    # ---------- 5) ReAct agent (docs/video mode) ----------

    def react_agent_node(self, state: AgentState) -> dict:
        mode = state.get("mode")
        if mode not in ("docs", "video"):
            return {}

        retrieved_docs = state.get("retrieved_docs", [])
        question = state.get("question", "")
        mem = state.get("memory_snippet") or ""
        tool_context = state.get("tool_context") or ""

        if LOGGING_ENABLED:
            log_react_step(1, "start", f"Mode={mode}, Question={question[:100]}")

        if mode == "video":
            # For video mode - USE ReAct AGENT WITH VIDEO-SPECIFIC TOOLS
            transcript_text = "\n".join(
                f"{d.page_content}" for d in retrieved_docs[:15]
            )
            
            agent_prompt = f"""You are a YouTube video transcript analyzer with tools to search and analyze the transcript.

IMPORTANT:
- DO NOT CALL ANY TOOLS.
- DO NOT output JSON.
- DO NOT output {{"name": ...}}

You have access to these tools:
- transcript_search: Search for specific topics in the transcript
- timestamp_lookup: Find content at specific timestamps
- video_summarizer: Summarize sections of the video
- chapter_search: Find relevant chapters/sections

USER QUESTION: {question}

TRANSCRIPT CONTEXT (use tools for more specific searches):
{transcript_text[:3000]}

RULES:
- Answer based ONLY on the transcript
- Use tools if you need to find specific information
- Do NOT make up information not in the transcript
- If the answer is not in the transcript, say so clearly

Answer the user's question:"""
            
            try:
                from langgraph.prebuilt import create_react_agent
                from src.tools.tools_registry import get_tools_for_mode
                
                tools = get_tools_for_mode("video", self.retriever, self.llm)
                
                if tools:
                    agent = create_react_agent(model=self.llm, tools=tools)
                    result = agent.invoke({"messages": [HumanMessage(content=agent_prompt)]})
                    final_msg = result["messages"][-1]
                    answer = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
                    
                    if LOGGING_ENABLED:
                        log_react_step(2, "video_agent_complete", f"Messages: {len(result['messages'])}")
                else:
                    # Fallback to simple LLM if no tools
                    resp = self.llm.invoke(agent_prompt)
                    answer = resp.content if hasattr(resp, "content") else str(resp)
                    
            except Exception as e:
                if LOGGING_ENABLED:
                    error_logger.error(f"Video ReAct agent failed: {e}")
                resp = self.llm.invoke(agent_prompt)
                answer = resp.content if hasattr(resp, "content") else str(resp)
            
            return {"intermediate_answer": answer or "Could not generate answer."}
        
        else:
            # For docs mode - USE REAL ReAct AGENT WITH DOCS TOOLS
            docs_text = "\n\n".join(
                f"[DOC {i+1}] {d.page_content}" for i, d in enumerate(retrieved_docs[:6])
            )
            
            agent_prompt = f"""You are a helpful assistant with access to tools. Answer the user's question using the retrieved documents and tools available.

User question:
{question}

User memory (may be empty):
{mem}

Pre-analysis context:
{tool_context}

Retrieved documents:
{docs_text}

Available tools:
- corpus_retriever: Fetch more relevant passages from documents
- wikipedia: Search Wikipedia for external knowledge
- code_explainer: Explain code snippets

Instructions:
1. Analyze the retrieved documents carefully
2. Use corpus_retriever if you need more specific information
3. Use wikipedia for external/general knowledge
4. Use code_explainer for code-related questions
5. Provide a comprehensive, well-structured answer
6. Cite specific parts of the documents when relevant

Provide your answer:"""
            
            try:
                from langgraph.prebuilt import create_react_agent
                from src.tools.tools_registry import get_tools_for_mode
                
                tools = get_tools_for_mode("docs", self.retriever, self.llm)
                
                agent = create_react_agent(model=self.llm, tools=tools)
                
                if LOGGING_ENABLED:
                    log_react_step(2, "docs_agent_start", f"Tools: {[t.name for t in tools]}")
                
                result = agent.invoke({"messages": [HumanMessage(content=agent_prompt)]})
                
                # Log each step
                if LOGGING_ENABLED:
                    for i, msg in enumerate(result["messages"]):
                        msg_type = type(msg).__name__
                        content_preview = str(msg.content)[:100] if hasattr(msg, "content") else ""
                        log_react_step(i+1, msg_type, content_preview)
                
                final_msg = result["messages"][-1]
                answer = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
                
            except Exception as e:
                if LOGGING_ENABLED:
                    error_logger.error(f"Docs ReAct agent failed: {e}")
                print(f"ReAct agent failed, falling back to simple LLM: {e}")
                # Fallback prompt without tool mentions
                fallback_prompt = f"""You are a helpful assistant answering questions based on provided documents.

IMPORTANT RULES:
- DO NOT CALL ANY TOOLS
- DO NOT return any JSON tool call
- DO NOT return {{"name": "..."}} 
- ALWAYS answer in plain text ONLY
- STAY inside the retrieved docs ONLY

User question:
{question}

User memory (may be empty):
{mem}

Pre-analysis context:
{tool_context}

Retrieved documents:
{docs_text}

Answer based ONLY on docs:"""
                resp = self.llm.invoke(fallback_prompt)
                answer = resp.content if hasattr(resp, "content") else str(resp)

            return {"intermediate_answer": answer or "Could not generate answer."}

    # ---------- 6) product builder (product mode) ----------

    def product_builder_node(self, state: AgentState) -> dict:
        if state.get("mode") != "product":
            return {}

        question = state.get("question", "")
        user_id = state.get("user_id", "default_user")
        
        if LOGGING_ENABLED:
            log_react_step(1, "product_builder_start", f"Idea: {question[:100]}")
        
        # Product mode: memory is OPTIONAL context
        raw_mem = self.memory_store.get_memory(user_id, category="product") or ""
        mem_lines = raw_mem.strip().split("\n") if raw_mem else []
        mem = "\n".join(mem_lines[-2:]) if mem_lines else ""

        # USE ReAct AGENT WITH PRODUCT-SPECIFIC TOOLS
        agent_prompt = f"""You are an expert product manager and system architect with specialized tools.

IMPORTANT:
- DO NOT CALL ANY TOOLS.
- DO NOT return {{"name": "..."}}
- ALWAYS answer in Markdown text ONLY.

## CRITICAL: Build an MVP for THIS EXACT product idea:
{question}

## Available Tools:
- feature_generator: Generate MVP feature lists
- user_persona_generator: Create detailed user personas
- system_architect: Design system architecture
- competitor_analyzer: Analyze market and competitors
- tech_stack_recommender: Recommend technology stack

## Your Task:
Use the tools to research and then generate a complete MVP blueprint with:
1. Product Name
2. One-line Pitch
3. Target Users (use user_persona_generator)
4. Problems to Solve
5. MVP Features (use feature_generator)
6. User Journey (step-by-step)
7. System Architecture (use system_architect)
8. Database Tables
9. API Endpoints
10. Tech Stack (use tech_stack_recommender)
11. Future Features

## Previous Context (IGNORE if not relevant):
{mem}

Generate the MVP blueprint in well-formatted Markdown:"""

        try:
            from langgraph.prebuilt import create_react_agent
            from src.tools.tools_registry import get_tools_for_mode
            
            tools = get_tools_for_mode("product", None, self.llm)
            
            if tools:
                agent = create_react_agent(model=self.llm, tools=tools)
                
                if LOGGING_ENABLED:
                    log_react_step(2, "product_agent_start", f"Tools: {[t.name for t in tools]}")
                
                result = agent.invoke({"messages": [HumanMessage(content=agent_prompt)]})
                
                if LOGGING_ENABLED:
                    log_react_step(3, "product_agent_complete", f"Messages: {len(result['messages'])}")
                
                final_msg = result["messages"][-1]
                content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
            else:
                # Fallback to simple prompt without tool mentions
                fallback_prompt = f"""You are an expert product manager and system architect.

IMPORTANT:
- DO NOT CALL ANY TOOLS.
- DO NOT return {{"name": "..."}}
- ALWAYS answer in Markdown text ONLY.

## CRITICAL: Build an MVP for THIS EXACT product idea:
{question}

## Your Task:
Generate a complete MVP blueprint with:
1. Product Name
2. One-line Pitch
3. Target Users
4. Problems to Solve
5. MVP Features
6. User Journey (step-by-step)
7. System Architecture
8. Database Tables
9. API Endpoints
10. Tech Stack
11. Future Features

## Previous Context (IGNORE if not relevant):
{mem}

Generate the MVP blueprint in well-formatted Markdown:"""
                resp = self.llm.invoke(fallback_prompt)
                content = resp.content if hasattr(resp, "content") else str(resp)
                
        except Exception as e:
            if LOGGING_ENABLED:
                error_logger.error(f"Product ReAct agent failed: {e}")
            # Fallback to simple LLM without tool mentions
            fallback_prompt = f"""You are an expert product manager and system architect.

IMPORTANT:
- DO NOT CALL ANY TOOLS.
- DO NOT return {{"name": "..."}}
- ALWAYS answer in Markdown text ONLY.

## CRITICAL: Build an MVP for THIS EXACT product idea:
{question}

## Your Task:
Generate a complete MVP blueprint with:
1. Product Name
2. One-line Pitch
3. Target Users
4. Problems to Solve
5. MVP Features
6. User Journey (step-by-step)
7. System Architecture
8. Database Tables
9. API Endpoints
10. Tech Stack
11. Future Features

## Previous Context (IGNORE if not relevant):
{mem}

Generate the MVP blueprint in well-formatted Markdown:"""
            resp = self.llm.invoke(fallback_prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
        
        return {"intermediate_answer": content}

    # ---------- 7) writer node ----------

    def writer_node(self, state: AgentState) -> dict:
        """
        Clean final answer and decide what to store in memory.
        """
        mode = state.get("mode", "docs")
        intermediate_answer = state.get("intermediate_answer", "")
        question = state.get("question", "")

        # For video mode, just pass through the answer with minimal processing
        if mode == "video":
            return {"answer": intermediate_answer, "memory_to_save": None}

        # For product mode, pass through the answer and save specific memory
        if mode == "product":
            # Extract a short summary of what product was built
            memory_snippet = f"Built MVP for: {question[:100]}"
            return {"answer": intermediate_answer, "memory_to_save": memory_snippet}

        # For docs mode, enhance the answer
        mem = state.get("memory_snippet") or ""

        prompt = f"""
You are the final response writer for a multi-agent system.

MODE: {mode}

Intermediate answer:
{intermediate_answer}

User memory (may be empty):
{mem}

Your tasks:
1. Improve and clean the answer, keep Markdown format.
2. Add a final one-line summary starting with 'TL;DR:'.
3. Propose ONE short memory snippet about the user (their goals/preferences/idea).
   If nothing useful, use an empty string "".

Return JSON ONLY in this format:
{{
  "answer": "...",
  "memory_to_save": "..."
}}
"""
        resp = self.llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)

        try:
            obj = json.loads(raw)
            answer = obj.get("answer", intermediate_answer)
            memory_to_save = obj.get("memory_to_save") or None
        except Exception:
            answer = intermediate_answer
            memory_to_save = None

        return {"answer": answer, "memory_to_save": memory_to_save}

    # ---------- 8) memory write ----------

    def memory_write_node(self, state: AgentState) -> dict:
        memory_to_save = state.get("memory_to_save")
        mode = state.get("mode", "docs")
        user_id = state.get("user_id", "default_user")
        
        if memory_to_save:
            # Save with appropriate category for better retrieval
            self.memory_store.save_memory(
                user_id=user_id,
                snippet=memory_to_save,
                category=mode,
            )
            
            if LOGGING_ENABLED:
                react_logger.info(f"Memory saved for user {user_id}: {memory_to_save[:100]}")
        
        return {}

    # ---------- 9) RESEARCH PRE-CONTEXT NODE ----------

    def research_precontext_node(self, state: AgentState) -> dict:
        """
        Analyze research question and generate a research plan/strategy.
        Only runs for research mode.
        """
        if state.get("mode") != "research":
            return {}

        question = state.get("question", "")

        prompt = f"""You are a senior research strategist and web analyst.

USER'S RESEARCH QUESTION:
{question}

Your task is to create a detailed research plan. Think step by step:

1. INFORMATION NEEDED: What specific data points are required? (prices, specs, reviews, comparisons)

2. SOURCE STRATEGY: Which types of websites are most likely to have accurate, up-to-date information?
   - E-commerce sites (Amazon, Flipkart, Best Buy)
   - Review sites (GSMArena, TechRadar, CNET)
   - Official product pages
   - Comparison aggregators
   - News/blog articles

3. SEARCH QUERIES: What 2-3 specific search queries would yield the best results?

4. OUTPUT FORMAT: How should the final answer be structured?
   - Comparison table
   - Bullet points with pros/cons
   - Price breakdown
   - Summary recommendation

Provide a clear, actionable research plan in 5-7 bullet points.
"""
        
        if LOGGING_ENABLED:
            react_logger.info(f"[RESEARCH] Creating research plan for: {question[:100]}")

        resp = self.llm.invoke(prompt)
        plan = resp.content if hasattr(resp, "content") else str(resp)
        
        existing = state.get("tool_context") or ""
        merged = (existing + "\n[RESEARCH-PLAN]\n" + plan).strip()

        return {"tool_context": merged, "research_plan": plan}

    # ---------- 10) RESEARCH AGENT NODE (ReAct with web tools) ----------

    def research_agent_node(self, state: AgentState) -> dict:
        """
        Auto Research Agent using web_search + web_scrape tools.
        Performs multi-step web research with tool calling.
        """
        if state.get("mode") != "research":
            return {}

        question = state.get("question", "")
        plan = state.get("research_plan") or state.get("tool_context") or ""

        if LOGGING_ENABLED:
            react_logger.info(f"[RESEARCH] Starting research agent for: {question[:100]}")

        try:
            from langgraph.prebuilt import create_react_agent
            from src.tools.tools_registry import build_research_tools
            
            tools = build_research_tools()
            
            if not tools:
                raise ValueError("No research tools available")

            # Simplified system prompt to avoid infinite loops
            system_prompt = """You are a web research assistant. Follow these steps EXACTLY:

STEP 1: Use web_search to find 3-5 relevant URLs for the query.
STEP 2: Use web_scrape on the 1-2 BEST URLs to get detailed content.
STEP 3: Summarize your findings and provide a clear answer.

RULES:
- Do NOT search more than once
- Do NOT scrape more than 2 URLs
- After getting content, STOP and provide your final answer
- Format your answer with bullet points or tables
- If you can't find info, say so and provide what you know
"""

            agent = create_react_agent(
                self.llm,
                tools=tools,
                prompt=system_prompt,
            )

            agent_input = f"""Research this question and provide a helpful answer:

{question}

Remember: Search once, scrape 1-2 URLs max, then give your final answer."""

            if LOGGING_ENABLED:
                log_react_step(1, "research_agent_start", f"Query: {question[:100]}")

            # Add recursion limit to prevent infinite loops
            result = agent.invoke(
                {"messages": [HumanMessage(content=agent_input)]},
                config={"recursion_limit": 15}
            )
            
            messages = result.get("messages", [])
            
            if LOGGING_ENABLED:
                log_react_step(2, "research_agent_complete", f"Messages: {len(messages)}")
            
            # Extract final answer
            answer = ""
            if messages:
                final_msg = messages[-1]
                answer = getattr(final_msg, "content", "") or str(final_msg)

            return {"intermediate_answer": answer}

        except Exception as e:
            if LOGGING_ENABLED:
                error_logger.error(f"Research agent failed: {e}")
            print(f"Research agent error: {e}")
            
            # Fallback: Simple LLM response
            fallback_prompt = f"""You are a research assistant. The user asked:

{question}

Research Plan:
{plan}

NOTE: Web search tools are unavailable. Provide the best answer you can based on your training knowledge. Be clear that this is from your knowledge base, not live web data.

Structure your response with:
1. A clear summary
2. Key points/comparisons
3. Recommendation (if applicable)
4. Note that prices/availability should be verified online
"""
            resp = self.llm.invoke(fallback_prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            
            return {"intermediate_answer": content}

