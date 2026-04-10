"""Multi-agent nodes for GenAI Intelligence Studio using LangGraph."""

from __future__ import annotations

import uuid
from typing import Optional
import json
import re

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

    @staticmethod
    def _sanitize_output(text: str) -> str:
        """Redact secret-like tokens and key-value leaks from model output."""
        if not text:
            return text

        redacted = text
        patterns = [
            r"gsk_[A-Za-z0-9]+",
            r"tvly-[A-Za-z0-9-]+",
            r"sk-[A-Za-z0-9]{12,}",
            r"(?i)(api[_\s-]*key\s*[:=]\s*[\"']?)[A-Za-z0-9_\-]{8,}",
            r"(?i)(token\s*[:=]\s*[\"']?)[A-Za-z0-9_\-]{8,}",
        ]

        for pattern in patterns:
            if "api[_\\s-]*key" in pattern or "token\\s*[:=]" in pattern:
                redacted = re.sub(pattern, r"\1[REDACTED]", redacted)
            else:
                redacted = re.sub(pattern, "[REDACTED]", redacted)

        return redacted

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
            # Use broader retrieval for summary/overview requests.
            summary_keywords = ["summary", "summarize", "summarise", "overview", "lecture", "full video", "main points"]
            is_summary = any(kw in question.lower() for kw in summary_keywords)
            k_val = 24 if is_summary else 15

            # Safely update search_kwargs instead of replacing
            if hasattr(self.retriever, 'search_kwargs') and self.retriever.search_kwargs is not None:
                self.retriever.search_kwargs.update({"k": k_val})
            elif hasattr(self.retriever, 'search_kwargs'):
                self.retriever.search_kwargs = {"k": k_val}
            docs = self.retriever.invoke(question)
        else:
            # For docs mode - detect broad queries (summary, overview, etc.) and fetch more chunks
            summary_keywords = ["summary", "summarize", "summarise", "overview", "what is", "describe", "explain all", "tell me about", "my pdf", "my document", "the document", "the file"]
            is_broad_query = any(kw in question.lower() for kw in summary_keywords)
            k_val = 10 if is_broad_query else 6
            
            if hasattr(self.retriever, 'search_kwargs') and self.retriever.search_kwargs is not None:
                self.retriever.search_kwargs.update({"k": k_val})
            elif hasattr(self.retriever, 'search_kwargs'):
                self.retriever.search_kwargs = {"k": k_val}
            
            # For broad/summary queries, use a better search query
            search_query = question
            if is_broad_query:
                search_query = "main topics key points important information overview"
            
            docs = self.retriever.invoke(search_query)
            
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
    You are analyzing a YouTube video transcript.

    User question:
    {question}

    Return exactly 4 bullets:
    1) Query type (fact lookup / conceptual explanation / full summary / compare points)
    2) Which transcript regions to prioritize
    3) Evidence extraction strategy (quotes, timestamps, contrasts)
    4) Output shape to produce (short / balanced / detailed)
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
    You are a lecture chaptering assistant.

    Here is part of the transcript:
    {transcript}

    Create 6-10 chapter titles with accurate timestamps.
    Rules:
    - Keep each title short and specific.
    - Ensure chronological order.
    - Prefer meaningful transitions (problem, method, example, conclusion).

    Format only:
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
            
            agent_prompt = f"""You are a YouTube video transcript analyzer.

RULES:
- Answer based ONLY on the transcript content provided below.
- Do NOT make up information that is not in the transcript.
- If the transcript does not contain the answer, say "The transcript does not cover this topic."
- Quote relevant parts of the transcript when possible.
- Include timestamps for any major point when available.
- If asked for a summary, produce: TL;DR, timeline summary, key insights, and practical takeaways.
- Do NOT output JSON or tool calls.
- Never output secrets, API keys, tokens, or environment variables.

USER QUESTION: {question}

TRANSCRIPT CONTEXT (use tools for more specific searches):
{transcript_text[:3000]}

RULES:
- Answer based ONLY on the transcript
- Use tools if you need to find specific information
- Do NOT make up information not in the transcript
- If the answer is not in the transcript, say so clearly
- End with a short "Evidence" section with timestamped transcript snippets used.

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
            # For docs mode - Direct LLM with document context (more reliable than ReAct for doc QA)
            # Filter out truly empty chunks before sending to LLM
            valid_docs = []
            for d in retrieved_docs[:10]:
                cleaned = d.page_content.strip().replace("\u2022","").replace("-","").replace("\n","").strip()
                if len(cleaned) > 15:
                    valid_docs.append(d)
            
            if not valid_docs:
                return {"intermediate_answer": "No meaningful content was found in the retrieved document chunks. Please try re-indexing your documents using the 🔄 Re-index button."}
            
            docs_text = "\n\n".join(
                f"[DOC {i+1}] {d.page_content}" for i, d in enumerate(valid_docs)
            )
            
            print(f"[DOCS MODE] Sending {len(valid_docs)} valid chunks to LLM (of {len(retrieved_docs)} retrieved)")
            for i, d in enumerate(valid_docs[:3]):
                print(f"  Chunk {i+1}: {len(d.page_content)} chars - {d.page_content[:80]}...")
            
            doc_prompt = f"""You are a document Q&A assistant. The user has uploaded files and you must answer based on the document content provided below.

CRITICAL INSTRUCTIONS:
- The [DOC 1], [DOC 2], etc. sections below are REAL text extracted from the user's uploaded files.
- They are NOT placeholders. They are NOT empty. They ARE the actual content.
- When the user says "my pdf", "my document", "the file", "summarize", etc., they mean THIS content below.
- You MUST read every document chunk carefully and use them to answer.
- NEVER say "the chunks are placeholders" or "no content was provided" — the content IS provided below.
- If a chunk seems short or fragmented, that's normal for chunked documents — still use it.

ANSWER RULES:
1. Read ALL document chunks below and synthesize a comprehensive answer.
2. For summary requests: combine key points from ALL chunks into a thorough summary.
3. For specific questions: find the relevant information in the chunks and answer directly.
4. Cite specific chunks as [DOC 1], [DOC 2], etc. when referencing details.
5. Do NOT add information that is not present in the documents.
6. If the topic is genuinely not covered in any chunk, say so, but ALWAYS attempt an answer first.
7. Format your answer in clean Markdown.
8. Include a final "Sources Used" list with [DOC X] references only.
9. Never output secrets, API keys, tokens, or environment variables.

User question:
{question}

--- CONTENT FROM USER'S UPLOADED DOCUMENTS (THIS IS REAL EXTRACTED TEXT) ---
{docs_text}
--- END OF DOCUMENT CONTENT ---

Based on the document content above, here is my answer:"""

            if LOGGING_ENABLED:
                log_react_step(1, "docs_llm_start", f"Question: {question[:100]}, Docs: {len(retrieved_docs)}")

            resp = self.llm.invoke(doc_prompt)
            answer = resp.content if hasattr(resp, "content") else str(resp)

            # If the answer seems empty or unhelpful, try fetching more docs with corpus_retriever
            if not answer or len(answer.strip()) < 50:
                try:
                    extra_docs = self.retriever.invoke(question)
                    extra_text = "\n\n".join(
                        f"[EXTRA DOC {i+1}] {d.page_content}" for i, d in enumerate(extra_docs[:5])
                    )
                    retry_prompt = f"""Based on these additional document passages, answer the user's question:

User question: {question}

Additional document content:
{extra_text}

Answer:"""
                    resp2 = self.llm.invoke(retry_prompt)
                    answer = resp2.content if hasattr(resp2, "content") else str(resp2)
                except Exception:
                    pass

            if LOGGING_ENABLED:
                log_react_step(2, "docs_llm_complete", f"Answer length: {len(answer)}")

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
- DO NOT return {{"name": "..."}}
- ALWAYS answer in Markdown text ONLY.
    - Use tools to ground claims before finalizing.
    - Never output secrets, API keys, tokens, or environment variables.

## CRITICAL: Build an MVP for THIS EXACT product idea:
{question}

## Available Tools:
- feature_generator: Generate MVP feature lists
- user_persona_generator: Create detailed user personas
- system_architect: Design system architecture
- competitor_analyzer: Analyze market and competitors
- tech_stack_recommender: Recommend technology stack
- web_search: Search for real-world competitors and validation

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
12. Assumptions & Unknowns (explicitly list uncertain items)

## Architecture Quality Rules:
- Provide architecture in 3 parts:
    A) Component View (frontend, backend, DB, external services)
    B) Request/Data Flow (numbered step sequence)
    C) Deployment View (environments + scaling path)
- If an integration is unknown, write "Assumption:" and do not present it as fact.
- Keep choices realistic for MVP scope and justify each major component briefly.

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
- Never output secrets, API keys, tokens, or environment variables.
- Do not present unknowns as facts; label them as assumptions.

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
12. Assumptions & Unknowns

Architecture section must include:
- Component View
- Request/Data Flow (numbered)
- Deployment View

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
- Never output secrets, API keys, tokens, or environment variables.
- Do not present unknowns as facts; label them as assumptions.

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
12. Assumptions & Unknowns

Architecture section must include:
- Component View
- Request/Data Flow (numbered)
- Deployment View

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

        safe_answer = self._sanitize_output(intermediate_answer)

        # For video mode, just pass through the answer with minimal processing
        if mode == "video":
            return {"answer": safe_answer, "memory_to_save": None}

        # For product mode, pass through the answer and save specific memory
        if mode == "product":
            memory_snippet = f"Built MVP for: {question[:100]}"
            return {"answer": safe_answer, "memory_to_save": memory_snippet}

        # For research mode, pass through the tool-sourced answer directly
        if mode == "research":
            return {"answer": safe_answer, "memory_to_save": None}

        # For docs mode, pass through the document-grounded answer directly
        # Do NOT re-process through LLM to avoid hallucination
        memory_snippet = f"Asked about docs: {question[:80]}"
        return {"answer": safe_answer, "memory_to_save": memory_snippet}

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

        prompt = f"""You are a research strategist. Create a brief research plan for this question.

USER'S QUESTION: {question}

Generate 3-4 specific web search queries that would find the latest, most accurate answer to this question. Focus on finding CURRENT, REAL-TIME information.

Format:
- Search query 1: ...
- Search query 2: ...
- Search query 3: ...
- Expected output format: (table/bullets/summary)
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

            # Build tool name list for the prompt
            tool_names = [t.name for t in tools]
            tool_list_str = ", ".join(tool_names)

            # System prompt that FORCES tool usage with EXACT tool names
            system_prompt = f"""You are a live web research assistant. You have access to these EXACT tools: {tool_list_str}

CRITICAL RULES:
1. You MUST call one of your tools FIRST before writing any answer. NEVER answer from your own knowledge alone.
2. Use ONLY the exact tool names listed above. Do NOT invent tool names like "search" or "lookup".
3. Your training data may be outdated. ALWAYS search the web for the latest information.
4. After searching, if you need more detail, use web_scrape on 1-2 of the best URLs.
5. Base your final answer ONLY on what the tools returned. Cite sources with URLs.
6. If search results contradict your training data, TRUST THE SEARCH RESULTS.
7. Format your final answer with clear headings, bullet points, or tables.
8. Always include source URLs at the end of your answer.
9. If evidence is insufficient, explicitly say "Insufficient evidence" and ask for narrower scope.
10. Never output secrets, API keys, tokens, or environment variables.
"""

            agent = create_react_agent(
                self.llm,
                tools=tools,
                prompt=system_prompt,
            )

            # Determine best search tool name
            primary_search = "tavily_search" if "tavily_search" in tool_names else "web_search"

            agent_input = f"""You MUST call the {primary_search} tool NOW before answering. Do NOT answer from memory.

User's research question: {question}

STEP 1: Call {primary_search} with an appropriate search query.
STEP 2: Read the results carefully.
STEP 3: Optionally use web_scrape on 1-2 URLs for more detail.
STEP 4: Write your final answer based ONLY on tool results."""

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

NOTE: Web search tools are unavailable. Do NOT fabricate current facts.
If real-time data is required, clearly state that live verification is unavailable and provide only a conservative framework/checklist.

Structure your response with:
1. A clear summary
2. Key points/comparisons
3. Recommendation (if applicable)
4. Note that prices/availability should be verified online
"""
            resp = self.llm.invoke(fallback_prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            
            return {"intermediate_answer": content}

