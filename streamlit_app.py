"""Streamlit UI for GenAI Intelligence Studio (Groq + Pinecone + HF)."""

import streamlit as st
from pathlib import Path
import sys
import time

# add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from src.memory.chat_history import chat_history, ChatSession


def init_session_state():
    """Initialize all session state variables."""
    if "docs_graph" not in st.session_state:
        st.session_state.docs_graph = None
    if "docs_initialized" not in st.session_state:
        st.session_state.docs_initialized = False
    if "history_docs" not in st.session_state:
        st.session_state.history_docs = []
    if "history_product" not in st.session_state:
        st.session_state.history_product = []
    if "history_video" not in st.session_state:
        st.session_state.history_video = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user_1"
    if "active_mode" not in st.session_state:
        st.session_state.active_mode = "docs"
    # Chat sessions for conversation mode
    if "docs_session" not in st.session_state:
        st.session_state.docs_session = None
    if "product_session" not in st.session_state:
        st.session_state.product_session = None
    if "video_session" not in st.session_state:
        st.session_state.video_session = None
    if "research_session" not in st.session_state:
        st.session_state.research_session = None
    if "conversation_mode" not in st.session_state:
        st.session_state.conversation_mode = False
    if "history_research" not in st.session_state:
        st.session_state.history_research = []


def get_mode_indicator(mode: str) -> str:
    """Get visual indicator for active mode."""
    indicators = {
        "docs": "📚 Doc Brain",
        "video": "🎥 Video Brain",
        "product": "🚀 Product Builder",
        "research": "🧭 Research Agent",
    }
    return indicators.get(mode, "🤖 Unknown")


def render_mode_status():
    """Render the active mode status in sidebar."""
    mode = st.session_state.active_mode
    indicator = get_mode_indicator(mode)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Active Brain")
    
    # Mode status with color
    mode_colors = {
        "docs": "🟢",
        "video": "🔵", 
        "product": "🟠",
        "research": "🟣",
    }
    color = mode_colors.get(mode, "⚪")
    st.sidebar.markdown(f"{color} **{indicator}**")
    
    # Session info
    session = None
    if mode == "docs":
        session = st.session_state.docs_session
    elif mode == "video":
        session = st.session_state.video_session
    elif mode == "product":
        session = st.session_state.product_session
    elif mode == "research":
        session = st.session_state.research_session
    
    if session:
        st.sidebar.caption(f"Session: {session.session_id[:12]}...")
        st.sidebar.caption(f"Messages: {len(session.messages)}")


def render_chat_history(history: list, mode: str):
    """Render chat history in a chat-like format."""
    if not history:
        return
    
    st.markdown("### 💬 Conversation History")
    
    for i, item in enumerate(history[-10:]):  # Show last 10
        # User message
        with st.chat_message("user"):
            st.markdown(item.get("question", ""))
        
        # Assistant message
        with st.chat_message("assistant"):
            answer = item.get("answer", "")
            if len(answer) > 500:
                st.markdown(answer[:500] + "...")
                with st.expander("Show full answer"):
                    st.markdown(answer)
            else:
                st.markdown(answer)
            
            # Metadata
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"⏱️ {item.get('time', 0):.2f}s")
            with col2:
                if "intent" in item:
                    st.caption(f"🔎 {item.get('intent')}")


def get_conversation_context(mode: str) -> str:
    """Get conversation context for follow-up questions."""
    session = None
    if mode == "docs":
        session = st.session_state.docs_session
    elif mode == "video":
        session = st.session_state.video_session
    elif mode == "product":
        session = st.session_state.product_session
    elif mode == "research":
        session = st.session_state.research_session
    
    if session and st.session_state.conversation_mode:
        return session.get_context(max_messages=6)
    return ""


@st.cache_resource
def initialize_docs_system(uploaded_files):
    """
    Initialize Pinecone vector store & graph for DOCS mode based on uploaded files.
    """
    llm = Config.get_llm()
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )

    sources: list[Path] = []

    temp_dir = Path("uploaded_docs")
    temp_dir.mkdir(exist_ok=True)

    for uf in uploaded_files:
        file_path = temp_dir / uf.name
        with file_path.open("wb") as f:
            f.write(uf.read())
        sources.append(file_path)

    # process docs
    documents = doc_processor.process_sources(sources)

    # build vectorstore (Pinecone)
    vs = VectorStore(namespace=f"docs-{int(time.time())}")
    vs.add_documents(documents)
    retriever = vs.get_retriever()

    # build graph
    gb = GraphBuilder(retriever=retriever, llm=llm)
    gb.build()

    return gb, len(documents)


def main():
    init_session_state()

    st.set_page_config(
        page_title="GenAI Intelligence Studio",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 GenAI Intelligence Studio")
    st.caption("Groq + Pinecone + HuggingFace • Agentic RAG • Multi-Agent System")

    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    st.session_state.user_id = st.sidebar.text_input(
        "User ID", value=st.session_state.user_id
    )
    
    # Conversation mode toggle
    st.session_state.conversation_mode = st.sidebar.checkbox(
        "💬 Conversation Mode",
        value=st.session_state.conversation_mode,
        help="Enable follow-up questions with context from previous messages"
    )
    
    # Render mode status
    render_mode_status()
    
    # Session management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📜 Session History")
    user_sessions = chat_history.get_user_sessions(st.session_state.user_id, limit=5)
    if user_sessions:
        for sess in user_sessions:
            st.sidebar.caption(f"• {sess['mode']}: {sess['preview'][:30]}...")
    else:
        st.sidebar.caption("No previous sessions")
    
    if st.sidebar.button("🗑️ Clear All Sessions"):
        st.session_state.history_docs = []
        st.session_state.history_product = []
        st.session_state.history_video = []
        st.session_state.history_research = []
        st.session_state.docs_session = None
        st.session_state.product_session = None
        st.session_state.video_session = None
        st.session_state.research_session = None
        st.rerun()

    tab_docs, tab_product, tab_video, tab_research = st.tabs([
        "📚 Doc Brain (RAG)", 
        "🚀 Product Builder (MVP)", 
        "🎥 Video Brain",
        "🧭 Research Agent"
    ])

    # ---------- DOC BRAIN ----------
    with tab_docs:
        st.session_state.active_mode = "docs"
        
        # Initialize chat session for docs
        if st.session_state.docs_session is None:
            st.session_state.docs_session = chat_history.create_session(
                st.session_state.user_id, "docs"
            )
        
        st.subheader("📚 Ask Your Documents (Pinecone + HF Embeddings)")
        
        # Mode indicator
        col1, col2 = st.columns([3, 1])
        with col2:
            st.info("🟢 **Doc Brain Active**")

        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, DOCX, TXT, CSV, JSON, MD, HTML, PY)",
            type=["pdf", "txt", "docx", "doc", "csv", "json", "md", "markdown", "html", "htm", "py"],
            accept_multiple_files=True,
            key="docs_uploader",
        )

        if uploaded_files:
            if not st.session_state.docs_initialized:
                with st.spinner("Indexing documents in Pinecone and initializing system..."):
                    gb, num_chunks = initialize_docs_system(uploaded_files)
                    st.session_state.docs_graph = gb
                    st.session_state.docs_initialized = True
                st.success(f"✅ Documents indexed successfully! ({num_chunks} chunks)")
            else:
                st.info("System already initialized with previously uploaded docs.")
        else:
            st.info("Upload at least one document to enable Doc Brain.")

        st.markdown("---")
        
        # Show conversation history if enabled
        if st.session_state.conversation_mode and st.session_state.history_docs:
            with st.expander("💬 Conversation History", expanded=False):
                render_chat_history(st.session_state.history_docs, "docs")

        question_docs = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., Summarize chapter 3, or explain this concept...",
            key="docs_question",
        )

        ask_clicked = st.button(
            "🔍 Ask Doc Brain",
            disabled=not (uploaded_files and question_docs),
        )

        if ask_clicked:
            if not st.session_state.docs_initialized or st.session_state.docs_graph is None:
                st.error("System is not initialized. Upload documents first.")
            else:
                # Add conversation context if enabled
                full_question = question_docs
                if st.session_state.conversation_mode:
                    context = get_conversation_context("docs")
                    if context:
                        full_question = f"Previous conversation:\n{context}\n\nNew question: {question_docs}"
                
                with st.spinner("Thinking with multi-agent ReAct pipeline..."):
                    start = time.time()
                    try:
                        result_state = st.session_state.docs_graph.run(
                            question=full_question,
                            user_id=st.session_state.user_id,
                            mode="docs",
                        )
                        elapsed = time.time() - start

                        # Save to history
                        history_item = {
                            "question": question_docs,
                            "answer": result_state.get("answer", ""),
                            "time": elapsed,
                            "intent": result_state.get("intent"),
                        }
                        st.session_state.history_docs.append(history_item)
                        
                        # Save to chat session
                        st.session_state.docs_session.add_message("user", question_docs)
                        st.session_state.docs_session.add_message("assistant", result_state.get("answer", ""))
                        chat_history.save_session(st.session_state.docs_session)

                        st.markdown("### 💡 Answer")
                        st.markdown(result_state.get("answer", ""))
                        st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")
                        st.caption(f"🔎 Detected intent: {result_state.get('intent')}")

                        with st.expander("📄 Retrieved Document Snippets", expanded=False):
                            for i, doc in enumerate(result_state.get("retrieved_docs", [])[:4], start=1):
                                st.text_area(
                                    f"Document {i}",
                                    doc.page_content[:400]
                                    + ("..." if len(doc.page_content) > 400 else ""),
                                    height=120,
                                    disabled=True,
                                )

                        if result_state.get("memory_snippet"):
                            with st.expander("🧠 Loaded Memory", expanded=False):
                                st.code(result_state.get("memory_snippet"))

                        if result_state.get("memory_to_save"):
                            with st.expander("📝 New Memory Saved", expanded=False):
                                st.code(result_state.get("memory_to_save"))
                    except Exception as e:
                        elapsed = time.time() - start
                        if "Rate limit" in str(e) or "rate_limit" in str(e):
                            st.error("🚫 **Rate Limit Reached!**")
                            st.warning("""
                            You've hit your daily token limit with Groq. Here's what you can do:
                            
                            1. **Wait**: Limits reset daily. Try again in a few hours.
                            2. **Upgrade**: Go to [Groq Console](https://console.groq.com/settings/billing) and upgrade to Dev Tier for higher limits.
                            3. **Switch Models**: Some models have different limits.
                            
                            Current usage: ~200k tokens/day on free tier.
                            """)
                        else:
                            st.error(f"Error processing question: {e}")
                        st.caption(f"⏱️ Attempt took {elapsed:.2f} seconds")

        if st.session_state.history_docs:
            st.markdown("---")
            st.markdown("### 📜 Recent Doc Brain Queries")
            for item in reversed(st.session_state.history_docs[-3:]):
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s | Intent: {item['intent']}")

    # ---------- PRODUCT BUILDER ----------
    with tab_product:
        st.session_state.active_mode = "product"
        
        # Initialize chat session for product
        if st.session_state.product_session is None:
            st.session_state.product_session = chat_history.create_session(
                st.session_state.user_id, "product"
            )
        
        st.subheader("🚀 Product Builder – Idea → MVP Blueprint")
        
        # Mode indicator
        col1, col2 = st.columns([3, 1])
        with col2:
            st.warning("🟠 **Product Builder Active**")

        idea = st.text_area(
            "Describe your product idea:",
            placeholder="e.g., Build a chatbot for student Q&A, Build an e-commerce MVP...",
            height=150,
            key="product_idea",
        )
        
        # Show conversation history if enabled
        if st.session_state.conversation_mode and st.session_state.history_product:
            with st.expander("💬 Previous Ideas", expanded=False):
                for item in st.session_state.history_product[-3:]:
                    st.markdown(f"**Idea:** {item['question'][:100]}...")

        build_clicked = st.button(
            "🚀 Generate MVP Blueprint",
            disabled=not idea,
        )

        if build_clicked:
            llm = Config.get_llm()

            # Dummy retriever (no docs needed for product mode)
            class DummyRetriever:
                def invoke(self, query: str):
                    return []

            # Always create a fresh graph for each product request
            gb_product = GraphBuilder(retriever=DummyRetriever(), llm=llm)
            gb_product.build()

            with st.spinner("🔧 Building MVP with ReAct Agent + Product Tools..."):
                start = time.time()
                try:
                    result_state = gb_product.run(
                        question=idea,
                        user_id=st.session_state.user_id,
                        mode="product",
                    )
                    elapsed = time.time() - start

                    # Save to history
                    history_item = {
                        "question": idea,
                        "answer": result_state.get("answer", ""),
                        "time": elapsed,
                    }
                    st.session_state.history_product.append(history_item)
                    
                    # Save to chat session
                    st.session_state.product_session.add_message("user", idea)
                    st.session_state.product_session.add_message("assistant", result_state.get("answer", ""))
                    chat_history.save_session(st.session_state.product_session)

                    st.markdown("### 📄 MVP Blueprint")
                    st.markdown(result_state.get("answer", ""))
                    st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

                    if result_state.get("memory_to_save"):
                        with st.expander("📝 New Memory Saved", expanded=False):
                            st.code(result_state.get("memory_to_save"))
                except Exception as e:
                    elapsed = time.time() - start
                    if "Rate limit" in str(e) or "rate_limit" in str(e):
                        st.error("🚫 **Rate Limit Reached!**")
                        st.warning("""
                        You've hit your daily token limit with Groq. Here's what you can do:
                        
                        1. **Wait**: Limits reset daily. Try again in a few hours.
                        2. **Upgrade**: Go to [Groq Console](https://console.groq.com/settings/billing) and upgrade to Dev Tier for higher limits.
                        3. **Switch Models**: Some models have different limits.
                        
                        Current usage: ~200k tokens/day on free tier.
                        """)
                    else:
                        st.error(f"Error building MVP: {e}")
                    st.caption(f"⏱️ Attempt took {elapsed:.2f} seconds")

        if st.session_state.history_product:
            st.markdown("---")
            st.markdown("### 🧾 Recent Ideas")
            for item in reversed(st.session_state.history_product[-3:]):
                st.markdown(f"**Idea:** {item['question'][:120]}...")
                st.markdown(f"**Summary:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")

    # ---------- VIDEO BRAIN ----------
    with tab_video:
        st.session_state.active_mode = "video"
        
        # Initialize chat session for video
        if st.session_state.video_session is None:
            st.session_state.video_session = chat_history.create_session(
                st.session_state.user_id, "video"
            )
        
        st.subheader("🎥 Video Brain – Understand Any YouTube Lecture")
        
        # Mode indicator
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.session_state.get("video_mode_ready"):
                st.success("🟢 **Video Ready**")
            else:
                st.info("🔵 **Upload Video First**")

        video_url = st.text_input("Enter YouTube URL:", key="video_url_input")

        if st.button("Process Video", key="process_video_btn"):
            if video_url:
                from src.video.video_processor import VideoProcessor
                vp = VideoProcessor()

                with st.spinner("🔄 Processing video transcript..."):
                    try:
                        chunks = vp.process_video(video_url)

                        # index into Pinecone (new namespace)
                        vs = VectorStore(namespace=f"video-{int(time.time())}")
                        vs.add_documents(chunks)
                        retriever = vs.get_retriever()

                        gb = GraphBuilder(retriever=retriever, llm=Config.get_llm())
                        gb.build()

                        st.session_state.video_graph = gb
                        st.session_state.video_mode_ready = True
                        # Store the video URL for later use
                        st.session_state.current_video_url = video_url
                        st.success(f"✅ Transcript indexed! ({len(chunks)} chunks)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
            else:
                st.warning("Please enter a YouTube URL.")

        st.markdown("---")
        
        # Show conversation history if enabled
        if st.session_state.conversation_mode and st.session_state.history_video:
            with st.expander("💬 Video Q&A History", expanded=False):
                for item in st.session_state.history_video[-3:]:
                    st.markdown(f"**Q:** {item['question'][:80]}...")
                    st.markdown(f"**A:** {item['answer'][:150]}...")
                    st.divider()

        ask_video = st.text_input("Ask about the video:", key="video_ask")

        if st.button("🎬 Ask Video Brain", key="ask_video_btn"):
            if not st.session_state.get("video_mode_ready"):
                st.error("Upload and process a video first.")
            elif not ask_video:
                st.warning("Please enter a question.")
            else:
                with st.spinner("🤖 Analyzing with ReAct Agent + Video Tools..."):
                    start = time.time()
                    try:
                        # Pass video_url to the graph for context
                        current_video_url = st.session_state.get("current_video_url", "")
                        result_state = st.session_state.video_graph.run(
                            question=ask_video,
                            user_id=st.session_state.user_id,
                            mode="video",
                            video_url=current_video_url,
                        )
                        elapsed = time.time() - start

                        # Save to history
                        history_item = {
                            "question": ask_video,
                            "answer": result_state.get("answer", ""),
                            "time": elapsed,
                        }
                        st.session_state.history_video.append(history_item)
                        
                        # Save to chat session
                        st.session_state.video_session.add_message("user", ask_video)
                        st.session_state.video_session.add_message("assistant", result_state.get("answer", ""))
                        chat_history.save_session(st.session_state.video_session)

                        st.markdown("### 🎥 Answer")
                        st.markdown(result_state.get("answer", ""))
                        st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

                        video_chapters = result_state.get("video_chapters", [])
                        if video_chapters:
                            with st.expander("⏱️ Chapters", expanded=False):
                                st.code("\n".join(video_chapters))

                        retrieved_docs = result_state.get("retrieved_docs", [])
                        if retrieved_docs:
                            with st.expander("📄 Retrieved Transcript Snippets", expanded=False):
                                for i, doc in enumerate(retrieved_docs[:4], start=1):
                                    st.text_area(
                                        f"Snippet {i}",
                                        doc.page_content[:300] + "...",
                                        height=100,
                                        disabled=True,
                                    )
                    except Exception as e:
                        elapsed = time.time() - start
                        if "Rate limit" in str(e) or "rate_limit" in str(e):
                            st.error("🚫 **Rate Limit Reached!**")
                            st.warning("""
                            You've hit your daily token limit with Groq. Here's what you can do:
                            
                            1. **Wait**: Limits reset daily. Try again in a few hours.
                            2. **Upgrade**: Go to [Groq Console](https://console.groq.com/settings/billing) and upgrade to Dev Tier for higher limits.
                            3. **Switch Models**: Some models have different limits.
                            
                            Current usage: ~200k tokens/day on free tier.
                            """)
                        else:
                            st.error(f"Error analyzing video: {e}")
                        st.caption(f"⏱️ Attempt took {elapsed:.2f} seconds")

    # ---------- RESEARCH AGENT ----------
    with tab_research:
        st.session_state.active_mode = "research"
        
        # Initialize chat session for research
        if "research_session" not in st.session_state or st.session_state.research_session is None:
            st.session_state.research_session = chat_history.create_session(
                st.session_state.user_id, "research"
            )
        
        st.subheader("🧭 Auto Research Agent – Live Web Research")
        
        # Mode indicator
        col1, col2 = st.columns([3, 1])
        with col2:
            st.success("🟣 **Research Agent Active**")
        
        st.markdown("""
        **What can this agent do?**
        - 🔍 Search the real web for up-to-date information
        - 💰 Compare prices across websites
        - 📊 Extract product specs, reviews, and ratings
        - 📋 Generate structured comparison tables
        - 📝 Synthesize information from multiple sources
        """)
        
        # Initialize history
        if "history_research" not in st.session_state:
            st.session_state.history_research = []

        research_question = st.text_area(
            "What do you want to research?",
            placeholder="Examples:\n• Compare latest MacBook Air M3 prices in India\n• Best budget smartphones under $300 in 2024\n• Compare React vs Vue vs Angular for enterprise apps\n• Latest reviews of Tesla Model 3",
            height=120,
            key="research_question",
        )
        
        # Show previous research if enabled
        if st.session_state.conversation_mode and st.session_state.history_research:
            with st.expander("💬 Previous Research", expanded=False):
                for item in st.session_state.history_research[-3:]:
                    st.markdown(f"**Query:** {item['question'][:100]}...")

        research_clicked = st.button(
            "🔍 Run Research",
            disabled=not research_question,
            key="run_research_btn"
        )

        if research_clicked:
            # Dummy retriever (not used in research mode)
            class DummyRetriever:
                def invoke(self, query: str):
                    return []

            llm = Config.get_llm()
            gb_research = GraphBuilder(retriever=DummyRetriever(), llm=llm)
            gb_research.build()

            with st.spinner("🌐 Performing multi-step web research... This may take a minute."):
                progress_bar = st.progress(0)
                progress_bar.progress(10, "🔍 Creating research plan...")
                
                start = time.time()
                try:
                    result_state = gb_research.run(
                        question=research_question,
                        user_id=st.session_state.user_id,
                        mode="research",
                    )
                    elapsed = time.time() - start
                    
                    progress_bar.progress(100, "✅ Research complete!")

                    answer = result_state.get("answer", "") or result_state.get("intermediate_answer", "")

                    # Save to history
                    history_item = {
                        "question": research_question,
                        "answer": answer,
                        "time": elapsed,
                    }
                    st.session_state.history_research.append(history_item)
                    
                    # Save to chat session
                    st.session_state.research_session.add_message("user", research_question)
                    st.session_state.research_session.add_message("assistant", answer)
                    chat_history.save_session(st.session_state.research_session)

                    st.markdown("### 📊 Research Results")
                    st.markdown(answer)
                    st.caption(f"⏱️ Research completed in {elapsed:.2f} seconds")
                    
                    # Show research plan if available
                    research_plan = result_state.get("research_plan", "")
                    if research_plan:
                        with st.expander("📋 Research Strategy Used", expanded=False):
                            st.markdown(research_plan)
                except Exception as e:
                    elapsed = time.time() - start
                    progress_bar.progress(100, "❌ Research failed!")
                    if "Rate limit" in str(e) or "rate_limit" in str(e):
                        st.error("🚫 **Rate Limit Reached!**")
                        st.warning("""
                        You've hit your daily token limit with Groq. Here's what you can do:
                        
                        1. **Wait**: Limits reset daily. Try again in a few hours.
                        2. **Upgrade**: Go to [Groq Console](https://console.groq.com/settings/billing) and upgrade to Dev Tier for higher limits.
                        3. **Switch Models**: Some models have different limits.
                        
                        Current usage: ~200k tokens/day on free tier.
                        """)
                    else:
                        st.error(f"Error during research: {e}")
                    st.caption(f"⏱️ Attempt took {elapsed:.2f} seconds")

        # Show recent research history
        if st.session_state.history_research:
            st.markdown("---")
            st.markdown("### 🧾 Recent Research Queries")
            for item in reversed(st.session_state.history_research[-5:]):
                with st.expander(f"🔍 {item['question'][:80]}...", expanded=False):
                    st.markdown(f"**Answer:**\n{item['answer'][:500]}...")
                    st.caption(f"Time: {item['time']:.2f}s")


if __name__ == "__main__":
    main()
