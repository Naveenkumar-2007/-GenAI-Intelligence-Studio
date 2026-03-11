# 🧠 GenAI Intelligence Studio

**An End-to-End Multi-Agent System for Document Analysis, Product Research, Video Intel, and Deep Web Research.**

This project is a powerful **Streamlit** application orchestrated by **LangChain** and **LangGraph**. It replaces traditional vector operations with a local **FAISS** index for privacy and speed, and utilizes **Groq** for high-speed LLM inference.

## 🚀 Key Brain Modes

The application features 4 distinct "Brain Modes", each powered by specialized autonomous agents:

### 1. 📄 Doc Brain (RAG)
*Interactive Document Intelligence*
- **Capabilities**: Upload PDF/TXT files and chat with them.
- **Tech**: In-memory FAISS Vector Store, Multi-Document Retrieval.
- **Tools**: Code Explainer, Wikipedia integration.

### 2. 🚀 Product Builder (MVP Agent)
*From Idea to MVP Blueprint*
- **Capabilities**: Turn a simple prompt (e.g., "Uber for dog walking") into a full product specification.
- **Features**: Generates user personas, system architecture, tech stack recommendations, and database schema.
- **Enhanced**: Now includes **Real-time Web Search** to find actual competitors and validate market needs.

### 3. 🎥 Video Brain
*YouTube Intelligence Engine*
- **Capabilities**: Paste a YouTube URL to get summaries, chapter breakdowns, and Q&A.
- **Tech**: Recursive character splitting on transcripts, timestamp-aware context.
- **Enhanced**: Robust transcript fetching (handles auto-generated & disabled captions).

### 4. 🌐 Research Agent
*Deep Web Analyst*
- **Capabilities**: Performs multi-step research on complex topics.
- **Workflow**: Plan -> Search (DuckDuckGo) -> Scrape -> Analyze -> Report.
- **Tech**: Autonomous ReAct agent loop with recursion limits.

---

## 🛠️ Technical Stack

- **LLM**: Groq (Llama3 / Mixtral) - High-speed inference.
- **Orchestration**: LangGraph (Stateful Multi-Agent Workflow).
- **Framework**: LangChain & Streamlit.
- **Vector Store**: FAISS (CPU-based, local, privacy-first).
- **Search**: DuckDuckGo Search (Live web access).
- **Telemetry**: Custom structured logging & user memory persistence.

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd end_to_end
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment**
   Create a `.env` file:
   ```env
   GROQ_API_KEY=your_groq_key_here
   # Optional: HuggingFace Token if using gated models
   # HUGGINGFACEHUB_API_TOKEN=...
   ```

4. **Run the App**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🏗️ Architecture

The system uses a **Router Node** to classify user intent and route to the specific "Brain":

```
[User Input] --> [Router]
                    |
      +-------------+-------------+----------------+
      |             |             |                |
 [Doc Brain]  [Product Brain] [Video Brain] [Research Brain]
      |             |             |                |
  [Retrieval]   [MVP Tools]   [Transcript]    [Web Tools]
      |             |             |                |
      +-------------+-------------+----------------+
                    |
             [Writer Node] --> [Final Output]
```

## 🌟 Future Roadmap
- [ ] Persistence for FAISS index (Save/Load).
- [ ] Multi-modal upgrades (Image analysis).
- [ ] Database integration for user history.

---
*Built by [Your Name]*
