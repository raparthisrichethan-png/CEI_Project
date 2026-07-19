# AI-Powered Intelligent Hiring Assistant

An internship-level, production-inspired Generative AI application built to automate resume screening, semantic search, ranking, and candidate evaluation against Job Descriptions (JDs). This application uses local NLP and vector similarity models alongside an optimized cloud RAG architecture (Groq API - Llama 3.1 8B) to provide explainable insights and resume feedback while minimizing API token usage.

---

## Key Features

### 🏢 Recruiter Dashboard
- **Structured JD Extraction**: Paste or upload a JD; the AI parses it into structured sections (Skills, Experience, Education, Projects, Certifications).
- **Candidate Ranking**: Upload resumes to rank them dynamically against the JD requirements.
- **Custom Weight Tuning**: Change matching criteria weights dynamically (e.g. prioritizing Skills over Education) and see rankings update in real-time.
- **Talent Pool Semantic Search**: Search and rank candidates from a pre-loaded local resume dataset (Kaggle Resume Dataset format).
- **Side-by-Side Comparison**: Compare candidates' profile shapes, sub-scores, strengths, and AI insights side-by-side.
- **Analytics Visualization**: Interactive Plotly radar charts comparing matching profiles of selected candidates.

### 👤 Candidate Dashboard
- **Resume Semantic Chunking**: Upload a resume (PDF/DOCX) to split text into overlapping semantic chunks, mapping them to profile dimensions locally.
- **ATS Optimizer Checklist**: Review structural check results (presence of email, phone, social links, length heuristics) and receive an ATS Score out of 100.
- **AI Recommendation Engine**: Generate structured lists of resume strengths, weaknesses, and direct rewrite advice.
- **Personalized Learning Roadmap**: Receive step-by-step technical learning phases, project recommendation templates, and high-value industry certifications to bridge gaps.
- **Conversational RAG Chatbot**: Chat with an AI assistant that answers questions like *"Why is my score low?"* or *"What projects should I build?"* using the RAG pipeline context.

---

## Token Optimization Strategies
To minimize external API dependency and cost, this project enforces:
1. **Local Text Extraction**: Parsing is done entirely locally using PyMuPDF and python-docx.
2. **Local Embedding & Similarity Engines**: Sentence embeddings (`all-MiniLM-L6-v2`) and vector search (`FAISS`) run locally. Sub-score matching calculations are performed entirely offline using vector cosine similarity.
3. **Double Caching Layer (SQLite)**:
   - **Embedding Cache**: Persists generated text embeddings to a local SQLite database (`data/cache/embeddings_cache.db`), ensuring subsequent comparisons are instantaneous.
   - **LLM Response Cache**: Persists LLM responses (`data/cache/llm_response_cache.db`) for matching query/context hashes to avoid re-generating.
4. **Context-Only RAG Prompts**: Never sends complete raw resumes or JDs to the LLM. Only structured context strings and target metrics are sent to Groq to generate final explanations.

---

## Tech Stack
- **Frontend**: Streamlit (White Theme / Light Mode)
- **Backend Orchestration**: Python, LangChain
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local)
- **Vector Store**: `FAISS` (Local)
- **Large Language Model**: Llama 3.1 8B via Groq API
- **Document Parsers**: PyMuPDF (`fitz`), `python-docx`
- **Data Engineering**: Pandas, NumPy, SQLite, Plotly

---

## Installation & Setup

1. **Clone the Repository** (or open the workspace directory):
   ```bash
   cd CEI_PROJECT
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory (or enter your key in the Streamlit UI Sidebar):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Verify the Backend Integration**:
   Run the verification test script to ensure parsing, local embeddings, SQLite cache, and FAISS function correctly:
   ```bash
   python verify_backend.py
   ```

5. **Run the Streamlit App**:
   ```bash
   streamlit run frontend/app.py
   ```

---

## System Architecture Flowchart
```
Document Upload (Resume/JD)
      │
      ▼
Parser (Local PyMuPDF/docx) ──► Local Semantic Chunking Engine
      │
      ▼
Embedding Engine (Local Sentence-Transformers) ◄──► SQLite Embedding Cache
      │
      ▼
FAISS Vector Store (Local Search)
      │
      ▼
Similarity Engine (Local Cosine Match Metrics)
      │
      ▼
RAG Context Builder (Top-K Chunks)
      │
      ▼
Llama 3.1 8B via Groq API ◄──► SQLite Response Cache
      │
      ▼
Dashboard Display (Gauges, Tables, Charts, Roadmap, Chatbot)
```
