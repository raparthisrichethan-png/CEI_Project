import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to python path to resolve 'backend' imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from backend.rag_pipeline import RAGPipeline
from frontend.recruiter_view import run_recruiter_view
from frontend.candidate_view import run_candidate_view

# Initialize RAG Pipeline (cached across page reloads using st.cache_resource)
@st.cache_resource
def get_rag_pipeline(api_key=None):
    return RAGPipeline(api_key=api_key)

def main():
    st.set_page_config(
        page_title="AI-Powered Intelligent Hiring Assistant",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        
        .subtitle {
            font-family: 'Inter', sans-serif;
            color: #475569;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title block
    st.markdown('<h1 class="main-title">💼 AI-Powered Intelligent Hiring Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Production-grade GenAI Resume Screening, Semantic Search & Explainable Ranking Engine</p>', unsafe_allow_html=True)

    # 1. Sidebar - Setup & Navigation
    st.sidebar.image("https://img.icons8.com/clouds/100/resume.png", width=80)
    st.sidebar.title("Configuration & Navigation")
    
    # API Key handling
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        help="Input your Google Gemini API Key. If set in terminal environment variables, it will auto-populate."
    )
    
    # Initialize pipeline
    rag_pipeline = get_rag_pipeline()
    if api_key:
        rag_pipeline.set_api_key(api_key)
        st.sidebar.success("Gemini API Key configured!")
    else:
        st.sidebar.warning("No API Key detected. RAG features will be disabled. You can still use local semantic matching.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Navigation")
    dashboard_role = st.sidebar.radio(
        "Select User Role:",
        ["Recruiter Dashboard", "Candidate Dashboard"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Architecture Details")
    st.sidebar.markdown("""
    - **Resume Parser**: PyMuPDF & python-docx (Local)
    - **Embeddings**: Sentence Transformers `all-MiniLM-L6-v2` (Local)
    - **Vector Matching & DB**: Cosine Similarity & FAISS (Local)
    - **Orchestration**: LangChain Framework
    - **LLM**: Gemini 3.5 Flash (Cloud API)
    - **RAG Architecture**: Local Vector Retrieval + Cloud LLM
    """)

    # 2. Main Area navigation routes
    if dashboard_role == "Recruiter Dashboard":
        run_recruiter_view(rag_pipeline)
    else:
        run_candidate_view(rag_pipeline)

if __name__ == "__main__":
    main()
