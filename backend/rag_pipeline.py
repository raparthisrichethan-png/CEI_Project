import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.embedding_engine import EmbeddingEngine
from backend.config import (
    DEFAULT_LLM_MODEL, 
    ANALYSIS_PROMPT_TEMPLATE, 
    ROADMAP_PROMPT_TEMPLATE,
    CHAT_PROMPT_TEMPLATE
)

class RAGPipeline:
    """
    Orchestrates the retrieval-based text generation pipeline (RAG) using the
    LangChain framework to call Gemini models via ChatGoogleGenerativeAI.
    """
    
    def __init__(self, api_key=None):
        # Read the API key from argument or environment variables
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        
        # Initialize LangChain ChatGoogleGenerativeAI Client if API key is provided
        if self.api_key:
            self.client = ChatGoogleGenerativeAI(
                model=DEFAULT_LLM_MODEL,
                google_api_key=self.api_key,
                temperature=0.0
            )

    def has_api_key(self):
        """Check if an API Key is configured."""
        return self.api_key is not None and len(self.api_key.strip()) > 0

    def set_api_key(self, api_key):
        """Update API key dynamically at runtime."""
        self.api_key = api_key
        if api_key:
            self.client = ChatGoogleGenerativeAI(
                model=DEFAULT_LLM_MODEL,
                google_api_key=api_key,
                temperature=0.0
            )
        else:
            self.client = None

    def _retrieve_context(self, parsed_resume, parsed_jd, query, k=4):
        """
        Chunk and index parsed resume and JD into a local temporary FAISS vector store,
        retrieving the most semantically relevant chunks for context.
        This represents the local vector search component of the RAG pipeline.
        """
        documents = []

        # 1. Add structured resume sections as documents
        for sec in ['skills', 'experience', 'education', 'projects', 'certifications']:
            content = parsed_resume.get(sec, "").strip()
            if content:
                documents.append(Document(
                    page_content=f"Candidate Resume Section [{sec.upper()}]: {content}",
                    metadata={"source": "resume", "section": sec}
                ))

        # 2. Add structured JD sections as documents
        for sec in ['skills', 'experience', 'education', 'projects', 'certifications']:
            content = parsed_jd.get(sec, "").strip()
            if content:
                documents.append(Document(
                    page_content=f"Job Description Section [{sec.upper()}]: {content}",
                    metadata={"source": "jd", "section": sec}
                ))

        # 3. Chunk and add raw candidate resume text for deep semantic match
        raw_text = parsed_resume.get('text', "").strip()
        if raw_text:
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
            chunks = splitter.split_text(raw_text)
            for chunk in chunks:
                documents.append(Document(
                    page_content=f"Candidate Resume Text Chunk: {chunk}",
                    metadata={"source": "resume", "section": "raw_text"}
                ))

        if not documents:
            return "No profile or job details available for context retrieval."

        try:
            # Retrieve the model using our EmbeddingEngine lazy loader
            embedding_model = EmbeddingEngine.get_model()
            
            # Create a dynamic local FAISS database using local HuggingFace embeddings
            db = FAISS.from_documents(documents, embedding_model)
            retriever = db.as_retriever(search_kwargs={"k": min(len(documents), k)})
            
            # Search context
            docs = retriever.invoke(query)
            context_str = "\n\n".join([doc.page_content for doc in docs])
            return context_str
        except Exception as e:
            print(f"Error in FAISS local vector retrieval: {e}. Falling back to default formatting.")
            # Fallback to direct serialization
            return f"Resume Skills: {parsed_resume.get('skills', '')}\nResume Experience: {parsed_resume.get('experience', '')}\nJD Requirements: {parsed_jd.get('skills', '')} - {parsed_jd.get('experience', '')}"

    def extract_jd_requirements(self, jd_text):
        """
        Parse raw Job Description text into structured requirements.
        If no API Key is available, uses a basic fallback structure.
        """
        if not self.has_api_key():
            # Basic fallback structure if no API key is present
            return {
                "skills": jd_text[:500],
                "experience": jd_text,
                "education": "Not specified",
                "projects": "Not specified",
                "certifications": "Not specified"
            }

        # prompt to ask Gemini to structure the Job Description
        prompt = f"""
        You are a job description parser. Analyze the following Job Description and extract the key requirements for these five sections:
        1. skills (core technologies, technical skills, tools, programming languages, soft skills)
        2. experience (years of experience, specific roles, industry experience, preferred domain experience)
        3. education (degree requirements, field of study, academic benchmarks)
        4. projects (recommended portfolio items, systems or applications requested, coding challenges)
        5. certifications (industry standard certifications, licenses)

        Format the output STRICTLY as a JSON object with keys: "skills", "experience", "education", "projects", "certifications".
        Ensure the values are clean strings summarizing the requirements.
        Do not output any markdown formatting like ```json or trailing text. Output raw JSON string only.

        Job Description:
        {jd_text}
        """

        try:
            # Query Gemini via LangChain wrapper
            response = self.client.invoke(prompt)
            text = response.content.strip()
            
            # Clean markdown code blocks if Gemini returned them
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            parsed = json.loads(text)
            return parsed
        except Exception as e:
            print(f"Error parsing JD via LLM: {e}. Using fallback.")
            return {
                "skills": "Extracting skills failed. " + jd_text[:200],
                "experience": jd_text[:200],
                "education": "Not parsed",
                "projects": "Not parsed",
                "certifications": "Not parsed"
            }

    def generate_analysis(self, parsed_resume, parsed_jd):
        """Generate structured screening feedback (strengths, gaps, ATS, suggestions)."""
        if not self.has_api_key():
            return "### API Key Missing\nPlease provide a Gemini API Key in the sidebar to generate AI feedback."

        resume_context = f"""
        Skills: {parsed_resume.get('skills', '')}
        Experience: {parsed_resume.get('experience', '')}
        Education: {parsed_resume.get('education', '')}
        Projects: {parsed_resume.get('projects', '')}
        Certifications: {parsed_resume.get('certifications', '')}
        """

        jd_context = f"""
        Skills Required: {parsed_jd.get('skills', '')}
        Experience Required: {parsed_jd.get('experience', '')}
        Education Required: {parsed_jd.get('education', '')}
        Projects Required: {parsed_jd.get('projects', '')}
        Certifications Required: {parsed_jd.get('certifications', '')}
        """

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            resume_context=resume_context,
            jd_context=jd_context
        )

        try:
            # Generation via LangChain wrapper
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error calling Gemini: {e}"

    def generate_roadmap(self, parsed_resume, parsed_jd):
        """Generate learning roadmap, certification, and project recommendations."""
        if not self.has_api_key():
            return "### API Key Missing\nPlease provide a Gemini API Key in the sidebar to generate your learning roadmap."

        resume_context = f"""
        Skills: {parsed_resume.get('skills', '')}
        Projects: {parsed_resume.get('projects', '')}
        Certifications: {parsed_resume.get('certifications', '')}
        """

        jd_context = f"""
        Skills Required: {parsed_jd.get('skills', '')}
        Projects Required: {parsed_jd.get('projects', '')}
        Certifications Required: {parsed_jd.get('certifications', '')}
        """

        prompt = ROADMAP_PROMPT_TEMPLATE.format(
            resume_context=resume_context,
            jd_context=jd_context
        )

        try:
            # Generation via LangChain wrapper
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error calling Gemini: {e}"

    def generate_comparison(self, parsed_jd, selected_candidates):
        """Generate side-by-side executive candidate comparison using LangChain LLM."""
        if not self.has_api_key():
            return "### API Key Missing\nPlease provide a Gemini API Key in the sidebar to generate comparative analysis."

        comparison_context = ""
        for c in selected_candidates:
            comparison_context += f"""
            Candidate: {c['name']}
            Scores: Overall={c['overall_score']:.1f}%, Skills={c['skills_match']:.1f}%, Experience={c['experience_match']:.1f}%
            Extracted Experience: {c['parsed_resume'].get('experience', '')[:400]}...
            Extracted Skills: {c['parsed_resume'].get('skills', '')[:300]}
            ---
            """
            
        prompt = f"""
        You are an executive HR consultant. Provide a brief side-by-side comparative analysis of the following candidate fits for the target role:
        Target JD Requirements:
        - Skills: {parsed_jd.get('skills', '')}
        - Experience: {parsed_jd.get('experience', '')}

        Candidates Data:
        {comparison_context}

        Write a concise comparison detailing:
        - Who is the strongest candidate and why.
        - Key trade-offs between them (e.g. one has more experience, the other has better modern skills).
        - Final recommendation for interview sequencing.
        
        Return your response in clean Markdown.
        """

        try:
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating comparison insights: {e}"

    def run_chatbot(self, parsed_resume, parsed_jd, metrics, chat_history, user_question):
        """
        Interact with the user regarding the resume evaluation and JD,
        utilizing structured candidate metrics and local FAISS vector search context.
        """
        if not self.has_api_key():
            return "I'm sorry, I cannot answer questions without an API key. Please input your Gemini API Key in the sidebar."

        # Retrieve the most relevant resume & JD context dynamically using FAISS local vector database
        retrieved_context = self._retrieve_context(parsed_resume, parsed_jd, user_question, k=4)

        # Build chat history context string
        history_str = ""
        for msg in chat_history[-6:]:  # Keep last 3 turns
            role = "Candidate" if msg["role"] == "user" else "Hiring Assistant"
            history_str += f"{role}: {msg['content']}\n"

        prompt = CHAT_PROMPT_TEMPLATE.format(
            resume_context=retrieved_context,
            jd_context="Refer to context segments above.",
            skill_match=metrics.get("skills_match", 0),
            experience_match=metrics.get("experience_match", 0),
            education_match=metrics.get("education_match", 0),
            project_match=metrics.get("projects_match", 0),
            certification_match=metrics.get("certifications_match", 0),
            overall_match=metrics.get("overall_score", 0),
            chat_history=history_str,
            user_question=user_question
        )

        try:
            # Generation via LangChain wrapper
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error calling Gemini: {e}"
