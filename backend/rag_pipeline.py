import os
import json
from langchain_groq import ChatGroq
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
    LangChain framework to call models via Groq API.
    """
    
    def __init__(self, api_key=None):
        # Read the API key from argument or environment variables
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        
        # Initialize LangChain ChatGroq Client if API key is provided
        if self.api_key:
            self.client = ChatGroq(
                model=DEFAULT_LLM_MODEL,
                groq_api_key=self.api_key,
                temperature=0.0
            )
            
        from backend.evaluator import LLMEvaluator
        from backend.cache import LLMResponseCache
        self.evaluator = LLMEvaluator(api_key=self.api_key)
        self.cache = LLMResponseCache()

    def has_api_key(self):
        """Check if an API Key is configured."""
        return self.api_key is not None and len(self.api_key.strip()) > 0

    def set_api_key(self, api_key):
        """Update API key dynamically at runtime."""
        self.api_key = api_key
        from backend.evaluator import LLMEvaluator
        if api_key:
            self.client = ChatGroq(
                model=DEFAULT_LLM_MODEL,
                groq_api_key=api_key,
                temperature=0.0
            )
            self.evaluator = LLMEvaluator(api_key=api_key)
        else:
            self.client = None
            self.evaluator = LLMEvaluator(api_key=None)

    def _generate_with_eval(self, prompt, context, query, cache_key_prefix=""):
        """Generate LLM response, run real-time evaluation metrics, and cache metrics/responses."""
        # 1. Check cache
        cached = self.cache.get(prompt)
        if cached:
            try:
                import streamlit as st
                state_key = f"{cache_key_prefix}_eval"
                st.session_state[state_key] = {
                    "faithfulness_score": cached["faithfulness_score"],
                    "faithfulness_reason": cached["faithfulness_reason"],
                    "relevance_score": cached["relevance_score"],
                    "relevance_reason": cached["relevance_reason"]
                }
            except (ImportError, AttributeError):
                pass
            return cached["response_text"]

        # 2. Generate response if cache miss
        response = self.client.invoke(prompt)
        response_text = response.content

        # 3. Evaluate response quality (LLM-as-a-Judge)
        eval_results = self.evaluator.evaluate_response(context, query, response_text)

        # 4. Save to cache
        self.cache.set(
            prompt_text=prompt,
            response_text=response_text,
            faithfulness_score=eval_results["faithfulness_score"],
            faithfulness_reason=eval_results["faithfulness_reason"],
            relevance_score=eval_results["relevance_score"],
            relevance_reason=eval_results["relevance_reason"]
        )

        # 5. Store results in Streamlit session state for UI visibility
        try:
            import streamlit as st
            state_key = f"{cache_key_prefix}_eval"
            st.session_state[state_key] = eval_results
        except (ImportError, AttributeError):
            pass

        return response_text

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

    def _heuristic_extract_jd(self, jd_text):
        """Heuristic semantic chunking fallback parser for Job Descriptions."""
        sections = {
            "skills": "",
            "experience": "",
            "education": "",
            "projects": "",
            "certifications": ""
        }
        if not jd_text:
            return sections
            
        # 1. Chunking
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        chunks = splitter.split_text(jd_text)
        
        if not chunks:
            chunks = [jd_text]
            
        # 2. Semantic targets
        section_queries = {
            "skills": "required technical skills, technologies, qualifications, programming languages, databases, developer tools, hard skills",
            "experience": "required experience, background, years of experience, professional duties, key responsibilities, role description",
            "education": "required education, degree, university, academic requirements, certificates, scholastic background",
            "projects": "projects, tasks, job responsibilities, key functions, software development activities",
            "certifications": "required certifications, courses, credentials, licenses, awards"
        }
        
        try:
            from backend.embedding_engine import EmbeddingEngine
            from backend.similarity_engine import cosine_similarity
            
            engine = EmbeddingEngine()
            chunk_embeddings = [engine.get_embedding(chunk) for chunk in chunks]
            
            for sec, query in section_queries.items():
                query_embedding = engine.get_embedding(query)
                similarities = []
                
                for idx, c_emb in enumerate(chunk_embeddings):
                    sim = cosine_similarity(c_emb, query_embedding)
                    similarities.append((sim, idx))
                    
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                # Select top-k relevant chunks
                selected_indices = [idx for sim, idx in similarities[:3] if sim >= 0.22]
                
                if not selected_indices and similarities:
                    selected_indices = [similarities[0][1]]
                    
                selected_indices.sort()
                sections[sec] = "\n\n".join([chunks[idx] for idx in selected_indices])
                
        except Exception as e:
            print(f"Error mapping JD chunks: {e}")
            n = len(chunks)
            sections["skills"] = "\n\n".join(chunks[:max(1, int(n*0.5))])
            sections["experience"] = "\n\n".join(chunks[max(1, int(n*0.5)):])
            
        return sections

    def extract_jd_requirements(self, jd_text):
        """
        Parse raw Job Description text into structured requirements.
        If no API Key is available, uses the heuristic fallback parser.
        """
        if not self.has_api_key():
            return self._heuristic_extract_jd(jd_text)

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
            return self._heuristic_extract_jd(jd_text)

    def _handle_llm_exception(self, e):
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower() or "resource_exhausted" in err_str.lower() or "resourceexhausted" in err_str.lower():
            return "### ⚠️ Quota Exceeded (429 Error)\nThe configured Groq API key has exceeded its rate limit or daily quota. Please clear the pre-populated key in the sidebar and input your own Groq API key to run this feature."
        return f"Error calling Groq: {e}"

    def generate_analysis(self, parsed_resume, parsed_jd):
        """Generate structured screening feedback (strengths, gaps, ATS, suggestions)."""
        if not self.has_api_key():
            return "### API Key Missing\nPlease provide a Groq API Key in the sidebar to generate AI feedback."

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
            return self._generate_with_eval(
                prompt=prompt,
                context=f"Resume Context:\n{resume_context}\n\nJD Context:\n{jd_context}",
                query="Generate structured screening feedback including strengths, gaps, and suggestions.",
                cache_key_prefix="analysis"
            )
        except Exception as e:
            return self._handle_llm_exception(e)

    def generate_roadmap(self, parsed_resume, parsed_jd):
        """Generate learning roadmap, certification, and project recommendations."""
        if not self.has_api_key():
            return "### API Key Missing\nPlease provide a Groq API Key in the sidebar to generate your learning roadmap."

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
            return self._generate_with_eval(
                prompt=prompt,
                context=f"Resume Context:\n{resume_context}\n\nJD Context:\n{jd_context}",
                query="Generate customized learning roadmap, project recommendations, and industry certifications to bridge candidate gaps.",
                cache_key_prefix="roadmap"
            )
        except Exception as e:
            return self._handle_llm_exception(e)

    def generate_comparison(self, parsed_jd, selected_candidates):
        """Generate side-by-side executive candidate comparison using LangChain LLM."""
        if not self.has_api_key():
            return "### API Key Missing\nPlease provide a Groq API Key in the sidebar to generate comparative analysis."

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
            return self._generate_with_eval(
                prompt=prompt,
                context=comparison_context,
                query="Generate side-by-side executive candidate comparison detailing strengths and recommended interview sequencing.",
                cache_key_prefix="comparison"
            )
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "resource_exhausted" in err_str.lower() or "resourceexhausted" in err_str.lower():
                return "### ⚠️ Quota Exceeded (429 Error)\nThe configured Groq API key has exceeded its rate limit or daily quota. Please clear the pre-populated key in the sidebar and input your own Groq API key to run this feature."
            return f"Error generating comparison insights: {e}"

    def _get_offline_response(self, user_question, parsed_resume, parsed_jd, metrics):
        """Helper to return high-quality offline rule-based chatbot responses based on extracted metrics."""
        q = user_question.lower()
        
        # 1. Ask about skills / gap
        if any(x in q for x in ["skill", "gap", "technolog", "language", "framework"]):
            res_skills = parsed_resume.get("skills", "")
            jd_skills = parsed_jd.get("skills", "")
            return f"""
**Offline Assistant:** I've analyzed your skills match (Current Score: **{metrics.get('skills_match', 0):.1f}%**).

*   **Your Extracted Skills:** {res_skills if res_skills else 'None found'}
*   **Job Requirements:** {jd_skills if jd_skills else 'None found'}

**Suggestions to improve:**
1. Focus on aligning the wording of your skills exactly with the Job Description keywords.
2. Consider adding relevant side projects to demonstrate practical applications of these tools.
"""

        # 2. Ask about experience / job history
        elif any(x in q for x in ["experience", "work", "job", "career", "year"]):
            return f"""
**Offline Assistant:** Your experience match score is **{metrics.get('experience_match', 0):.1f}%**.

**Tips to improve your experience description:**
1. **Use the STAR method:** Describe your work in terms of Situation, Task, Action, and Result (e.g. *"Improved database queries by 40% using indexing"*).
2. **Quantify achievements:** Try to add numbers, percentages, and metrics to demonstrate business impact.
3. **Include target keywords:** Make sure tools mentioned in the Job Description (like AWS, Docker, Python) are naturally woven into your job bullet points.
"""

        # 3. Ask about ATS
        elif any(x in q for x in ["ats", "score", "format", "check", "structure"]):
            return f"""
**Offline Assistant:** Your resume formatting score is evaluated by our local ATS optimizer.

**Core Checkpoints:**
*   Ensure your contact email and phone are clearly visible in the header.
*   Use standard headings like **Skills**, **Experience**, **Education**, and **Projects** to help automated parsers index your resume.
*   Avoid multi-column tables or complex graphics which can scramble text during extraction.
"""

        # 4. Ask about projects / education / certifications
        elif any(x in q for x in ["project", "education", "degree", "certificat", "course"]):
            return f"""
**Offline Assistant:**
*   **Projects Match Score:** {metrics.get('projects_match', 0):.1f}%
*   **Education Match Score:** {metrics.get('education_match', 0):.1f}%
*   **Certifications Match Score:** {metrics.get('certifications_match', 0):.1f}%

**Recommendations:**
1. List projects that directly showcase the required technology stack. Include a link to GitHub repositories if possible.
2. Position your highest and most relevant degree/certifications clearly in the education section.
"""

        # 5. Default generic response
        else:
            return f"""
**Offline Assistant:** I'm running in offline mode due to a missing or exhausted Groq API Key. Here is a summary of your profile match:

*   **Overall Role Match:** **{metrics.get('overall_score', 0):.1f}%**
*   **Skills Match:** **{metrics.get('skills_match', 0):.1f}%**
*   **Experience Match:** **{metrics.get('experience_match', 0):.1f}%**
*   **Education Match:** **{metrics.get('education_match', 0):.1f}%**

*Tip: You can ask me specific questions about your "skills", "experience", "ATS score", or "projects" to get target suggestions offline! Or configure a valid Groq API Key in the sidebar for full open-ended conversations.*
"""

    def run_chatbot(self, parsed_resume, parsed_jd, metrics, chat_history, user_question):
        """
        Interact with the user regarding the resume evaluation and JD,
        utilizing structured candidate metrics and local FAISS vector search context.
        """
        if not self.has_api_key():
            return self._get_offline_response(user_question, parsed_resume, parsed_jd, metrics)

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
            return self._generate_with_eval(
                prompt=prompt,
                context=retrieved_context,
                query=user_question,
                cache_key_prefix="chatbot"
            )
        except Exception as e:
            err_str = str(e)
            offline_resp = self._get_offline_response(user_question, parsed_resume, parsed_jd, metrics)
            if "429" in err_str or "quota" in err_str.lower() or "resource_exhausted" in err_str.lower() or "resourceexhausted" in err_str.lower():
                return f"### ⚠️ Quota Exceeded (429 Error)\nThe configured Groq API key has exceeded its rate limit or daily quota. Clearing the key or using offline mode will allow offline responses.\n\n---\n\n{offline_resp}"
            return f"### ⚠️ Chatbot Offline Mode Active (API Error)\n*Could not reach Groq API ({err_str}). Falling back to local offline assistant.*\n\n---\n\n{offline_resp}"
