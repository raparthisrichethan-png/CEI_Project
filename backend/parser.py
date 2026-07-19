import re
import os
import json
import fitz  # PyMuPDF
import docx
from langchain_groq import ChatGroq

try:
    from backend.config import DEFAULT_LLM_MODEL
except ImportError:
    DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"

def extract_text_from_pdf(file_path):
    """Extract raw text from a PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extract raw text from a DOCX file using python-docx."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def extract_text(file_path):
    """Wrapper to extract text based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(file_path)
    else:
        # Fallback to reading as text if possible
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

def classify_document_type(text):
    """
    Classifies a document's text as either 'resume' or 'jd'.
    Returns 'resume', 'jd', or 'unknown'.
    """
    if not text:
        return 'unknown'
    text_lower = text.lower()
    
    # Resume indicators
    resume_indicators = [
        "education", "academic background", "gpa", "work experience", 
        "professional summary", "projects", "certifications", "skills",
        "objective", "hobbies", "personal details", "strengths", "languages",
        "cgp", "internship", "employment history", "career summary"
    ]
    
    # Job Description indicators
    jd_indicators = [
        "job description", "responsibilities", "requirements", "qualifications",
        "we are looking for", "about the role", "key responsibilities", 
        "job summary", "duties", "reporting to", "apply now", "equal opportunity",
        "ideal candidate", "role description", "position overview", "what you will do",
        "about us", "who you are", "what we offer", "benefits", "compensation",
        "successful candidate", "join our team", "target role", "jd for"
    ]
    
    resume_score = sum(1 for word in resume_indicators if word in text_lower)
    jd_score = sum(1 for word in jd_indicators if word in text_lower)
    
    # Check for email/phone patterns (strong indicators of a candidate resume)
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    
    # Exclude typical company emails like jobs@, hr@, careers@ for resume scoring
    emails = re.findall(email_pattern, text)
    has_personal_email = False
    for email in emails:
        email_lower = email.lower()
        if not any(x in email_lower for x in ["hr@", "careers@", "jobs@", "recruiting@", "apply@", "info@", "contact@"]):
            has_personal_email = True
            break
            
    if has_personal_email:
        resume_score += 3
    if re.search(phone_pattern, text):
        resume_score += 2
        
    if resume_score > jd_score:
        return 'resume'
    elif jd_score > resume_score:
        return 'jd'
    else:
        return 'unknown'

def clean_text(text):
    """Basic cleaning of text to remove redundant spacing and control characters."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_contact_info(text, api_key=None):
    """Extract Name, Email, Phone, and LinkedIn/GitHub links from text."""
    api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            llm = ChatGroq(
                model=DEFAULT_LLM_MODEL,
                groq_api_key=api_key,
                temperature=0.0
            )
            prompt = f"""
            You are an expert contact information extractor. Analyze the following resume text and extract the candidate's contact details:
            - name (the candidate's name)
            - email (email address)
            - phone (phone number)
            - linkedin (LinkedIn profile URL)
            - github (GitHub profile URL)

            Format the output STRICTLY as a JSON object with keys: "name", "email", "phone", "linkedin", "github".
            If any field is not found, return an empty string for that field (or "Unknown Candidate" for the name if name cannot be found).
            Do not output any markdown formatting like ```json or trailing text. Output raw JSON string only.

            Resume Text:
            {text}
            """
            response = llm.invoke(prompt)
            content = response.content.strip()
            # Clean markdown code blocks if returned
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            info = {
                "name": str(data.get("name", "Unknown Candidate") or "Unknown Candidate"),
                "email": str(data.get("email", "") or ""),
                "phone": str(data.get("phone", "") or ""),
                "linkedin": str(data.get("linkedin", "") or ""),
                "github": str(data.get("github", "") or "")
            }
            return info
        except Exception as e:
            print(f"Error extracting contact info via LangChain: {e}. Falling back to regex.")

    info = {
        "name": "Unknown Candidate",
        "email": "",
        "phone": "",
        "linkedin": "",
        "github": ""
    }
    
    if not text:
        return info

    # Try to extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match:
        info["email"] = email_match.group(0)

    # Try to extract phone number (supports common formats)
    phone_match = re.search(r'(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+?\d{1,4}[-.\s]?\d{2,5}[-.\s]?\d{2,5}[-.\s]?\d{2,5})', text)
    if phone_match:
        info["phone"] = phone_match.group(0)

    # Try to extract LinkedIn and GitHub URLs
    linkedin_match = re.search(r'(https?://(www\.)?linkedin\.com/in/[\w\-]+)', text, re.IGNORECASE)
    if linkedin_match:
        info["linkedin"] = linkedin_match.group(1)

    github_match = re.search(r'(https?://(www\.)?github\.com/[\w\-]+)', text, re.IGNORECASE)
    if github_match:
        info["github"] = github_match.group(1)

    # Try to guess name
    # Usually the name is at the top. Let's look at the first few lines of the text.
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        # Avoid common section headers or emails/phones as the name
        for line in lines[:5]:
            if (len(line.split()) >= 2 and 
                len(line.split()) <= 4 and 
                "resume" not in line.lower() and 
                "curriculum" not in line.lower() and
                "page" not in line.lower() and
                "contact" not in line.lower() and
                "email" not in line.lower() and
                "phone" not in line.lower() and
                "@" not in line and
                not re.search(r'\d', line)):
                info["name"] = line
                break
                
    return info

def segment_resume(text, api_key=None):
    """
    Segment the resume text into logical parts using semantic chunking & relevance mapping.
    Splits the raw text into chunks and assigns chunks to standard categories:
    Skills, Experience, Education, Projects, Certifications.
    """
    sections = {
        "contact": "",
        "skills": "",
        "experience": "",
        "education": "",
        "projects": "",
        "certifications": "",
        "other": ""
    }
    
    if not text:
        return sections

    # 1. Chunking raw text into overlapping windows
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_text(text)
    
    if not chunks:
        chunks = [text]

    # 2. Define semantic search anchors
    section_queries = {
        "skills": "technical skills, programming languages, software, frameworks, databases, developer tools, hard skills, soft skills",
        "experience": "work experience, employment history, career background, job description, roles, achievements, professional experience",
        "education": "academic degrees, university education, college, schools, GPA, graduation, scholastic accomplishments",
        "projects": "technical projects, github repositories, project descriptions, key contributions, software development projects",
        "certifications": "professional certifications, industry courses, credentials, awards, honors, achievements"
    }

    try:
        from backend.embedding_engine import EmbeddingEngine
        from backend.similarity_engine import cosine_similarity
        
        # Load local embedding engine
        engine = EmbeddingEngine()
        chunk_embeddings = [engine.get_embedding(chunk) for chunk in chunks]
        
        # Match chunks to sections
        for sec, query in section_queries.items():
            query_embedding = engine.get_embedding(query)
            similarities = []
            
            for idx, c_emb in enumerate(chunk_embeddings):
                sim = cosine_similarity(c_emb, query_embedding)
                similarities.append((sim, idx))
                
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Select top-k chunks above threshold
            selected_indices = [idx for sim, idx in similarities[:3] if sim >= 0.22]
            
            # Fallback to single best chunk if none met the threshold
            if not selected_indices and similarities:
                selected_indices = [similarities[0][1]]
                
            selected_indices.sort()
            
            top_chunks = [chunks[idx] for idx in selected_indices]
            sections[sec] = "\n\n".join(top_chunks)
            
    except Exception as e:
        print(f"Error mapping chunks to sections: {e}. Falling back to default list partitioning.")
        n = len(chunks)
        sections["skills"] = "\n\n".join(chunks[:max(1, int(n*0.3))])
        sections["experience"] = "\n\n".join(chunks[max(1, int(n*0.3)):max(2, int(n*0.7))])
        sections["education"] = "\n\n".join(chunks[max(2, int(n*0.7)):])

    # Contact details heuristic (usually in the first 2 chunks)
    first_few = "\n".join(chunks[:min(len(chunks), 2)])
    sections["contact"] = first_few

    # Clean text in each section
    for k in sections:
        sections[k] = clean_text(sections[k])
        
    return sections
