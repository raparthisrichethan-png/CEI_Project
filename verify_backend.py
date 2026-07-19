import os
import sys
import numpy as np
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_tests():
    print("==================================================")
    print("AI Hiring Assistant: Simplified Verification Tests")
    print("==================================================")

    # Add project root to python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # Test 1: Import all modules
    print("\nTest 1: Importing modules...")
    try:
        from backend.config import DEFAULT_WEIGHTS
        from backend.parser import segment_resume, extract_contact_info
        from backend.embedding_engine import EmbeddingEngine
        from backend.similarity_engine import SimilarityEngine, cosine_similarity
        from backend.rag_pipeline import RAGPipeline
        print("[SUCCESS] All backend modules imported successfully.")
    except Exception as e:
        print(f"[FAILED] Module imports failed: {e}")
        return

    # Test 2: Local Parsing Check
    print("\nTest 2: Rule-based section parsing check...")
    sample_resume = """
    John Doe
    Email: john.doe@example.com | Phone: +1 123 456 7890 | github.com/jdoe
    Category: Software Engineering
    
    Technical Skills
    Python, Java, Go, Kubernetes, Terraform, Docker, AWS
    
    Work Experience
    Senior Software Engineer at ABC Tech (2021-Present)
    - Developed microservices in Python and Go, deployed to Kubernetes.
    - Improved API latency by 45% using Redis caching.
    
    Education
    BS in Computer Science, Stanford University (2017-2021)
    
    Projects
    Kubernetes Scaling Controller: Built custom controller in Go.
    
    Certifications
    AWS Certified Solutions Architect Associate
    """
    
    parsed = segment_resume(sample_resume)
    contact = extract_contact_info(sample_resume)
    
    if parsed["skills"] and parsed["experience"] and parsed["education"]:
        print(f"[SUCCESS] Parsed sections: Skills length={len(parsed['skills'])}, Experience length={len(parsed['experience'])}")
    else:
        print("[WARNING] Resume parsing returned empty sections.")
        
    if contact["name"] == "John Doe" and contact["email"] == "john.doe@example.com":
        print(f"[SUCCESS] Parsed contact details: {contact}")
    else:
        print(f"[WARNING] Contact details parsing failed: {contact}")

    # Test 3: Embedding Engine Check
    print("\nTest 3: Local Sentence Transformers Check...")
    try:
        engine = EmbeddingEngine()
        
        t0 = time.time()
        v1 = engine.get_embedding("Software Engineer Python AWS")
        t1 = time.time()
        duration = t1 - t0
        
        print(f"Embedding generated in {duration:.4f}s")
        assert len(v1) == 384, "Embedding size should be 384"
        print(f"[SUCCESS] Local embedding generated successfully.")
    except Exception as e:
        print(f"[FAILED] Embedding engine failed: {e}")

    # Test 4: Direct Cosine Similarity
    print("\nTest 4: Direct Cosine Similarity check...")
    try:
        engine = EmbeddingEngine()
        v1 = engine.get_embedding("Python programming language")
        v2 = engine.get_embedding("Python development coding")
        v3 = engine.get_embedding("Financial market trading analysis")
        
        sim_match = cosine_similarity(v1, v2)
        sim_diff = cosine_similarity(v1, v3)
        
        print(f"Similarity (matching concepts): {sim_match:.4f}")
        print(f"Similarity (unrelated concepts): {sim_diff:.4f}")
        assert sim_match > sim_diff, "Related concepts should have higher similarity"
        print("[SUCCESS] Direct Cosine Similarity calculates and discriminates concepts correctly.")
    except Exception as e:
        print(f"[FAILED] Cosine similarity test failed: {e}")

    # Test 5: Similarity Engine Check
    print("\nTest 5: Semantic Similarity calculation check...")
    try:
        sim_engine = SimilarityEngine()
        parsed_jd = {
            "skills": "Python, AWS, Kubernetes, Docker, PostgreSQL",
            "experience": "5+ years developing cloud applications, scaling systems.",
            "education": "BS in Computer Science or related engineering field",
            "projects": "Distributed systems, scale controller, database design",
            "certifications": "AWS Solutions Architect Professional"
        }
        
        scores = sim_engine.calculate_scores(parsed, parsed_jd)
        print(f"Sub-scores calculated: {scores}")
        print(f"[SUCCESS] Overall Match Score: {scores['overall_score']:.1f}%")
    except Exception as e:
        print(f"[FAILED] Similarity engine failed: {e}")

    print("\n==================================================")
    print("Verification Completed.")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
