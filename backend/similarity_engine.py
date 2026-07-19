import numpy as np
from backend.embedding_engine import EmbeddingEngine
from backend.config import DEFAULT_WEIGHTS

def cosine_similarity(v1, v2):
    """Calculate the cosine similarity between two vectors."""
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_prod / (norm_v1 * norm_v2))

class SimilarityEngine:
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()

    def calculate_scores(self, parsed_resume, parsed_jd, weights=None):
        """
        Calculate semantic similarity scores for each resume section against JD requirements.
        
        parsed_resume: dict with keys: 'skills', 'experience', 'education', 'projects', 'certifications'
        parsed_jd: dict with keys: 'skills', 'experience', 'education', 'projects', 'certifications'
        weights: dict with relative weights of each score (optional)
        
        Returns:
            dict containing:
                - skill_match
                - experience_match
                - education_match
                - project_match
                - certification_match
                - overall_score
        """
        if weights is None:
            weights = DEFAULT_WEIGHTS

        scores = {}
        
        # We compare section to section using local embeddings
        sections = ['skills', 'experience', 'education', 'projects', 'certifications']
        
        for sec in sections:
            res_text = parsed_resume.get(sec, "").strip()
            jd_text = parsed_jd.get(sec, "").strip()
            
            # Cases:
            if not jd_text:
                # If the JD doesn't specify requirements for this section, they get full match
                scores[f"{sec}_match"] = 100.0
            elif not res_text:
                # If resume lacks this section but JD requires it, they get 0 match
                scores[f"{sec}_match"] = 0.0
            else:
                # Calculate local semantic similarity
                res_vector = self.embedding_engine.get_embedding(res_text)
                jd_vector = self.embedding_engine.get_embedding(jd_text)
                
                sim = cosine_similarity(res_vector, jd_vector)
                
                # Normalize similarity from [-1, 1] to [0, 1], and map to [0, 100]
                # In practice, Sentence Transformers similarities are rarely negative
                match_percentage = max(0.0, sim) * 100.0
                
                # Scale up scores slightly to make it feel standard (often similarity is ~0.3-0.8)
                # Let's map it: a similarity of 0.3 is a basic match (e.g. 50%), 0.7+ is a strong match (e.g. 90%+)
                if match_percentage > 0:
                    scaled = 30 + (match_percentage / 100.0) * 70
                    scores[f"{sec}_match"] = min(100.0, max(0.0, scaled))
                else:
                    scores[f"{sec}_match"] = 0.0

        # Calculate Overall Score using weights
        weighted_sum = 0.0
        total_weight = 0.0
        
        for sec in sections:
            # Only weight the sections that are actually evaluated (i.e. JD specified them)
            # Or if JD didn't specify them, we can either keep them at 100% or exclude them.
            # Excluding them or keeping them at 100% is fine. Let's include them in the weight.
            weight = weights.get(sec, 0.0)
            weighted_sum += scores[f"{sec}_match"] * weight
            total_weight += weight

        if total_weight > 0:
            scores["overall_score"] = float(weighted_sum / total_weight)
        else:
            scores["overall_score"] = 0.0

        return scores
