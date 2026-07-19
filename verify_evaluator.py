import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from backend.evaluator import LLMEvaluator
from backend.cache import LLMResponseCache

def test_evaluator():
    print("Testing LLM Evaluator...")
    evaluator = LLMEvaluator()
    context = "Candidate has 5 years of Python experience and a BS in CS. Job Description asks for a Python coder."
    query = "Evaluate candidate's qualifications."
    response = "The candidate has 5 years of experience in Python and holds a BS degree in Computer Science, matching the requirements."
    
    metrics = evaluator.evaluate_response(context, query, response)
    print(f"Computed metrics: {metrics}")
    assert "faithfulness_score" in metrics
    assert "relevance_score" in metrics
    print("Evaluator test PASSED.")

def test_cache():
    print("Testing response cache...")
    cache = LLMResponseCache()
    prompt = "Test prompt content"
    cache.set(
        prompt_text=prompt,
        response_text="Test response",
        faithfulness_score=95,
        faithfulness_reason="Reason F",
        relevance_score=98,
        relevance_reason="Reason R"
    )
    
    cached = cache.get(prompt)
    print(f"Cached values: {cached}")
    assert cached is not None
    assert cached["response_text"] == "Test response"
    assert cached["faithfulness_score"] == 95
    assert cached["relevance_score"] == 98
    print("Cache test PASSED.")

if __name__ == "__main__":
    test_cache()
    # Only test LLM call if API Key is configured
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            test_evaluator()
        except Exception as e:
            print(f"LLM evaluator call failed (API config/network?): {e}")
    else:
        print("Skipping LLM evaluator test as GROQ_API_KEY is not configured in environment.")
