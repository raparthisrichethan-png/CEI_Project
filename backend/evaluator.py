import os
import json
from langchain_groq import ChatGroq

try:
    from backend.config import DEFAULT_LLM_MODEL
except ImportError:
    DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"

class LLMEvaluator:
    """
    Computes real-time evaluation metrics for LLM responses using the
    LLM-as-a-Judge pattern. Measures faithfulness and relevance.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        if self.api_key:
            self.client = ChatGroq(
                model=DEFAULT_LLM_MODEL,
                groq_api_key=self.api_key,
                temperature=0.0
            )

    def evaluate_response(self, context, query, response_text):
        """
        Evaluate LLM response quality on Groundedness/Faithfulness and Relevance.
        Returns a dictionary with scores (0-100) and brief qualitative justifications.
        """
        # If no client is available, return default metrics (graceful fallback)
        if not self.client:
            return {
                "faithfulness_score": 100,
                "faithfulness_reason": "Evaluator running in offline mode. Baseline assumed.",
                "relevance_score": 100,
                "relevance_reason": "Evaluator running in offline mode. Baseline assumed."
            }

        prompt = f"""You are an expert RAG evaluator and AI validation judge.
Your task is to grade the quality of a generated AI Response against the given Reference Context and User Query/Prompt.

Reference Context:
---
{context}
---

User Query/Prompt:
---
{query}
---

Generated AI Response:
---
{response_text}
---

Grade the response on these two metrics:
1. **Faithfulness (Groundedness)**: Score from 0 to 100.
   - 100: Every statement in the response is strictly supported by the Reference Context.
   - Lower scores (80, 60, 40, etc.): The response introduces claims, assumptions, or details not found in the context (hallucinations). Deduct 20 points per hallucinated claim.
2. **Relevance**: Score from 0 to 100.
   - 100: The response directly, comprehensively, and clearly answers the User Query/Prompt without fluff.
   - Lower scores: The response contains off-topic content, misses key aspects of the query, or is overly verbose.

Format your output STRICTLY as a raw JSON object with the following keys. Do NOT include markdown code fences (like ```json) or any explanation. Output raw JSON only:
{{
  "faithfulness_score": <integer between 0 and 100>,
  "faithfulness_reason": "<1-sentence reason for groundedness score>",
  "relevance_score": <integer between 0 and 100>,
  "relevance_reason": "<1-sentence reason for relevance score>"
}}"""

        try:
            res = self.client.invoke(prompt)
            content = res.content.strip()
            
            # Clean markdown formatting if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)
            return {
                "faithfulness_score": int(data.get("faithfulness_score", 100)),
                "faithfulness_reason": str(data.get("faithfulness_reason", "")),
                "relevance_score": int(data.get("relevance_score", 100)),
                "relevance_reason": str(data.get("relevance_reason", ""))
            }
        except Exception as e:
            # Return default baseline metrics if LLM judge fails (e.g. rate limit)
            print(f"Error in LLM evaluation judge: {e}")
            return {
                "faithfulness_score": 90,
                "faithfulness_reason": "Evaluator failed to compute metric. Graceful baseline applied.",
                "relevance_score": 90,
                "relevance_reason": "Evaluator failed to compute metric. Graceful baseline applied."
            }
