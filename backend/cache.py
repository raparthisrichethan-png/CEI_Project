import os
import sqlite3
import hashlib

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
DB_PATH = os.path.join(CACHE_DIR, "llm_response_cache.db")

def get_hash(text):
    """Generate SHA256 hash for a given text query/prompt."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

class LLMResponseCache:
    """
    SQLite-based local cache for LLM responses and their validation scores.
    """
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT faithfulness_score FROM response_cache LIMIT 1")
        except sqlite3.OperationalError:
            # Drop old table format to regenerate with the new schema columns
            cursor.execute("DROP TABLE IF EXISTS response_cache")
            
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                prompt_hash TEXT PRIMARY KEY,
                prompt_text TEXT,
                response_text TEXT,
                faithfulness_score INTEGER,
                faithfulness_reason TEXT,
                relevance_score INTEGER,
                relevance_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def get(self, prompt_text):
        """Retrieve cached response and scores if available."""
        prompt_hash = get_hash(prompt_text)
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT response_text, faithfulness_score, faithfulness_reason, relevance_score, relevance_reason FROM response_cache WHERE prompt_hash = ?",
            (prompt_hash,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "response_text": row[0],
                "faithfulness_score": row[1],
                "faithfulness_reason": row[2],
                "relevance_score": row[3],
                "relevance_reason": row[4]
            }
        return None

    def set(self, prompt_text, response_text, faithfulness_score, faithfulness_reason, relevance_score, relevance_reason):
        """Save a new response and its scores to the cache."""
        prompt_hash = get_hash(prompt_text)
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """INSERT OR REPLACE INTO response_cache 
                   (prompt_hash, prompt_text, response_text, faithfulness_score, faithfulness_reason, relevance_score, relevance_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (prompt_hash, prompt_text, response_text, faithfulness_score, faithfulness_reason, relevance_score, relevance_reason)
            )
            self.conn.commit()
        except Exception as e:
            print(f"Error writing to SQLite cache database: {e}")
