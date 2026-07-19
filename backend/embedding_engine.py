import os
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.config import EMBEDDING_MODEL_NAME

class EmbeddingEngine:
    """
    A simple embedding engine that uses the LangChain HuggingFaceEmbeddings wrapper
    to convert raw text segments into numerical vector embeddings.
    """
    _model_instance = None
    
    @classmethod
    def get_model(cls):
        """Lazy loads the LangChain HuggingFaceEmbeddings model only when needed to save memory."""
        if cls._model_instance is None:
            print(f"Loading Sentence Transformer model via LangChain: {EMBEDDING_MODEL_NAME}...")
            cls._model_instance = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}
            )
        return cls._model_instance

    def get_embedding(self, text):
        """
        Convert a string of text into a 384-dimensional vector embedding.
        If the text is empty, returns a zero vector.
        """
        if not text or not text.strip():
            return np.zeros(384, dtype=np.float32)

        # Get the model and embed the query
        model = self.get_model()
        embedding = model.embed_query(text)
        return np.array(embedding, dtype=np.float32)

    def get_embeddings_batch(self, texts):
        """
        Convert a list of text strings into a list of vector embeddings.
        This is faster than encoding strings one-by-one.
        """
        if not texts:
            return np.array([], dtype=np.float32)
            
        model = self.get_model()
        embeddings = model.embed_documents(texts)
        return np.array(embeddings, dtype=np.float32)

