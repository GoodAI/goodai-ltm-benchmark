# app/services/embedding_service.py
from openai import AsyncOpenAI
import numpy as np
from app.config import config
from app.utils.logging import get_logger
import os

logger = get_logger('custom')

class EmbeddingService:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)

    async def get_embedding(self, text: str) -> np.ndarray:
        try:
            logger.debug("Generating embedding using OpenAI")
            response = await self.openai_client.embeddings.create(
                input=[text],
                model=config.EMBEDDING['model']
            )
            embedding = response.data[0].embedding
            logger.debug("Embedding generated successfully")
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            raise

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))