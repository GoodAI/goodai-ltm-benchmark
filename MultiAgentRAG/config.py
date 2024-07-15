import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        
        # General settings
        # self.MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192") #! groq
        self.MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-70b-chat-hf") #? Together AI
        
        # Database settings
        self.personal_db_path = self._get_personal_db_path()
        
        # API Keys
        self.OPENAI_API_KEY = os.getenv("GOODAI_OPENAI_API_KEY_LTM01")
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
        
        # Other settings
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
        
        # Processing agent settings
        self.PROCESSING_AGENT_MEMORIES_INCLUDED = int(os.getenv("PROCESSING_AGENT_MEMORIES_INCLUDED", "5"))
        self.MEMORY_RETRIEVAL_THRESHOLD = float(os.getenv("MEMORY_RETRIEVAL_THRESHOLD", "0.75"))
        self.MEMORY_RETRIEVAL_LIMIT = int(os.getenv("MEMORY_RETRIEVAL_LIMIT", "10"))

    def _get_memory_db_path(self):
        return "/app/memory.db" if os.path.exists("/.dockerenv") else "memory.db"
    
    def _get_personal_db_path(self):
        container_id = os.environ.get('HOSTNAME', 'local')
        return f"/app/data/{container_id}.db" if os.path.exists("/.dockerenv") else f"{container_id}.db"

    def validate_api_keys(self):
        return bool(self.GROQ_API_KEY and self.TAVILY_API_KEY)

    L2_NORM_THRESHOLD = float(os.getenv("L2_NORM_THRESHOLD", "0.75"))
    COSINE_SIMILARITY_THRESHOLD = float(os.getenv("COSINE_SIMILARITY_THRESHOLD", "0.75"))
    BM25_THRESHOLD = float(os.getenv("BM25_THRESHOLD", "0.5"))
    JACCARD_SIMILARITY_THRESHOLD = float(os.getenv("JACCARD_SIMILARITY_THRESHOLD", "0.3"))

config = Config()