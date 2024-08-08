from goodai.ltm.mem.config import TextMemoryConfig


class Config:
    DATABASE_URL = "sqlite:///ltm_agent.db"
    SEMANTIC_MEMORY_CONFIG = TextMemoryConfig(
        chunk_capacity=50,
        chunk_overlap_fraction=0.0,
    )