import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MAX_TOKENS_PER_AGENT = 4096
NMN_MODEL = "gpt-4o-mini"
MEMORY_MODEL = "gpt-4o-mini"
ROOT_MODEL = "gpt-4o-mini"
LEAF_AGENT_DATA_DIR = './leaf_agent_data'
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"