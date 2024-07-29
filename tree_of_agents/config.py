import os
from dotenv import load_dotenv

load_dotenv()

TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
MAX_TOKENS_PER_AGENT = 4096
NMN_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
MEMORY_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
ROOT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
LEAF_AGENT_DATA_DIR = './leaf_agent_data'
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"