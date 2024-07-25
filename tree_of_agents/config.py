import os
from dotenv import load_dotenv

load_dotenv()

TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
MAX_TOKENS_PER_AGENT = 4096
NMN_MODEL = "meta-llama/Llama-2-70b-chat"
MEMORY_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LEAF_AGENT_DATA_DIR = './leaf_agent_data'