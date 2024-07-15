import json
# from groq import Groq # type: ignore #! groq version
from together import Together  # Import the Together SDK
from config import config
import asyncio

from src.utils.structured_logging import get_logger
from src.utils.enhanced_logging import log_execution_time
from src.utils.error_handling import log_error_with_traceback

class Agent:
    # def __init__(self, memory_manager, groq_api_key: str): #! groq version
    def __init__(self, memory_manager, together_api_key: str):
        self.logger = get_logger('chat')
        self.memory_manager = memory_manager
        # self.groq_client = Groq(api_key=groq_api_key) #! groq version
        self.together_client = Together(api_key=together_api_key)
        
    @log_execution_time
    async def process_query(self, query: str) -> str:
        try:
            self.logger.info("Processing query", extra={"query": query})
            relevant_memories = await self.memory_manager.retrieve_relevant_memories(query)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a highly efficient, no-nonsense administrator at the core of a sophisticated memory system. Your primary role is to analyze retrieved memories and provide concise, relevant summaries or indicate their lack of relevance. You value brevity and directness above all else.

        The retrieved memories you will be processing are structured as follows:

        {
        "Relevant Injected Content": {
            "UID": {
            "Time": "timestamp",
            "Query": "previous query",
            "Response": "previous response",
            "Similarity": float value,
            "linked_UIDs": ["list of linked memory UIDs"]
            },
            ...
        }
        }

        Analyze these memories in relation to the current query efficiently, paying special attention to linked memories."""
                },
                {
                    "role": "user",
                    "content": f"""QUERY: {query}

        RETRIEVED MEMORIES:
        {relevant_memories}

        Your task:

        1. Assess if the retrieved memories contain ANY relevant information to the query.
        2. If NO relevant information exists, respond ONLY with: "NO RELEVANT INFORMATION"
        3. If relevant information exists, assess, making an internal summary of ONLY the most pertinent points.
        4. Pay special attention to linked memories and their relevance to the current query.
        5. After summarizing, provide a concise response to the original query based on the relevant information.

        Imagine your internal summary in the following format:
        SUMMARY:
        • Point 1
        • Point 2
        ...

        RESPONSE BASED ON INTERNAL SUMMARY:
        [Your concise response to the original query]
    Note - only respond with the terse response appropriate to the query, provide none of the summary information that is for your eyes only."""
                }
            ]

            # Log the full request
            self.logger.info("LLM Request", extra={"messages": json.dumps(messages, indent=2)})

            # Use asyncio.to_thread to run the synchronous API call in a separate thread
            response = await asyncio.to_thread(
                self.together_client.chat.completions.create,  # Use Together API method
                messages=messages,
                model=config.MODEL_NAME,
            )

            # Log the full response
            self.logger.info("LLM Response", extra={"response": json.dumps(response.model_dump(), indent=2)})

            self.logger.info("Query processed", extra={"query": query})
            return response.choices[0].message.content
        except Exception as e:
            log_error_with_traceback(self.logger, "Error processing query", e)
            raise
