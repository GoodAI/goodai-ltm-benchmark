import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from typing import List, Tuple
from config import config
import json
from src.utils.api_utils import rate_limited, exponential_backoff, cached

logger = logging.getLogger("master")
chat_logger = logging.getLogger("chat")


class Agent:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.chat_model = ChatOpenAI(model_name=config.MODEL_NAME)

    async def process_query(self, query: str) -> str:
        relevant_memories = await self.memory_manager.retrieve_relevant_memories(query)
        messages = [
            HumanMessage(
                content=f"""REQUEST = "{query}"

CONTEXT = "{relevant_memories}"

INSTRUCTIONS = You are a memory summarization agent. Your task is to analyze the provided CONTEXT, which contains relevant memories, and create a concise summary that will help in responding to the REQUEST. Follow these guidelines:

1. Memory format: The CONTEXT contains memories in the following format:
   Similar by [Similarity Metric] (ordered by timestamp - ascending):
   <memory_id>, <previous_query>, <previous_response>, <timestamp>, <similarity_score>
   
   For previously mentioned memories:
   <memory_id>, <similarity_score>

2. Analyze relevance: Consider the similarity metrics and scores to determine each memory's relevance to the current REQUEST.

3. Summarize key information: Focus on details that directly relate to or could inform a response to the REQUEST.

4. Chronological perspective: If relevant, note how information or responses have evolved over time.

5. Highlight conflicts: If memories contain conflicting information, briefly mention these discrepancies.

6. Omit redundancy: If multiple memories contain the same information, mention it only once, noting its recurrence.

Provide your summary in the following format:

SUMMARY:
1. Most relevant information: [Concise bullet points of the most pertinent details]
2. Chronological developments: [If applicable, brief timeline of how information or responses have changed]
3. Conflicting data: [If present, short description of any contradictions in the memories]
4. Recurring themes: [Common elements or responses that appear multiple times]
5. Potential gaps: [Mention any apparent missing information that might be useful for addressing the REQUEST]

RELEVANCE SCORE: [Provide a score from 1-10 indicating how relevant and useful the summarized memories are to the REQUEST, with 10 being highly relevant and 1 being minimally relevant]

This summary aims to provide a comprehensive yet concise overview of the relevant memories to assist in formulating an optimal response to the REQUEST."""
            )
        ]

        # Log the full message sent to the API
        chat_logger.info(
            f"Full ChatGPT API request: {json.dumps([m.dict() for m in messages], indent=2)}"
        )

        response = await self.chat_model.ainvoke(messages)

        # Log the full API response
        chat_logger.info(
            f"Full ChatGPT API response: {json.dumps(response.dict(), indent=2)}"
        )

        logger.debug(f"Processed query: {query} with context: {relevant_memories}")

        return response.content
