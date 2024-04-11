import enum
from langchain.chains import ConversationChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory, ConversationKGMemory, ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from model_interfaces.interface import ChatSession
from utils.llm import get_max_prompt_size, token_cost
from utils.ui import colour_print

_default_prompt_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""


class CostHandler(BaseCallbackHandler):
    def __init__(self, agent: ChatSession):
        self.agent = agent

    def on_llm_end(self, response, *args, **kwargs):
        in_cost, out_cost = token_cost(response.llm_output["model_name"])
        usage = response.llm_output["token_usage"]
        cost_usd = usage["prompt_tokens"] * in_cost + usage["completion_tokens"] * out_cost
        self.agent.costs_usd += cost_usd


class LangchainAgent(ChatSession):
    def __init__(self, model_name: str, mem_type: "LangchainMemType", max_prompt_size: int):
        super().__init__()
        self.model_name = model_name
        max_prompt_size = min(max_prompt_size, get_max_prompt_size(model_name))
        self.max_token_limit = max_prompt_size
        llm = OpenAI(temperature=0.01, model_name=model_name, callbacks=[CostHandler(self)])
        self.memory = mem_type.create(llm=llm, max_token_limit=max_prompt_size)
        self.conversation = ConversationChain(
            llm=llm,
            verbose=True,
            prompt=mem_type.template,
            memory=self.memory,
        )
        colour_print("CYAN", "WARN: The Langchain agent does not save its memory as of yet. Loading and saving will be NOOPs.")

    @property
    def name(self):
        return f"{super().name} - {self.model_name} - {self.max_token_limit}"

    def reply(self, user_message: str) -> str:
        result = self.conversation.predict(input=user_message)
        return result

    def reset(self):
        self.memory.clear()

    def save(self):
        pass

    def load(self):
        pass


class LangchainMemType(enum.Enum):
    SUMMARY_BUFFER = 0
    KG = 1
    CONVERSATION_ENTITY = 2

    def create(self, llm, max_token_limit: int):
        if self == self.SUMMARY_BUFFER:
            return ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_token_limit)
        elif self == self.KG:
            return ConversationKGMemory(llm=llm, max_token_limit=max_token_limit)
        elif self == self.CONVERSATION_ENTITY:
            return ConversationEntityMemory(llm=llm, max_token_limit=max_token_limit)
        else:
            raise ValueError(f"Unrecognized: {self}")

    @property
    def template(self) -> PromptTemplate:
        if self == self.CONVERSATION_ENTITY:
            return ENTITY_MEMORY_CONVERSATION_TEMPLATE
        else:
            return PromptTemplate(input_variables=["history", "input"], template=_default_prompt_template)


