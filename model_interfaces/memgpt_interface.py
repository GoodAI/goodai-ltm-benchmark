import json
from dataclasses import dataclass
from typing import Optional

from litellm import cost_per_token
from memgpt import create_client, Admin
from memgpt.client.client import LocalClient
from memgpt.data_types import AgentState, LLMConfig
from memgpt.memory import ChatMemory

from model_interfaces.interface import ChatSession
from utils.llm import get_max_prompt_size



@dataclass
class MemGPTInterface(ChatSession):
    max_prompt_size: int = None
    model: str = None
    max_response_tokens: int = 4096
    agent_state: AgentState = None
    client: LocalClient = None

    @property
    def name(self):
        name = f"{super().name} - {self.model} - {self.max_prompt_size}"
        return name.replace("/", "-")

    def __post_init__(self):
        super().__post_init__()

        if self.max_prompt_size is None:
            self.max_prompt_size = get_max_prompt_size(self.model)

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:

        if not self.client:
            self.reset()

        send_message_response = self.client.user_message(agent_id=self.agent_state.id, message=user_message)
        # print(f"Recieved response: \n{json.dumps(send_message_response.messages, indent=4)}")

        # Costs
        cost_prompt, cost_completion = cost_per_token(self.model, send_message_response.usage.prompt_tokens, send_message_response.usage.completion_tokens)
        self.costs_usd += (cost_prompt + cost_completion)

        texts = []
        for message in send_message_response.messages:
            response_text = message.get("assistant_message", None)
            if response_text:
                texts.append(response_text)

        return "\n".join(texts)

    def reset(self):
        self.client = create_client()
        llm_config = LLMConfig(model=self.model, context_window=self.max_prompt_size,
                               model_endpoint='https://api.openai.com/v1', model_endpoint_type='openai')

        agent = self.client.get_agent(agent_name="benchmark_agent")
        if agent:
            self.client.delete_agent(agent.id)

        # Create an agent
        self.agent_state = self.client.create_agent(name="benchmark_agent", llm_config=llm_config,
                                                    memory=ChatMemory(human="", persona="I am a friendly AI."))
        # self.agent_state = self.client.create_agent(name="benchmark_agent", memory=ChatMemory(human="", persona="I am a friendly AI."))
        print(f"Created agent: {self.agent_state.name} with ID {str(self.agent_state.id)}")

    def save(self):
        self.client.save()

    def load(self):
        self.client = create_client()
        self.agent_state = self.client.get_agent(agent_name="benchmark_agent")


