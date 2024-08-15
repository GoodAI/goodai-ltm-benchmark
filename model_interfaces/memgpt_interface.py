import json
from dataclasses import dataclass
from typing import Optional

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

        self.client = create_client()
        llm_config = LLMConfig(model='gpt-4-turbo', context_window=self.max_prompt_size, model_endpoint='https://api.openai.com/v1', model_endpoint_type='openai')

        agent = self.client.get_agent(agent_name="benchmark_agent")
        if agent:
            self.client.delete_agent(agent.id)

        # Create an agent
        self.agent_state = self.client.create_agent(name="benchmark_agent", llm_config=llm_config, memory=ChatMemory(human="", persona="I am a friendly AI."))
        # self.agent_state = self.client.create_agent(name="benchmark_agent", memory=ChatMemory(human="", persona="I am a friendly AI."))
        print(f"Created agent: {self.agent_state.name} with ID {str(self.agent_state.id)}")


    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:

        send_message_response = self.client.user_message(agent_id=self.agent_state.id, message=user_message)
        # print(f"Recieved response: \n{json.dumps(send_message_response.messages, indent=4)}")

        # Costs
        cost_in = send_message_response.usage.prompt_tokens * (10 / 1_000_000)
        cost_out = send_message_response.usage.completion_tokens * (30 / 1_000_000)
        self.costs_usd += (cost_in + cost_out)

        texts = []
        for message in send_message_response.messages:
            response_text = message.get("assistant_message", None)
            if response_text:
                texts.append(response_text)

        return "\n".join(texts)


    def reset(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

