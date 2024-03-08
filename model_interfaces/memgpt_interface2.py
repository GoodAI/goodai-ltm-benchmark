from dataclasses import dataclass, field

from memgpt.client.client import LocalClient

from model_interfaces.interface import ChatSession


from memgpt import create_client


@dataclass
class MemGPTChatSession(ChatSession):

    _max_prompt_size: int = 8192
    _proxy_path: str = "model_interfaces/memgpt_proxy.py"
    client: LocalClient = field(default_factory=create_client)
    agent_info:  = None

    def __post_init__(self):
        self.client = create_client()
        self.reset()


    def reply(self, user_message: str) -> str:
        response = self.client.user_message(agent_id=self.agent_info.id, message=user_message)

        return response

    def reset(self):
        self.agent_info =  self.client.create_agent(name=)

    def save(self):
        self.client.save()

    def load(self):
        self.clien