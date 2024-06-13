from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

class ResponseAgent:
    def __init__(self, model_name: str):
        self.chat_model = ChatOpenAI(model_name=model_name)

    def generate_response(self, query: str, result: str) -> str:
        messages = [
            HumanMessage(content=f"Given the following query:\n{query}\n\nAnd the following result:\n{result}\n\nGenerate a final response.")
        ]
        response = self.chat_model(messages)
        return response.content