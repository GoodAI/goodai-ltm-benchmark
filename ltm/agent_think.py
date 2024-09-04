from dataclasses import dataclass
from typing import Callable
from ltm.agent import LTMAgent
from utils.ui import colour_print
from utils.llm import LLMContext, make_user_message, make_assistant_message


@dataclass
class LTMReflexionAgent(LTMAgent):

    def _truncated_completion(self, context: LLMContext, max_messages: int = None, **kwargs) -> str:
        max_messages = max_messages or len(context) + 1
        while len(context) + 1 > max_messages or self.count_tokens(messages=context) > self.max_input_tokens:
            context.pop(1)
        return self.ask_llm(context, **kwargs)

    def _reflect(self, context: LLMContext, response: str, cost_callback: Callable[[float], None] = None) -> str:
        context = context[:]
        reflect = lambda q: self._elaborate(context, q, cost_callback=cost_callback)
        user_content = context[-1]["content"]
        colour_print("lightblue", response)
        context.append(make_assistant_message(response))
        r = reflect(
            "In relation to your last response, what information do you have that is related to it? Give me just "
            "a short list of summarized points. A very short text.",
        )
        colour_print("lightyellow", r)
        r = reflect("Did you use that information effectively? Is there anything that you missed?")
        colour_print("lightblue", r)
        r = reflect("Based on this reflection, would you like to change or improve your response? Answer simply 'yes' or 'no'.")
        colour_print("lightyellow", r)
        if "yes" in r.lower():
            response = reflect(f"Alright. Let's try again then.\n(going back to the original message... act as if nothing happened)\n\n{user_content}")
            colour_print("lightblue", response)
        return response

    def _elaborate(self, context: LLMContext, question: str, cost_callback: Callable[[float], None] = None) -> str:
        context.append(make_user_message(question))
        reflection = self._truncated_completion(
            context, temperature=self.temperature, label="reflection", cost_callback=cost_callback,
        )
        context.append(make_assistant_message(reflection))
        return reflection
