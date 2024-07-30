import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime
from ltm.agent import InsertedContextAgent
from utils.ui import colour_print
from utils.llm import LLMContext, make_system_message, make_user_message, make_assistant_message, \
    ask_llm, log_llm_call, count_tokens_for_model
from utils.text import stamp_content
from goodai.ltm.agent import Message
from ltm.extended_thinking import message_notes_and_analysis


CostCallback = Callable[[float], None]


@dataclass
class LTMReflexionAgent(InsertedContextAgent):
    user_info: dict = field(default_factory=dict)

    @property
    def max_input_tokens(self) -> int:
        return int(0.9 * (self.max_prompt_size - self.max_completion_tokens))

    def count_tokens(self, *, text: str | list[str] = None, messages: LLMContext = None) -> int:
        return count_tokens_for_model(self.model, text=text, context=messages)

    def reply(
        self, user_message: str, agent_response: Optional[str] = None, cost_callback: Callable[[float], None] = None,
    ) -> str:
        perform_task = """

Here are some additional tasks which you are to perform right now:
{task}

These tasks contains two things: a trigger condition, and a payload.
The trigger conditions have been met for all task. So they can be safely disregarded. Now all you need to do is perform the tasks described in the "payload" fields. 

You should append the result of your task to your reply to the query above.

Append the results of your tasks to your current response."""

        self.now = datetime.now()

        keywords = self.keywords_for_message(user_message, cost_cb=cost_callback)

        if agent_response is None:
            memories = self.collect_memories(
                user_message, max_prompt_size=self.max_prompt_size, previous_interactions=0, cost_cb=cost_callback,
            )
        else:
            memories = []

        # Update task memory
        context = [make_system_message(self.system_message)] + memories + [
            make_user_message(stamp_content(user_message, self.now, dt=self.now))
        ]
        self.add_task(deepcopy(context), cost_callback, user_message)

        # Get any tasks that might be due now.
        extra_tasks = self.progress_and_get_extra_tasks(context)
        if len(extra_tasks) > 0 or agent_response is None:
            task_appendix = ""
            if len(extra_tasks) > 0:
                task_appendix = perform_task.format(task=json.dumps(extra_tasks, indent=2))
            extended_user_msg = user_message + task_appendix
            reply_fn = lambda ctx, t: self._completion(ctx, t, "reply", cost_callback)
            response_text = message_notes_and_analysis(extended_user_msg, memories, self.now, reply_fn)
        else:
            response_text = agent_response

        self.llm_call_idx += 1

        # Remove outdated tasks
        self.remove_completed_tasks()

        # Save interaction to memory
        self.hybrid_memory.add_interaction(
            self.session_id, user_message, response_text, self.now.timestamp(), keywords,
        )

        return response_text

    def _completion(self, context: LLMContext, temperature: float, label: str, cost_callback: CostCallback) -> str:
        response_text = ask_llm(
            context, self.model,
            temperature=temperature, cost_callback=cost_callback, max_overall_tokens=self.max_prompt_size,
        )
        log_llm_call(self.run_name, self.save_name, self.debug_level, label=label)
        return response_text

    def _truncated_completion(self, context: LLMContext, max_messages: int = None, **kwargs) -> str:
        max_messages = max_messages or len(context) + 1
        while len(context) + 1 > max_messages or self.count_tokens(messages=context) > self.max_input_tokens:
            context.pop(1)
        return self._completion(context, **kwargs)

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

    def collect_memories(
        self, user_message: str, max_prompt_size: int, previous_interactions: int, cost_cb: CostCallback,
    ) -> LLMContext:
        relevant_interactions: list[tuple[Message, Message]] = self.get_relevant_memories(user_message, cost_cb)

        # Add the previous messages
        recent_messages = self.hybrid_memory.get_recent_messages(self.session_id, limit=previous_interactions)
        relevant_interactions.extend([(Message(**msg.__dict__), Message(**msg.__dict__)) for msg in recent_messages])

        # Add in memories up to the max prompt size
        context = []
        current_size = 0
        shown_mems = 0
        target_size = max_prompt_size - self.max_message_size

        for interaction in reversed(relevant_interactions):
            user_interaction, assistant_interaction = interaction
            new_context = [
                make_user_message(stamp_content(user_interaction.content, self.now, ts=user_interaction.timestamp)),
                assistant_interaction.as_llm_dict(),
            ] + context
            future_size = self.count_tokens(messages=new_context)

            # If this message is going to be too big, then skip it
            if future_size > target_size:
                continue

            # Add the interaction and count the tokens
            context = new_context
            current_size = future_size

            shown_mems += 1
            if shown_mems >= 100:
                break

            current_size = future_size

        colour_print("lightyellow", f"current context size: {current_size}")

        return context
