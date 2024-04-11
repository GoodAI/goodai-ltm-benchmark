import os
from typing import Optional, Callable

import openai
import litellm
from litellm import completion, ModelResponse
from utils.ui import colour_print

litellm.modify_params = True  # To allow it adjusting the prompt for Claude LLMs

LLMMessage = dict[str, str]
LLMContext = list[LLMMessage]
# This list should only contain exact IDs for latest models.
# Add previous ID if specifically supported.
SUPPORTED_MODELS: dict[str, tuple[int, tuple[float, float]]] = {
    "gpt-3.5-turbo-0125": (16_384, (5e-7, 1.5e-6)),
    "gpt-4": (8_192, (3e-5, 6e-5)),
    "gpt-4-1106-preview": (128_000, (1e-5, 3e-5)),
    "gpt-4-turbo-2024-04-09": (128_000, (1e-5, 3e-5)),
    "claude-2.1": (200_000, (8e-6, 2.4e-5)),
    "claude-3-haiku-20240229": (200_000, (2.5e-7, 1.25e-6)),
    "claude-3-sonnet-20240229": (200_000, (3e-6, 1.5e-5)),
    "claude-3-opus-20240229": (200_000, (1.5e-5, 7.5e-5)),
}
MODEL_ALIASES = {
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "gpt-4-1106": "gpt-4-1106-preview",
    "claude-3-haiku": "claude-3-haiku-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-opus": "claude-3-opus-20240229",
}
GPT_CHEAPEST = "gpt-3.5-turbo"
GPT_4_TURBO_BEST = "gpt-4-1106-preview"


def get_model(model_name: Optional[str]) -> str:
    if model_name is None:
        model_name = GPT_CHEAPEST
    debug_name = model_name
    model_name = MODEL_ALIASES.get(model_name, model_name)
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model {debug_name} is not supported and is not an alias.")
    return model_name


def get_max_prompt_size(model: str):
    if model == "gpt-3.5-turbo-instruct":
        colour_print("red", "WARNING: gpt-3.5-turbo-instruct is a completion model.")
        return 4_096
    model = get_model(model)
    prompt_size, _ = SUPPORTED_MODELS[model]
    return prompt_size


def token_cost(model: str) -> tuple[float, float]:
    if model == "text-embedding-ada-002":
        return 2e-9, 2e-9
    if model == "gpt-3.5-turbo-instruct":
        return 1.5e-6, 2e-6
    model = get_model(model)
    _, cost_info = SUPPORTED_MODELS[model]
    return cost_info


def response_cost(response: ModelResponse) -> float:
    input_cost, output_cost = token_cost(response.model)
    usage = response.usage
    return usage.prompt_tokens * input_cost + usage.completion_tokens * output_cost


def set_api_key():
    if openai.api_key is not None:
        return
    openai.api_key = os.getenv("OPENAI_API_KEY")


def ensure_context_len(
    context: LLMContext,
    model: Optional[str] = None,
    max_len: Optional[int] = None,
    response_len: int = 0,
) -> tuple[LLMContext, int]:
    model = get_model(model)
    max_len = max_len or get_max_prompt_size(model)
    messages = list()
    context_tokens = litellm.token_counter(model, messages=context[:1])

    for message in reversed(context[1:]):
        message_tokens = litellm.token_counter(model, text=message["content"])
        if context_tokens + message_tokens + response_len > max_len:
            break
        messages.append(message)
        context_tokens += message_tokens
    messages.reverse()
    context = context[:1] + messages
    # assert len(context) > 1, f"There are messages missing in the context:\n\n{context}"
    return context, context_tokens


def ask_llm(
    context: LLMContext,
    model: Optional[str] = None,
    temperature: float = 1,
    context_length: int = None,
    cost_callback: Callable[[float], None] = None,
    timeout: float = 300,
    max_response_tokens: int = 1000,
) -> str:
    set_api_key()

    model = get_model(model)
    context, context_tokens = ensure_context_len(context, model, context_length, response_len=max_response_tokens)
    response = completion(model=model, messages=context, max_tokens=max_response_tokens, temperature=temperature)

    if cost_callback is not None:
        cost_callback(response_cost(response))
    return response.choices[0].message.content


def make_message(role: str, content: str) -> LLMMessage:
    assert role in {"system", "user", "assistant"}
    return {"role": role, "content": content}


def make_system_message(content: str) -> LLMMessage:
    return make_message("system", content)


def make_user_message(content: str) -> LLMMessage:
    return make_message("user", content)


def make_assistant_message(content: str) -> LLMMessage:
    return make_message("assistant", content)


def context_token_len(context: LLMContext, model: Optional[str] = None, response: bool = False) -> int:
    model = get_model(model)
    num_tokens = 0
    for message in context:
        num_tokens += 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        num_tokens += litellm.token_counter(model, text=message["content"])
    if response:
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def tokens_in_script(script: list[str], model: str = "claude-3-opus-20240229") -> int:
    model = get_model(model)
    num_tokens = 0
    for line in script:
        num_tokens += 4
        num_tokens += litellm.token_counter(model, text=line)

    return num_tokens


def tokens_in_text(text: str, model: str = "claude-3-opus-20240229") -> int:
    model = get_model(model)
    return litellm.token_counter(model, text=text)