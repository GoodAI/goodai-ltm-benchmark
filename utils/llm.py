import os
from typing import Optional, Callable

import openai
import tiktoken
from openai import OpenAI
from utils.ui import colour_print

LLMMessage = dict[str, str]
LLMContext = list[LLMMessage]
SUPPORTED_MODELS: dict[str, tuple[int, tuple[float, float]]] = {
    "gpt-3.5-turbo": (16_384, (5e-7, 1.5e-6)),
    "gpt-4": (8_192, (3e-5, 6e-5)),
    "gpt-4-32k": (32_768, (6e-5, 1.2e-4)),
    "gpt-4-turbo-preview": (128_000, (1e-5, 3e-5)),
    "gpt-4-1106-preview": (128_000, (1e-5, 3e-5)),
    "gpt-4-0125-preview": (128_000, (1e-5, 3e-5)),
    "claude-2.1": (200_000, (8e-6, 2.4e-5)),
    "claude-3-haiku-20240229": (200_000, (2.5e-7, 1.25e-6)),
    "claude-3-sonnet-20240229": (200_000, (3e-6, 1.5e-5)),
    "claude-3-opus-20240229": (200_000, (1.5e-5, 7.5e-5)),
}
MODEL_ALIASES = {
    "gpt-4-turbo": "gpt-4-turbo-preview",
    "gpt-4-1106": "gpt-4-1106-preview",
    "gpt-4-0125": "gpt-4-0125-preview",
    "claude-3-haiku": "claude-3-haiku-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-opus": "claude-3-opus-20240229",
}
GPT_CHEAPEST = "gpt-3.5-turbo"
GPT_4_TURBO_BEST = "gpt-4-1106-preview"
client = OpenAI()


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


def response_cost(response: openai.ChatCompletion) -> float:
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
    system_message: bool = True,
) -> tuple[LLMContext, int]:
    max_len = max_len or get_max_prompt_size(model)
    messages = list()

    if system_message:
        sys_idx = 1
    else:
        sys_idx = 0

    context_tokens = context_token_len(context[:sys_idx])

    for message in reversed(context[sys_idx:]):
        message_tokens = context_token_len([message])
        if context_tokens + message_tokens + response_len > max_len:
            break
        messages.append(message)
        context_tokens += message_tokens
    messages.reverse()
    context = context[:sys_idx] + messages
    # assert len(context) > 1, f"There are messages missing in the context:\n\n{context}"
    return context, context_tokens


def ask_llm(
    context: LLMContext,
    model: Optional[str] = None,
    temperature: float = 1,
    context_length: int = None,
    cost_callback: Callable[[float], None] = None,
    timeout: float = 300,
    max_tokens: Optional[int] = None,
) -> str:
    set_api_key()
    model = get_model(model)
    context, context_tokens = ensure_context_len(context, model, context_length, response_len=256)
    response = openai.chat.completions.create(
        model=model,
        messages=context,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
    )
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
    # Extracted from here:
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    encoder = tiktoken.encoding_for_model(get_model(model))
    num_tokens = 0
    for message in context:
        num_tokens += 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        num_tokens += len(encoder.encode(message["content"]))
    if response:
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
