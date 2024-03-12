import os
from copy import deepcopy
from typing import Optional, Callable

import openai
import tiktoken
from openai import OpenAI
from utils.ui import colour_print

LLMMessage = dict[str, str]
LLMContext = list[LLMMessage]
MODEL_LIST_CACHE: Optional[set[str]] = None
client = OpenAI()


def allowed_models() -> set[str]:
    global MODEL_LIST_CACHE
    if MODEL_LIST_CACHE is None:
        models = client.models.list()
        MODEL_LIST_CACHE = set(m.id for m in models.data if m.id.startswith("gpt-") and "instruct" not in m.id)
        MODEL_LIST_CACHE.add("claude-2.1")
        MODEL_LIST_CACHE.add("claude-3-opus-20240229")
    return deepcopy(MODEL_LIST_CACHE)


def get_model(model_name: Optional[str]) -> str:
    if model_name is None:
        model_name = "gpt-3.5-turbo"
    assert model_name in allowed_models()
    return model_name


def get_max_prompt_size(model: str):
    if model == "gpt-3.5-turbo-instruct":
        colour_print("red", "WARNING: gpt-3.5-turbo-instruct is a completion model.")
        return 4_096
    assert model in allowed_models()
    if model in ["gpt-4-1106-preview", "gpt-4-0125-preview"]:
        return 128_000
    if model == "gpt-3.5-turbo-0125":
        return 16_384
    if "32k" in model:
        return 32_768
    if "16k" in model:
        return 16_384
    if model.startswith("gpt-4"):
        return 8_192
    if model == "claude-2.1":
        return 200_000
    if model == "claude-3-opus-20240229":
        return 200_000
    if model == "gpt-3.5-turbo":
        return 16_384
    return 4_096


def token_cost(model: str) -> tuple[float, float]:
    if model == "gpt-3.5-turbo-instruct":
        return 0.000_001_5, 0.000_002
    if model == "gpt-3.5-turbo":
        return 0.000_000_5, 0.000_001_5
    if model == "gpt-3.5-turbo-0125":
        return 0.000_000_5, 0.000_001_5
    if model in ["gpt-4-1106-preview", "gpt-4-0125-preview"]:
        return 0.000_01, 0.000_03
    if model.startswith("gpt-4-32k"):
        return 0.000_06, 0.000_12
    if model.startswith("gpt-4"):
        return 0.000_03, 0.000_06
    if model.startswith("gpt-3.5-turbo"):
        return 0.000_001, 0.000_002
    if model == "claude-2.1":
        return 8e-06, 2.4e-05
    if model == "claude-3-opus-20240229":
        return 0.000_015, 0.000_075
    if model == "text-embedding-ada-002":
        return 0.000_000_002, 0.000_000_002
    raise ValueError(f"There's no cost registered for model {model}.")


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
