import os
from typing import Optional, Callable
import litellm
from litellm import completion
from litellm.exceptions import ContextWindowExceededError
from transformers import AutoTokenizer

litellm.modify_params = True  # To allow it adjusting the prompt for Claude LLMs
claude_adjust_factor = 1.1  # Approximate the real token count given by the API

LLMMessage = dict[str, str]
LLMContext = list[LLMMessage]
# This list should only contain exact IDs for latest models.
# Add previous ID if specifically supported.
MODEL_ALIASES = {
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "gpt-4-1106": "gpt-4-1106-preview",
    "claude-3-haiku": "claude-3-haiku-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-opus": "claude-3-opus-20240229",
}
GPT_CHEAPEST = "gpt-3.5-turbo"
GPT_4_TURBO_BEST = "gpt-4-turbo"
LEAST_EFFICIENT_TOKENISER = "claude-3-opus"
litellm.model_alias_map = MODEL_ALIASES


def model_from_alias(model: str):
    return litellm.model_alias_map.get(model, model)


def get_max_prompt_size(model: str):
    model = model_from_alias(model)
    model_info = litellm.model_cost.get(model, None)
    if model_info:
        return model_info["max_input_tokens"]
    return 0


def token_cost(model: str) -> tuple[float, float]:
    model = model_from_alias(model)
    input_cost = litellm.model_cost[model]["input_cost_per_token"]
    output_cost = litellm.model_cost[model]["output_cost_per_token"]
    return input_cost, output_cost


def set_api_key():
    try:
        if litellm.openai_key is None:
            litellm.openai_key = os.getenv("OPENAI_API_KEY")
    except:
        pass

    try:
        if litellm.anthropic_key is None:
            litellm.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    except:
        pass


def ensure_context_len(
    context: LLMContext,
    model: str = LEAST_EFFICIENT_TOKENISER,
    max_len: Optional[int] = None,
    response_len: int = 0,
) -> tuple[LLMContext, int]:
    model = model_from_alias(model)
    max_len = max_len or get_max_prompt_size(model)
    messages = list()

    context_tokens = count_tokens_for_model(model=model, context=context)
    if model.startswith("claude"):
        context_tokens *= claude_adjust_factor

    for message in reversed(context[1:]):
        message_tokens = count_tokens_for_model(model=model, context=[message])
        if model.startswith("claude"):
            message_tokens *= claude_adjust_factor
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
    model: str,
    temperature: float = 1,
    context_length: int = None,
    cost_callback: Callable[[float], None] = None,
    timeout: float = 300,
    max_response_tokens: int = 1024,
) -> str:
    global claude_adjust_factor
    set_api_key()
    model = model_from_alias(model)
    max_response_tokens = litellm.get_max_tokens(model) if max_response_tokens is None else max_response_tokens
    context, context_tokens = ensure_context_len(context, model, context_length, response_len=max_response_tokens)

    # Anthropic tokenizer is currently very inaccurate.
    # Which is why we have to rely on this loop.
    actual_count = -1
    while True:
        try:
            if model.startswith("huggingface"):
                run_context = [{"role": "system", "content": create_huggingface_chat_context(model, context)}]
            else:
                run_context = context

            response = completion(
                model=model,
                messages=run_context,
                max_tokens=max_response_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            break
        except ContextWindowExceededError as exc:
            if model.startswith("claude"):
                actual_count = int(str(exc).split(" tokens > ")[0].split("prompt is too long: ")[1])
            context = context[:1] + context[2:]

    if actual_count > 0:
        claude_adjust_factor *= 0.8
        claude_adjust_factor += 0.2 * (actual_count / context_tokens)
        
    if cost_callback is not None:
        cost_callback(litellm.completion_cost(response))
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


def count_tokens_for_model(model: str = LEAST_EFFICIENT_TOKENISER, context: LLMContext = None, script: list[str] = None, text: str = None) -> int:

    token_count = 0

    if "huggingface" in model.lower():
        model_only = model[model.index("/")+1:]
        tokeniser = AutoTokenizer.from_pretrained(model_only)

        if context:
            c = context[1:] if context[0]["role"] == "system" else context
            c = c[1:] if c[0]["role"] == "assistant" else c
            try:
                token_count += len(tokeniser.apply_chat_template(c))
            except:
                print(c)
                assert False

        if script:
            token_count += len(tokeniser.encode(script))

        if text:
            token_count += len(tokeniser.encode(text))

    else:
        if context:
            token_count += litellm.token_counter(model, messages=context)

        if script:
            for line in script:
                token_count += 4
                token_count += litellm.token_counter(model, text=line)

        if text:
            token_count += litellm.token_counter(model, text=text)

    return token_count


def create_huggingface_chat_context(model:str, context: LLMContext):
    model_only = model[model.index("/") + 1:]
    tokenizer = AutoTokenizer.from_pretrained(model_only)
    c = context[1:] if context[0]["role"] == "system" else context
    c = c[1:] if c[0]["role"] == "assistant" else c
    try:
        return tokenizer.apply_chat_template(c, tokenize=False)
    except:
        print(c)
        assert False