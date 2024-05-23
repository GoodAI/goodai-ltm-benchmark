import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from reporting.generate import get_summary_data
from matplotlib.patches import Patch


aliases = {
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "claude-3-opus": "claude-3-opus-20240229",
}


def get_data(run_name: str, agent_name: str):
    # TODO: Add the corresponding result files
    if run_name.endswith("1k") and "Llama-3-70B" in agent_name:
        return dict(score=1.9, score_std=2.3)
    for i in range(2):
        try:
            return get_summary_data(run_name, agent_name)
        except IndexError:
            if i == 2:
                return
            llm_name = agent_name.split(" - ")[1]
            if llm_name not in aliases:
                return
            agent_name = agent_name.replace(llm_name, aliases[llm_name])


def main():
    llm_names = [
        "together_ai-mistralai-Mixtral-8x7B-Instruct-v0.1",
        "together_ai-mistralai-Mixtral-8x22B-Instruct-v0.1",
        "huggingface-gradientai-Llama-3-70B-Instruct-Gradient-262k",
        "gpt-3.5-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4o",
        "claude-3-opus-20240229",
    ]
    llm_legend_names = [
        "Mixtral 8x7B",
        "Mixtral 8x22B",
        "Llama 3 70B",
        "GPT-3.5 turbo",
        "GPT-4 turbo",
        "GPT-4o",
        "Claude 3 Opus",
    ]
    llm_colors = {name: color for name, color in zip(llm_names, TABLEAU_COLORS.keys())}
    llm_colors["together_ai-meta-llama-Llama-3-70b-chat-hf"] = llm_colors["huggingface-gradientai-Llama-3-70B-Instruct-Gradient-262k"]
    agent_names = [
        "LLMChatSession - together_ai-mistralai-Mixtral-8x7B-Instruct-v0.1 - 32000",
        "LLMChatSession - together_ai-mistralai-Mixtral-8x22B-Instruct-v0.1 - 32000",
        "HFChatSession - huggingface-gradientai-Llama-3-70B-Instruct-Gradient-262k - 32768",
        "LLMChatSession - gpt-3.5-turbo - 16384",
        "LLMChatSession - gpt-4-turbo - 128000",
        "LLMChatSession - gpt-4o - 128000",
        "LLMChatSession - claude-3-opus - 200000",
        "LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - SEMANTIC_ONLY",
        "LTMAgentWrapper - gpt-4-turbo - 16384 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - gpt-4-turbo - 16384 - SEMANTIC_ONLY",
        "LTMAgentWrapper - claude-3-opus - 16384 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - claude-3-opus - 16384 - SEMANTIC_ONLY",
    ]
    bar_groups = [
        (2, agent_names),
        (32, agent_names),
        (120, agent_names),
        (200, agent_names),
        (500, agent_names),
    ]

    x_ticks_labels = list()
    x_ticks_pos = list()
    current_pos = 0
    for span, group_agents in bar_groups:
        run_name = f"Benchmark 3 - {span if span > 2 else 1}k"
        print(run_name)
        start_pos = current_pos
        num_group_agents = 0
        print(current_pos)
        for agent_name in group_agents:
            print(agent_name)
            data = get_data(run_name, agent_name)
            if data is None:
                continue
            llm = agent_name.split(" - ")[1]
            llm = aliases.get(llm, llm)
            kwargs = dict()
            if agent_name.startswith("LTMAgentWrapper"):
                kwargs["edgecolor"] = "white"
            if agent_name.endswith("SEMANTIC_ONLY"):
                kwargs["hatch"] = "/"
            elif agent_name.endswith("QG_JSON_USER_INFO"):
                kwargs["hatch"] = "//"
            color = llm_colors[llm]
            score = data["score"]
            error = data["score_std"]
            plt.bar(current_pos, score, yerr=[[min(error, score)], [min(error, 11 - score)]], color=color, **kwargs)
            current_pos += 1
            num_group_agents += 1
        x_ticks_labels.append(f"{span}k")
        x_ticks_pos.append(start_pos + 0.5 * (num_group_agents - 1))
        current_pos += 1

    # Manual legend
    plt.legend(handles=[
        Patch(color=llm_colors[name], label=label)
        for name, label in zip(llm_names, llm_legend_names)
    ] + [
        Patch(facecolor="black", hatch="//", edgecolor="white", label="LTM Agent 1"),
        Patch(facecolor="black", hatch="/", edgecolor="white", label="LTM Agent 2"),
    ], bbox_to_anchor=(-0.4, 0, 0.35, 1), loc="lower left")

    plt.ylabel("Score")
    plt.xticks(x_ticks_pos, x_ticks_labels)
    plt.xlabel("Memory Span")
    plt.show()


if __name__ == "__main__":
    main()
