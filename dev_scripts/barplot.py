import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from reporting.generate import get_summary_data
from matplotlib.patches import Patch


PLOT_STD = False


aliases = {
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "claude-3-opus": "claude-3-opus-20240229",
}


def get_data(run_name: str, agent_name: str):
    for i in range(2):
        try:
            return get_summary_data(run_name, agent_name)
        except IndexError:
            if i == 2 or " - " not in agent_name:
                return
            llm_name = agent_name.split(" - ")[1]
            if llm_name not in aliases:
                return
            agent_name = agent_name.replace(llm_name, aliases[llm_name])


def main():
    llm_names = [
        "together_ai-mistralai-Mixtral-8x7B-Instruct-v0.1",
        "together_ai-mistralai-Mixtral-8x22B-Instruct-v0.1",
        "together_ai-meta-llama-Llama-3-70b-chat-hf",
        "gpt-3.5-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4o",
        "claude-3-opus-20240229",
        "gemini",
    ]
    llm_legend_names = [
        "Mixtral 8x7B",
        "Mixtral 8x22B",
        "Llama 3 70B",
        "GPT-3.5 turbo",
        "GPT-4 turbo",
        "GPT-4o",
        "Claude 3 Opus",
        "Gemini 1.5 Pro",
    ]
    llm_colors = {name: color for name, color in zip(llm_names, TABLEAU_COLORS.keys())}
    agent_names = [
        "LLMChatSession - together_ai-mistralai-Mixtral-8x7B-Instruct-v0.1 - 32000",
        "LLMChatSession - together_ai-mistralai-Mixtral-8x22B-Instruct-v0.1 - 32000",
        "LLMChatSession - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000",
        "LLMChatSession - gpt-3.5-turbo - 16384",
        "LLMChatSession - gpt-4-turbo - 128000",
        "LLMChatSession - gpt-4o - 128000",
        "LLMChatSession - claude-3-opus - 200000",
        "GeminiProInterface",
        "LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - gpt-4-turbo - 16384 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - claude-3-opus - 16384 - QG_JSON_USER_INFO",
    ]
    bar_groups = [
        (0, agent_names),
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
        label = {
            0: "Isolated",
            2: "1k"
        }.get(span, f"{span}k")
        run_name = f"Benchmark 3 - {label}"
        print(run_name)
        start_pos = current_pos
        num_group_agents = 0
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
            if PLOT_STD:
                kwargs["yerr"] = [[min(error, score)], [min(error, 11 - score)]]
            plt.bar(current_pos, score, color=color, **kwargs)
            current_pos += 1
            num_group_agents += 1
        x_ticks_labels.append(label)
        x_ticks_pos.append(start_pos + 0.5 * (num_group_agents - 1))
        current_pos += 1

    # Manual legend
    plt.legend(handles=[
        Patch(color=llm_colors[name], label=label)
        for name, label in zip(llm_names, llm_legend_names)
    ] + [
        Patch(facecolor="black", label="Only LLM"),
        Patch(facecolor="black", hatch="//", edgecolor="white", label="LLM + LTM"),
    ], bbox_to_anchor=(1, 0, 0.5, 1), loc="center left")

    plt.ylabel("Score")
    plt.xticks(x_ticks_pos, x_ticks_labels)
    plt.xlabel("Memory Span")
    plt.show()


if __name__ == "__main__":
    main()
