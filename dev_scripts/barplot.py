import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from collections import defaultdict
from reporting.generate import get_summary_data
from runner.master_log import MasterLog
from utils.constants import DATA_DIR, EventType
from utils.llm import tokens_in_text
from utils.math import mean_std
import numpy as np
from matplotlib.patches import Patch


aliases = {
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "claude-3-opus": "claude-3-opus-20240229",
}


def get_data(run_name: str, agent_name: str):
    try:
        return get_summary_data(run_name, agent_name)
    except IndexError:
        llm_name = agent_name.split(" - ")[1]
        new_agent_name = agent_name.replace(llm_name, aliases[llm_name])
        return get_summary_data(run_name, new_agent_name)


def main():
    run_names = [(f"Benchmark 3 - {span}k", span) for span in [120, 200, 500]]
    x_labels = [f"{span}k" for _, span in run_names]
    # TODO: Rename all result dirs that have "gpt-4-turbo" or "claude-3-opus" to the exact name.
    # TODO: See if that is already fixed for new results.

    llm_names = [
        "gpt-4-turbo-2024-04-09",
        "claude-3-opus-20240229",
        "together_ai-meta-llama-Llama-3-70b-chat-hf",
    ]
    llm_legend_names = [
        "GPT-4 turbo",
        "Claude 3 Opus",
        "Llama 3 70B",
    ]
    llm_colors = {name: color for name, color in zip(llm_names, TABLEAU_COLORS.keys())}
    agent_names = [
        "LLMChatSession - gpt-4-turbo - 128000",
        "LLMChatSession - claude-3-opus - 200000",
        "LTMAgentWrapper - gpt-4-turbo - 16384 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - gpt-4-turbo - 16384 - SEMANTIC_ONLY",
        "LTMAgentWrapper - claude-3-opus - 16384 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - claude-3-opus - 16384 - SEMANTIC_ONLY",
        "LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - SEMANTIC_ONLY",
    ]
    bar_groups = [
        #(32, "agent_name"),
        (120, agent_names),
        (200, agent_names),
        (500, agent_names),
    ]

    x_ticks_labels = list()
    x_ticks_pos = list()
    current_pos = 0
    for span, group_agents in bar_groups:
        run_name = f"Benchmark 3 - {span}k"
        print(run_name)
        x_ticks_labels.append(f"{span}k")
        x_ticks_pos.append(current_pos + 0.5 * len(group_agents))
        for agent_name in group_agents:
            print(agent_name)
            data = get_data(run_name, agent_name)
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
            plt.bar(current_pos, data["score"], color=color, **kwargs)
            current_pos += 1
        current_pos += 1

    # Manual legend
    plt.legend(handles=[
        Patch(color=llm_colors[name], label=label)
        for name, label in zip(llm_names, llm_legend_names)
    ] + [
        Patch(facecolor="black", hatch="//", edgecolor="white", label="LTM Agent 1"),
        Patch(facecolor="black", hatch="/", edgecolor="white", label="LTM Agent 2"),
    ])

    plt.ylabel("Score")
    plt.xticks(x_ticks_pos, x_ticks_labels)
    plt.xlabel("Memory Span")
    plt.show()


if __name__ == "__main__":
    main()
