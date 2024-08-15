import random
import utils.math as m
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from reporting.generate import get_sorted_scores


PLOT_STD = False
CUSTOM_COLORS = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#76d7c4", "#ffb3e6", "#c4e17f", "#c2c2f0"]

benchmarks = {
    "Isolated": ["Benchmark 3 - 32k (isolated)"],
    "2k": ["Benchmark 3 - 1k", "Benchmark 3 - 1k #2"],
    "32k": ["Benchmark 3 - 32k", "Benchmark 3 - 32k #2"],
    "120k": []
}

aliases = {
    "LLMChatSession - together_ai-mistralai-Mixtral-8x7B-Instruct-v0.1 - 32000": "Mixtral 8x7B",
    "LLMChatSession - together_ai-mistralai-Mixtral-8x7B-Instruct-v0.1 - 32768": "Mixtral 8x7B",
    "LLMChatSession - together_ai-mistralai-Mixtral-8x22B-Instruct-v0.1 - 65536": "Mixtral 8x22B",
    "LLMChatSession - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000": "Llama 3 70B",
    "LLMChatSession - gpt-3.5-turbo - 16384": "GPT-3.5 turbo",
    "LLMChatSession - gpt-4-turbo - 128000": "GPT-4 turbo",
    "LLMChatSession - gpt-4-turbo-2024-04-09 - 128000": "GPT-4 turbo",
    "LLMChatSession - gpt-4o - 128000": "GPT-4o",
    "LLMChatSession - gpt-4o-2024-05-13 - 128000": "GPT-4o",
    "LLMChatSession - claude-3-opus - 200000": "Claude 3 Opus",
    "GeminiProInterface": "Gemini 1.5 Pro",
    "LTMAgentWrapper - claude-3-opus-20240229 - 16384 - QG_JSON_USER_INFO": "LTM Claude 3 Opus",
    "LTMAgentWrapper - claude-3-opus - 16384 - QG_JSON_USER_INFO": "LTM Claude 3 Opus",
    "LTMAgentWrapper - gpt-4-turbo-2024-04-09 - 16384 - QG_JSON_USER_INFO": "LTM GPT-4 turbo",
    "LTMAgentWrapper - gpt-4-turbo - 16384 - QG_JSON_USER_INFO": "LTM GPT-4 turbo",
    "LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - QG_JSON_USER_INFO": "LTM Llama 3 70B",
}

model_order = [
    "Mixtral 8x7B",
    "Mixtral 8x22B",
    "Llama 3 70B",
    "LTM Llama 3 70B",
    "GPT-3.5 turbo",
    "GPT-4 turbo",
    "LTM GPT-4 turbo",
    "GPT-4o",
    "Claude 3 Opus",
    "LTM Claude 3 Opus",
    "Gemini 1.5 Pro",
]


def zipped_sum(sorted_scores: dict[str, list[float]]) -> list[float]:
    return [sum(vs) for vs in zip(sorted_scores.values())]


def main():

    print("Loading results...")
    results = dict()
    for bench_label in ["Isolated", "2k", "32k", "120k", "200k", "500k"]:
        run_prefix = {
            "Isolated": "Benchmark 3 - 32k (isolated)",
            "2k": "Benchmark 3 - 1k",
        }.get(bench_label, f"Benchmark 3 - {bench_label}")
        for run_suffix in ["", " #2"]:
            agents_loaded = set()
            run_name = run_prefix + run_suffix
            print(run_name)
            for agent_name, agent_alias in aliases.items():
                if agent_alias in agents_loaded:
                    raise ValueError(f"Agent {agent_alias} is loading for a second time from run {repr(run_name)}.")
                scores = get_sorted_scores(run_name, agent_name)
                if len(scores) == 0:
                    continue
                agents_loaded.add(agent_alias)
                print(" ", agent_alias)
                if bench_label not in results:
                    results[bench_label] = dict()
                run_results = results[bench_label]
                if agent_alias not in run_results:
                    run_results[agent_alias] = list()
                agent_results = run_results[agent_alias]
                for _ in range(1000):
                    r = sum(random.choice(dataset_results) for dataset_results in scores.values())
                    agent_results.append(r)
                agent_results.append(sum(m.mean(dataset_results) for dataset_results in scores.values()))

    print("-------------------")
    print("Plotting results...")

    data = list()
    positions = list()
    colors = list()
    hatches = list()

    llm_names = [name for name in model_order if not name.startswith("LTM")]
    llm_colors = {name: color for name, color in zip(llm_names, CUSTOM_COLORS)}
    x_ticks_pos = list()
    x_ticks_labels = list()
    current_pos = 0
    for bench_label, bench_results in results.items():
        print(bench_label)
        first_pos = current_pos + 1
        for agent_alias, agent_results in bench_results.items():
            std = m.std(agent_results[:-1])
            print(" ", agent_alias, f"| std: {std:.1f}")
            current_pos += 1
            data.append(agent_results[:-1])
            colors.append(llm_colors[agent_alias.removeprefix("LTM ")])
            hatches.append(agent_alias.startswith("LTM"))
            positions.append(current_pos)
        x_ticks_pos.append(first_pos + 0.5 * (current_pos - first_pos))
        x_ticks_labels.append(bench_label)
        current_pos += 1
    bplot = plt.boxplot(data, patch_artist=True, positions=positions)
    for patch, color, hatch in zip(bplot['boxes'], colors, hatches):
        patch.set_facecolor(color)
        if hatch:
            patch.set_hatch("//")

    # Manual legend
    plt.legend(handles=[
        Patch(color=llm_colors[name], label=name) for name in llm_names
    ] + [
        Patch(facecolor="white", edgecolor="black", label="Only LLM"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="LLM + LTM"),
    ], bbox_to_anchor=(1, 0, 0.5, 1), loc="center left")

    plt.ylabel("Score")
    plt.xticks(x_ticks_pos, x_ticks_labels)
    plt.xlabel("Memory Span")
    plt.show()


if __name__ == "__main__":
    main()
