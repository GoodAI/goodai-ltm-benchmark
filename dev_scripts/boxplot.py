import random
import utils.util_math as m
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from reporting.generate import get_sorted_scores
from utils.ui import colour_print


CUSTOM_COLORS = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#76d7c4", "#ffb3e6", "#c4e17f", "#c2c2f0"]

# Agent names may vary over time, so link them here with their corresponding short labels.
# The order is important, and will be kept the same in the figure / table.
# The script expects results from all aliases listed here, so remove any agent from both
# the aliases and context_lengths dicts if there are no results for them.
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

context_lengths = {
    "Mixtral 8x7B": 32_000,
    "Mixtral 8x22B": 65_536,
    "Llama 3 70B": 8_000,
    "GPT-3.5 turbo": 16_384,
    "GPT-4 turbo": 128_000,
    "GPT-4o": 128_000,
    "Claude 3 Opus": 200_000,
    "Gemini 1.5 Pro": 1_000_000,
    "LTM Claude 3 Opus": 16_384,
    "LTM GPT-4 turbo": 16_384,
    "LTM Llama 3 70B": 8_000,
}


def main():

    print("Loading results...")
    extended = set()
    actual_means = dict()
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
                scores = get_sorted_scores(run_name, agent_name)
                if len(scores) == 0:
                    continue
                if agent_alias in agents_loaded:
                    colour_print(
                        "red",
                        f"WARNING: Attempt to load agent {agent_alias} for a second time from run {repr(run_name)}.\n"
                        "Only the first set of results will be considered.",
                    )
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
                # Compute the actual mean (not sampled) and store it separately
                key = f"{bench_label}/{agent_alias}"
                actual_mean = sum(m.mean(dataset_results) for dataset_results in scores.values())
                actual_means[key] = actual_means.get(key, []) + [actual_mean]
                if run_suffix != "":
                    extended.add(key)

    print("-------------------")
    print("Plotting results...")

    data = list()
    positions = list()
    colors = list()
    hatches = list()

    llm_names = list()
    for name in aliases.values():
        if not name.startswith("LTM") and name not in llm_names:
            llm_names.append(name)
    assert len(llm_names) == len(CUSTOM_COLORS), llm_names
    llm_colors = {name: color for name, color in zip(llm_names, CUSTOM_COLORS)}
    x_ticks_pos = list()
    x_ticks_labels = list()
    current_pos = 0
    for bench_label, bench_results in results.items():
        print(bench_label)
        first_pos = current_pos + 1
        for agent_alias, agent_results in bench_results.items():
            std = m.std(agent_results)
            print(" ", agent_alias, f"| std: {std:.1f}")
            current_pos += 1
            data.append(agent_results)
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

    print("-------------------")
    print("--- LaTeX Table ---")
    print("-------------------")

    # Fill in values
    best_scores = set()
    rows = [[name, str(ctx_len)] for name, ctx_len in context_lengths.items()]
    for bench_label, bench_results in results.items():
        max_bench_score = -1
        best_score_key = ""
        for row_idx, (agent_alias, agent_results) in enumerate(bench_results.items()):
            means = actual_means[f"{bench_label}/{agent_alias}"]
            actual_mean = sum(means) / len(means)
            rows[row_idx].extend([actual_mean, m.std(agent_results)])
            if max_bench_score < 0 or actual_mean > max_bench_score:
                max_bench_score = actual_mean
                best_score_key = f"{bench_label}/{agent_alias}"
        assert max_bench_score >= 0
        best_scores.add(best_score_key)

    # Highlight best scores and underline extended results
    for bench_idx, (bench_label, bench_results) in enumerate(results.items()):
        for row_idx, (agent_alias, agent_results) in enumerate(bench_results.items()):
            i = 2 + 2 * bench_idx
            mean, std = [f"{v:.1f}" for v in rows[row_idx][i:i+2]]
            key = f"{bench_label}/{agent_alias}"
            if key in best_scores:
                mean, std = [f"\\textbf{{{v}}}" for v in [mean, std]]
            if key in extended:
                mean, std = [f"\\underline{{{v}}}" for v in [mean, std]]
            rows[row_idx][i] = mean
            rows[row_idx][i+1] = std

    # Print and copy to clipboard
    latex_table = "\n".join([" & ".join(row) + r" \\" for row in rows])
    print(latex_table)
    try:
        import pyperclip
        pyperclip.copy(latex_table)
        colour_print("green", "Copied to clipboard. Use ctrl-v to paste it.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
