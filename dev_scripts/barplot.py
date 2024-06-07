import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PLOT_STD = False
CUSTOM_COLORS = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#76d7c4", "#ffb3e6", "#c4e17f", "#c2c2f0"]

aliases = {
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B",
    "Mixtral-8x22B-instruct-v0.1": "Mixtral 8x22B",
    "Meta Llama 3 70B Instruct": "Llama 3 70B",
    "gpt-3.5-turbo": "GPT-3.5 turbo",
    "gpt-4-turbo-2024-04-09": "GPT-4 turbo",
    "gpt-4o": "GPT-4o",
    "Claude Opus": "Claude 3 Opus",
    "Gemini 1.5 Pro": "Gemini 1.5 Pro",
    "LTM Agent 1 Meta Llama 3 70B Instruct": "LTM Llama 3 70B",
    "LTM Agent 1 gpt-4-turbo-2024-04-09": "LTM GPT-4 turbo",
    "LTM Agent 1 Opus": "LTM Claude 3 Opus"
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


def load_csv_data() -> dict:
    results = dict()
    csv_path = input("Enter the path to the CSV results: ")
    with open(csv_path) as fd:
        for line in fd:
            if line[:2] in ["Is", "1k", "32", "12", "20", "50"]:
                k = line.split(" ")[0]
                if k == "1k":
                    k = "2k"
                results[k] = dict()
            elif line[:3] in ["gpt", "Cla", "LTM", "Mix", "Met", "Gem"]:
                model, _, score, std = line.split(",")[:4]
                score = score.strip()
                std = std.strip()
                score = float(score) if score != "" else 0
                std = float(std) if std != "" else 0
                model = aliases[model]
                results[k][model] = dict(score=score, score_std=std)
            if line.startswith("Others not included"):
                break
    return results


def main():
    x_ticks_labels = list()
    x_ticks_pos = list()
    current_pos = 0
    llm_colors = None
    llm_names = None
    for span_label, agent_values in load_csv_data().items():
        if llm_colors is None:
            llm_names = [name for name in agent_values.keys() if not name.startswith("LTM")]
            llm_colors = {name: color for name, color in zip(llm_names, CUSTOM_COLORS)}
        start_pos = current_pos
        num_group_agents = 0
        for agent_name in model_order:
            if agent_name not in agent_values:
                continue
            result = agent_values[agent_name]
            kwargs = dict(edgecolor="black")
            if agent_name.startswith("LTM"):
                kwargs["hatch"] = "//"
            color = llm_colors[agent_name.removeprefix("LTM ")]
            score = result["score"]
            error = result["score_std"]
            if PLOT_STD:
                kwargs["yerr"] = [[min(error, score)], [min(error, 11 - score)]]
            plt.bar(current_pos, score, color=color, **kwargs)
            current_pos += 1
            num_group_agents += 1
        x_ticks_labels.append(span_label)
        x_ticks_pos.append(start_pos + 0.5 * (num_group_agents - 1))
        current_pos += 1

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
