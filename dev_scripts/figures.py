import matplotlib.pyplot as plt
from collections import defaultdict
from reporting.generate import get_summary_data


def main():
    run_names = [(f"Benchmark 3 - {span}k", span) for span in [120, 200, 500]]
    x_labels = [f"{span}k" for _, span in run_names]
    # TODO: Rename all result dirs that have "gpt-4-turbo" or "claude-3-opus" to the exact name.
    # TODO: See if that is already fixed for new results.
    aliases = {
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "claude-3-opus": "claude-3-opus-20240229",
    }
    agent_names = [
        ("LLMChatSession - gpt-4-turbo - 128000", "GPT4"),
        ("LLMChatSession - claude-3-opus - 200000", "Claude"),
        ("LTMAgentWrapper - gpt-4-turbo - 16384 - QG_JSON_USER_INFO", "A1(GPT4)"),
        # ("LTMAgentWrapper - gpt-4-turbo - 16384 - SEMANTIC_ONLY", "A2(GPT4)"),
        ("LTMAgentWrapper - claude-3-opus - 16384 - QG_JSON_USER_INFO", "A1(Claude)"),
        # ("LTMAgentWrapper - claude-3-opus - 16384 - SEMANTIC_ONLY", "A2(Claude)"),
        ("LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - QG_JSON_USER_INFO", "A1(Llama)"),
        # ("LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - SEMANTIC_ONLY", "A2(Llama)"),
    ]
    metrics = ["score", "cost", "speed"]
    units = dict(
        score="Points",
        cost="USD",
        speed="Tokens per second",
    )

    # Collect data from files
    data = dict()
    for name, short in agent_names:
        print(name)
        data[name] = agent_data = defaultdict(list)
        for run, _ in run_names:
            print(run)
            try:
                sum_data = get_summary_data(run, name)
            except IndexError:
                new_name = name
                for alias, actual in aliases.items():
                    if alias in name:
                        new_name = new_name.replace(alias, actual)
                        break
                sum_data = get_summary_data(run, new_name)
            for m in metrics:
                agent_data[m].append(sum_data[m])
                if (m_std := f"{m}_std") in sum_data:
                    agent_data[m_std].append(sum_data[m_std])

    for m in metrics:
        m_std = f"{m}_std"
        for name, short in agent_names:
            #if not name.startswith("LLMChatSession"):
            #    continue
            values = data[name][m]
            kwargs = dict()
            if m_std in data[name]:
                kwargs["yerr"] = data[name][m_std]
                kwargs["capsize"] = 4
            plt.errorbar(x_labels, values, label=short, **kwargs)
        plt.ylabel(units[m])
        plt.title(m.capitalize())
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()