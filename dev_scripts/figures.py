import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from reporting.generate import get_summary_data
from runner.master_log import MasterLog
from utils.constants import DATA_DIR, EventType
from utils.llm import tokens_in_text
from utils.math import mean_std
import numpy as np


def message_tokens(message: str, agt_name: str) -> int:
    model = agt_name.split(" - ")[1]
    if model.startswith("gpt-4"):
        model = "gpt-4-turbo-2024-04-09"
    elif model.startswith("claude-3"):
        model = "claude-3-opus-20240229"
    else:
        model = "together_ai/meta-llama-Llama-3-70b-chat-hf"
    return tokens_in_text(message, model)


def speed_stats(run: str, agt_name: str) -> tuple[float, float]:
    log = MasterLog(DATA_DIR.joinpath("tests", run, "results", agt_name, "master_log.jsonl"))
    log.load()
    speeds = list()
    total_tokens = 0
    total_time = 0
    for evt in log.test_events(event_type={EventType.SEND_MESSAGE, EventType.RESPONSE_MESSAGE}):
        if evt.type == EventType.SEND_MESSAGE:
            t0 = evt.timestamp
        else:
            msg = evt.data["message"]
            t = (evt.timestamp - t0).total_seconds()
            tokens = message_tokens(msg, agt_name)
            speeds.append(tokens / t)
            total_tokens += tokens
            total_time += t
    # return mean_std(speeds)
    return total_tokens / total_time, 0


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
        ("LTMAgentWrapper - gpt-4-turbo - 16384 - SEMANTIC_ONLY", "A2(GPT4)"),
        ("LTMAgentWrapper - claude-3-opus - 16384 - QG_JSON_USER_INFO", "A1(Claude)"),
        ("LTMAgentWrapper - claude-3-opus - 16384 - SEMANTIC_ONLY", "A2(Claude)"),
        ("LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - QG_JSON_USER_INFO", "A1(Llama)"),
        ("LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - SEMANTIC_ONLY", "A2(Llama)"),
    ]
    sorted_tab_colours = list(mcolors.TABLEAU_COLORS.keys())
    agent_names = [(name, short, sorted_tab_colours[i]) for i, (name, short) in enumerate(agent_names)]
    metrics = ["score", "cost", "speed"]
    units = dict(
        score="Points",
        cost="USD",
        speed="Tokens per second",
    )

    # Collect data from files
    data = dict()
    for name, short, col in agent_names:
        print(name)
        data[name] = agent_data = defaultdict(list)
        for run, _ in run_names:
            print(run)
            agt_name = name
            try:
                sum_data = get_summary_data(run, agt_name)
            except IndexError:
                for alias, actual in aliases.items():
                    if alias in name:
                        agt_name = agt_name.replace(alias, actual)
                        break
                sum_data = get_summary_data(run, agt_name)
            # sum_data["speed"], sum_data["speed_std"] = speed_stats(run, agt_name)
            sum_data["speed"], _ = speed_stats(run, agt_name)
            for m in metrics:
                agent_data[m].append(sum_data[m])
                if (m_std := f"{m}_std") in sum_data:
                    agent_data[m_std].append(sum_data[m_std])

    for m in metrics:
        m_std = f"{m}_std"
        for name, short, col in agent_names:
            # if not name.startswith("LLMChatSession"):
            #     continue
            # if not short.startswith("A1("):
            #     continue
            values = np.array(data[name][m])
            plt.plot(x_labels, values, c=col, label=short)
            # if m_std in data[name]:
            #     y_err = np.array(data[name][m_std])
            #     plt.fill_between(x_labels, values - y_err, values + y_err, color=col, alpha=0.1)
        plt.ylabel(units[m])
        plt.title(m.capitalize())
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
