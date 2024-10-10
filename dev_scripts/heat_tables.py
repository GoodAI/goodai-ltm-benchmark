import matplotlib.pyplot as plt
from utils.files import gather_result_files
from utils.constants import GOODAI_RED, GOODAI_GREEN
from reporting.results import TestResult
from collections import defaultdict


LIGHT_GREY = (0.424, 0.447, 0.459)
DARK_GREY = (0.302, 0.318, 0.325)
RUN_NAMES = [
    "Benchmark 3 - 32k (isolated)",
    "Benchmark 3 - 1k",
    "Benchmark 3 - 32k",
    "Benchmark 3 - 120k",
    "Benchmark 3 - 200k",
    "Benchmark 3 - 500k",
]
TO_RETRIEVE = {
    span: [
        "LLMChatSession - together_ai-mistralai-Mixtral-8x7B-Instruct-v0.1 - 32768",
        "LLMChatSession - together_ai-mistralai-Mixtral-8x22B-Instruct-v0.1 - 65536",
        "LLMChatSession - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000",
        "LLMChatSession - gpt-3.5-turbo - 16384",
        "LLMChatSession - gpt-4-turbo-2024-04-09 - 128000",
        "LLMChatSession - gpt-4o - 128000",
        "LLMChatSession - gpt-4o-mini - 128000",
        "LLMChatSession - claude-3-opus - 200000",
        "GeminiProInterface",
        "LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - gpt-4-turbo - 16384 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - gpt-4o-mini - 16384 - QG_JSON_USER_INFO",
        "LTMAgentWrapper - claude-3-opus - 16384 - QG_JSON_USER_INFO",
        "MemGPTInterface - gpt-4o-mini - 16384",
        "MemoryBankInterface",
    ] for span in RUN_NAMES
}
ALIASES = {
    "LLMChatSession - together_ai-mistralai-Mixtral-8x7B-Instruct-v0.1 - 32768": "Mixtral 8x7B",
    "LLMChatSession - together_ai-mistralai-Mixtral-8x22B-Instruct-v0.1 - 65536": "Mixtral 8x22B",
    "LLMChatSession - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000": "Llama 3 70B",
    "LLMChatSession - gpt-3.5-turbo - 16384": "GPT-3.5 turbo",
    "LLMChatSession - gpt-4-turbo-2024-04-09 - 128000": "GPT-4 turbo",
    "LLMChatSession - gpt-4o - 128000": "GPT-4o",
    "LLMChatSession - gpt-4o-mini - 128000": "GPT-4o-mini",
    "LLMChatSession - claude-3-opus - 200000": "Claude 3 Opus",
    "GeminiProInterface": "Gemini 1.5 Pro",
    "LTMAgentWrapper - together_ai-meta-llama-Llama-3-70b-chat-hf - 8000 - QG_JSON_USER_INFO": "LTM (Llama 3 70B)",
    "LTMAgentWrapper - gpt-4-turbo - 16384 - QG_JSON_USER_INFO": "LTM (GPT-4 turbo)",
    "LTMAgentWrapper - gpt-4o-mini - 16384 - QG_JSON_USER_INFO": "LTM (GPT-4o-mini)",
    "LTMAgentWrapper - claude-3-opus - 16384 - QG_JSON_USER_INFO": "LTM (Claude 3 Opus)",
    "MemGPTInterface - gpt-4o-mini - 16384": "MemGPT (GPT-4o-mini)",
    "MemoryBankInterface": "MemoryBank (GPT-4o-mini)",
}


# Naming exceptions
def set_alt_name(old_name: str, alt_name: str, mem_spans: list[str]):
    ALIASES[alt_name] = ALIASES[old_name]
    for span in mem_spans:
        name_list = TO_RETRIEVE[f"Benchmark 3 - {span}"]
        name_list[name_list.index(old_name)] = alt_name


set_alt_name("LLMChatSession - together_ai-mistralai-Mixtral-8x7B-Instruct-v0.1 - 32768",
             "LLMChatSession - together_ai-mistralai-Mixtral-8x7B-Instruct-v0.1 - 32000",
             ["1k", "32k"])
set_alt_name("LLMChatSession - gpt-4-turbo-2024-04-09 - 128000",
             "LLMChatSession - gpt-4-turbo - 128000",
             ["120k", "200k"])
set_alt_name("LTMAgentWrapper - claude-3-opus - 16384 - QG_JSON_USER_INFO",
             "LTMAgentWrapper - claude-3-opus-20240229 - 16384 - QG_JSON_USER_INFO",
             ["120k", "200k", "500k"])
set_alt_name("LTMAgentWrapper - gpt-4-turbo - 16384 - QG_JSON_USER_INFO",
             "LTMAgentWrapper - gpt-4-turbo-2024-04-09 - 16384 - QG_JSON_USER_INFO",
             ["200k", "500k"])


def get_color(score: float) -> tuple:
    return tuple((g * score + r * (1 - score)) / 255 for g, r in zip(GOODAI_GREEN, GOODAI_RED))


def main():

    for run_name in RUN_NAMES:
        print("Showing:", run_name)
        num_repetitions = None
        rows = list()
        colours = list()
        num_expected_tasks = 10 if run_name == "Benchmark 3 - 1k" else 11
        for agent_name in TO_RETRIEVE[run_name]:
            result_paths = gather_result_files(run_name=run_name, agent_name=agent_name)
            num_results = len(result_paths)
            assert num_results >= num_expected_tasks and num_results % num_expected_tasks == 0, (
                f"Unexpected number of results: {num_results=}; {run_name=}; {agent_name=}\n{'\n'.join(result_paths)}"
            )
            results = defaultdict(lambda: dict())
            for p in result_paths:
                r = TestResult.from_file(p)
                results[r.dataset_name][r.unique_id] = r
            row = list()
            row_col = list()
            for dataset_name in sorted(results.keys()):
                if num_repetitions is None:
                    num_repetitions = len(results[dataset_name])
                assert len(results[dataset_name]) == num_repetitions,(
                    f"Found an inconsistent number of repetitions: {num_repetitions} vs {len(results[dataset_name])} in "
                    f"{run_name=}, {agent_name=}, {dataset_name=}."
                )
                for result_id in sorted(results[dataset_name].keys()):
                    r = results[dataset_name][result_id]
                    row.append(f"{r.score:.1f}")
                    row_col.append(get_color(r.score))
            rows.append(row)
            colours.append(row_col)

        # Second row are repetition labels
        num_tasks = len(rows[0]) // num_repetitions
        columns = [f"T{i + 1}" for i in range(num_repetitions)] * num_tasks
        cell_width = 0.8 / len(rows[0])
        cell_height = 1 / len(rows)
        num_rows = len(rows)
        num_cols = len(columns)

        name_aliases = {
            "Locations Directions": "Loc. Dirs",
            "Prospective Memory": "Prosp. Mems",
            "Trigger Response": "Trigger Res.",
        }

        fig, ax = plt.subplots()
        fig.set_size_inches(15, 4)
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        task_header = ax.table(
            cellText=[[""] * len(results)],
            colLabels=[name_aliases.get(k, k) for k in sorted(results.keys())],
            colColours=[DARK_GREY] * len(results),
            cellLoc="center",
            bbox=[0.2, cell_height * (num_rows - 1), cell_width * num_cols, 2 * cell_height],
        )
        for cell in task_header.properties()["children"]:
            text = cell.get_text()
            text.set_fontsize("xx-large")
            text.set_color("white")
            text.set_fontweight("black")
        ax.table(
            cellText=rows,
            cellColours=colours,
            colLabels=columns,
            colColours=[LIGHT_GREY] * len(columns),
            rowLabels=[ALIASES[agent_name] for agent_name in TO_RETRIEVE[run_name]],
            cellLoc="center",
            bbox=[0.2, 0, cell_width * num_cols, cell_height * num_rows],
        )
        plt.show()


if __name__ == "__main__":
    main()
