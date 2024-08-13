from collections import defaultdict
from utils.files import gather_result_files, parse_result_path
from utils.math import mean_std
from reporting.results import TestResult


def wilson_score(values) -> tuple[float, float]:
    p = sum(values) / len(values)
    std = ((p * (1 - p)) / len(values)) ** 0.5
    return p, std


def main():
    binomials = {"ChapterBreak", "Colours", "Jokes", "Locations Directions", "Prospective Memory", "SallyAnne"}
    runs_agents = [
        ("Benchmark 3 - 32k", "LLMChatSession - gpt-4o - 128000"),
        ("Benchmark 3 - 32k #2", "LLMChatSession - gpt-4o-2024-05-13 - 128000"),
    ]
    results = defaultdict(list)
    for run_name, agent_name in runs_agents:
        files = gather_result_files(run_name, agent_name)
        for res_file in files:
            meta = parse_result_path(res_file)
            result = TestResult.from_file(res_file)
            assert 0 <= result.score <= 1
            results[meta["dataset_name"]].append(result.score)

    total_avg = total_std = 0
    for dataset_name, dataset_results in results.items():
        if dataset_name in binomials:
            avg, std = wilson_score(dataset_results)
        else:
            avg, std = mean_std(dataset_results)
        total_avg += avg
        total_std += std

    print("Avg:", total_avg)
    print("Std:", total_std)


if __name__ == "__main__":
    main()
