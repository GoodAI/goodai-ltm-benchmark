import os

from reporting.detailed_report import _main

run_name_list = ["Benchmark 3 - 1k", "Benchmark 3 - 32k", "Benchmark 3 - 32k (isolated)", "Benchmark 3 - 120k", "Benchmark 3 - 200k", "Benchmark 3 - 500k"]

for run_name in run_name_list:
    path = f"data/tests/{run_name}/results"
    for agent in os.scandir(path):
        _main(run_name, agent.name, f"Detailed Report - {run_name} - {agent.name}", show=False)
