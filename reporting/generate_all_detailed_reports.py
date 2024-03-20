import os

from reporting.detailed_report import _main

run_name_list = ["Benchmark 2 - 1k Filler", "Benchmark 2 - 10k Filler"]


for run_name in run_name_list:
    path = f"data/tests/{run_name}/results"
    for agent in os.scandir(path):
        _main(run_name, agent.name, f"Detailed Report - {run_name} - {agent.name}")
