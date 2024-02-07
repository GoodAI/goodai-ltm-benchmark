import os

from reporting.detailed_report import _main

target_tests = ["Benchmark 1 - 1k Filler", "Benchmark 1 - 10k Filler"]


for t in target_tests:
    path = f"data/tests/{t}/results"
    for agent in os.scandir(path):
        try:
            _main(t, agent.name, None)
        except:
            continue