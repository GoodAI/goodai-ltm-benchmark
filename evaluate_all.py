import os

from evaluate import _main

target_tests = ["Benchmark 1 - 1k Filler", "Benchmark 1 - 10k Filler"]


for t in target_tests:
    path = f"data/tests/{t}/results"
    for agent in os.scandir(path):
        _main(t, agent.name, dataset_name="*", y=True)
