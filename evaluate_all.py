import os

from evaluate import _main

target_tests = ["Benchmark 3 - 1k", "Benchmark 3 - 32k", "Benchmark 3 - 32k (isolated)", "Benchmark 3 - 120k", "Benchmark 3 - 200k", "Benchmark 3 - 500k"]


for t in target_tests:
    path = f"data/tests/{t}/results"
    for agent in os.scandir(path):
        _main(t, agent.name, dataset_name="Spy Meeting", y=True)
