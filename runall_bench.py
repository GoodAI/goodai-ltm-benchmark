from runner.run_benchmark import _main


tests = [
    # "./configurations/blogpost_tests/benchmark-1k.yml",
    "./configurations/blogpost_tests/benchmark-10k.yml",
]

models = [
    # "claude",
    "claude",
]

sizes = [
    # 200_000,
    200_000,
]

logfile = "automated_log.txt"

for t, m, s in zip(tests, models, sizes):
    try:
        _main(t, m, s, True)
        with open(logfile, "a") as f:
            f.write(f"Run {m} - {s} on {t} SUCCEEDED!\n")
    except Exception as e:
        with open(logfile, "a") as f:
            f.write(f"Run {m} - {s} on {t} FAILED!\n{e}\n\n\n")
        continue
