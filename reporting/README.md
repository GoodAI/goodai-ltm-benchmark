# Reports

These files here generate the reports for the benchmarks. There are two report generators:
* `detailed_report.py` Which generates a detailed list of the tests that an agent has performed in a benchmark run.
* `comparative_report.py` Which generates a report comparing multiple agents in a benchmark.

## Detailed Reports

The detailed report is run automatically on the end of a benchmark run, but if you wish to run the report generation manually use:
```bash
python detailed_report.py <benchmark_folder> <agent_folder> -o <output_name>
```

As an example:
```bash
python detailed_report.py "Benchmark 1 - 1k Filler" \
                          "GPTChatSession - gpt-4-1106-preview - 8192" \
                          -o output
```
Where `GPTChatSession - gpt-4-1106-preview - 8192` is a folder inside `Benchmark 1 - 1k Filler/results/`. The folder name corresponds to the unique agent ID, which is expected to include any setting value that is subject to change.

## Comparative Reports

Comparative reports contrast multiple agents against each other.
```bash
python comparative_report.py <benchmark_folder>
```
You will be prompted to select which runs should be compared, and which labels you want to assign to the agents in the report.
