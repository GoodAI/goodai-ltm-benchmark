# RetrievalEvaluator Usage Guide
The RetrievalEvaluator is a tool for capturing and analyzing memory retrieval data. Here's a quick guide on how to use it in your project:
## 1. Initialization
First, import and create an instance of the RetrievalEvaluator:
```
from retrieval_evaluator.retrieval_evaluator import RetrievalEvaluator

evaluator = RetrievalEvaluator()
```

## 2. Capturing Comparison Data
During your benchmark run, capture data for each query:
```
query = "Your query here"
relevant_memories = [...]  # List of retrieved memories
filtered_memories = [...]  # List of filtered memories

evaluator.capture_comparison_data(query, relevant_memories, filtered_memories)
```
Note: You don't need to specify the benchmark version at this stage.

## 3. Generating Output
After your benchmark run is complete, generate the output for a specific benchmark version:
   - For benchmark version 4-1
   `evaluator.output("4-1")`

   - For benchmark version 2-2
   `evaluator.output("2-2")`

## 4. Output Location
The evaluation results will be saved in:

Plots and result files: retrieval_evaluator/evaluation_outputs/evaluation_{benchmark_version}_{timestamp}/

Log file: retrieval_evaluator/logs/evaluation_{benchmark_version}.log

## Important Notes

Ensure that the reference data files exist in retrieval_evaluator/dev_bench_reference_data/ directory.
The comparison data is saved in retrieval_evaluator/comparison_data/comparison_data.json.
You can run evaluations for different benchmark versions without reinitializing the evaluator.
