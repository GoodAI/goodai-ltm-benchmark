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
relevant_memories = [(Message(User), Message(Agent)), (Message(User), Message(Agent)), ...]  # List of retrieved memories
filtered_memories = [(Message(User), Message(Agent)), (Message(User), Message(Agent)), ...]  # List of filtered memories

evaluator.capture_comparison_data(user_message, relevant_memories, filtered_memories)
```
Note: You don't need to specify the benchmark version at this stage.

## 3. Generating Output
After your benchmark run is complete, generate the output for a specific benchmark version:
   `evaluator.output("Dev Benchmark 1 32k - 2 Examples")`

## 4. Output Location
The evaluation results will be saved in:

Plots and result files: data/retrieval_evaluator/evaluation_outputs/evaluation_{benchmark_version}_{timestamp}/

Log file: data/retrieval_evaluator/logs/evaluation_{benchmark_version}.log

## Important Notes

Ensure that the reference data files exist in data/retrieval_evaluator/dev_bench_reference_data/ directory.
The comparison data is saved in data/retrieval_evaluator/comparison_data/comparison_data.json.
You can run evaluations for different benchmark versions without reinitializing the evaluator.
