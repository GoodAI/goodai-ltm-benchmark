# GoodAI LTM Benchmark

![GoodAI Logo. A cybernetic owl, which is half robot, half organic, and next to it the company name: GoodAI](logo.png "GoodAI Research s.r.o.")

This repository contains the code and data to supplement [our blogpost](https://www.goodai.com/introducing-goodai-ltm-benchmark/).

As part of our research efforts in the area of continual learning, we are open-sourcing this benchmark for testing agents’ ability to perform tasks involving the advanced use of the memory over very long conversations. Among others, we evaluate the agent’s performance on tasks that require dynamic upkeep of memories or integration of information over long periods of time.

We are open-sourcing:
 * The living GoodAI LTM Benchmark (this repository).
 * Our [LTM agents](model_interfaces/).
 * Our experiment data and results

This benchmark has demonstrated that our **LTM agents with 8k context are comparable to long context GPT-4-1106 with 128k**
tokens when recalling and correctly using information in short form conversational contexts. In a longer benchmark, our agents
outperform long context GPT by **13%** for **16% of the running costs.** See the [Benchmark section](#benchmark-1---022024) for the scores.

## Running the Benchmarks

These tests require python 3.10 or higher.

First, set your `OPENAI_API_KEY`, and optionally `ANTHROPIC_API_KEY` environment variables and clone the repository: 
```bash
git clone git@github.com:GoodAI/goodai-ltm-benchmark.git
```

The file `run_benchmark.py` can be executed by giving it a configuration `.yml` file using `-c` (examples are located in `./configurations/`), an agent using `-a` (see below), and optionally a limit for the size of the context with `-m`.

For example, to run the 1k blogpost benchmark on the GPT4-turbo-preview LLM with a context size of 4096 tokens:

```bash
python run_benchmark.py -c ./configurations/blogpost_tests/benchmark-1k.yml \
                        -a gpt-4-1106 -m 4096
```

This will generate a set of test specifications if there is not one already, and start to produce result files, one for each test. The result files will be located at `./tests/Benchmark 1 - 1k Filler/results/GPTChatSession - gpt-4-1106-preview - 4096/`.

At the end of testing, an HTML report will be generated in the main directory which will give a detailed breakdown of the tests run, responses, and evaluations. It will be given a name of the form `<time stamp> - Detailed Report - <run_name> - <agent_name>.html`.

## Agents

The agents currently implemented in this repository are the ones shown below. For implementing your own agent, please see the more detailed instructions [here](model_interfaces/README.md).

```text
# OpenAI models
gpt/gpt-4           # GPT4
gpt-3.5-turbo       # GPT3.5
gpt-4-1106          # GPT4-turbo preview
ts-gpt-3.5-turbo    # GPT3.5 with timestamped messages
ts-gpt-4-1106       # GPT4-turbo preview with timestamped messages

# Anthopic Models
claude              #  Claude-2.1 200k context model

# Langchain Models
langchain_sb_a    # Using 3.5-turbo-instruct and a summary buffer memory
langchain_kg_a    # Using 3.5-turbo-instruct and a knowledge graph memory
langchain_ce_a    # Using 3.5-turbo-instruct and a conversation entity memory

# GoodAI LTM models
LTMAgent1             # GoodAI LTM, 4-turbo preview, semantic retrieval + query generation + JSON scratchpad
LTMAgent2             # GoodAI LTM, 4-turbo preview, semantic retrieval

# Memgpt
memgpt            # An actively managed LTM/RAG conversational agent


# Cost Estimation
cost(<cost_in_tokens>,<cost_out_tokens>) # Model for estimating the cost of a 
                                           benchmark based on the input and output
                                           costs

# Human models
human             # A CLI interface for a human to use the tests.
```

## Configurations

The configuration used in the blogpost benchmark can be found in `./configurations/blogpost_tests/benchmark-1k.yml`, in which `1k` refers to the information gap between relevant statements. For the `10k` benchmark, we used the very same test definitions as for the `1k` benchmark, but we increased the amount of filler tokens directly in the test definition files. This way we ensured that the length of the information gap is the only thing that changes between both benchmarks.


## Datasets

The datasets that are implemented for this benchmark can be found in `./datasets/`. Briefly, they are:

```
chapterbreak
colours
conflicting_personal_info
delayed_recall
how_to_think
instruction_recall
jokes
kv
locations
locations_directions
names
name_list
prospective_memory
sallyanne
shopping
trigger_response
```

More details for each of the tests can be found from their descriptions inside each of their individual files.

## Technical Details

The repository consists of four parts:

- **Datasets:** These are test generators, either through random combination of words, phrases, and numbers, sampling lines from an existent dataset, or generating them via a prompted GPT.
- **Models:** A model is an agent that can be set to perform the tasks of the dataset. This part presents a very simple interface and facilitates the integration of agents with the benchmark.
- **Runner:** This script takes a configuration and model specification, optionally generates the set of test instances, and executes the benchmark.
- **Reports:** These files generate the reports as self-contained HTML files, with support for individual and comparative reporting. 

More details for each of these parts can be found here: [datasets](datasets/README.md), [models](model_interfaces/README.md), [runner](runner/README.md), [reports](reporting/README.md).


## Benchmark 1 - 02/2024
### Benchmark 1 - 1k Distraction tokens

| Model  | Context Tokens | Score / 92 | Time (m) | Cost ($) | Mean Memory Span |
|--------|----------------|------------|----------|----------| ---------------- |
| LTMAgent1  | 4096           | 85         | 153      | 14.82    | 6579 |
| LTMAgent1  | 8192           | 89         | 148.5    | 19.14    | 7253 |
| LTMAgent2  | 8192           | 86         | 31       | 14.31    | 7347 |
| MemGPT | 4096           | 7          | 150      | 81.24    | 5990 |
| MemGPT | 8192           | 44         | 103.3    | 91.69    | 6767 |
| Claude-2.1 | 200000         | 74         | 57.3     | 11.78 | 7291 |
| GPT-4-1106 | 4096           | 49         | 44       | 8.80     | 7344 | 
| GPT-4-1106 | 8192           | 77         | 34.7     | 13.85    | 7344 |
| GPT-4-1106 | 128000         | 82         | 42.56    | 15.99    | 7283 |


### Benchmark 1 - 10k Distraction Tokens

| Model      | Context Tokens | Score / 92 | Time (m) | Cost ($) | Mean Memory Span |
|------------|----------------|------------|----------|----------|------------------|
| LTMAgent1  | 8192           | 86         | 529      | 61.38    | 57095            |
| LTMAgent2  | 8192           | 85         | 117      | 38.98    | 57231            |
 | Claude-2.1 | 200000 | 42         | 346      | 227      | 60488            | 
| GPT-4-1106 | 8192           | 11         | 90.2     | 37.38    | 58060            | 
| GPT-4-1106 | 128000         | 76         | 154.58   | 255.30   | 57343            |

## Licence and usage
This code is licenced under MIT. Some datasets use data generated by GPT, so those specific tests are unsuitable for commercial purposes.

## Acknowledgements
* The filler is drawn from the [TriviaQA dataset](https://github.com/mandarjoshi90/triviaqa) which is licenced under Apache 2.0.
* The data for the SallyAnne dataset (labelled `data/tomi_data/`) was generated using [this code](https://github.com/kayburns/tom-qa-dataset) implementing the paper [Evaluating Theory of Mind in Question Answering](https://arxiv.org/abs/1808.09352), which is currently (as of 22/01/2024) unlicenced.
* The ChapterBreak dataset is described in the paper [ChapterBreak: A Challenge Dataset for Long-Range Language Models](https://arxiv.org/abs/2204.10878) and the repository is found on [GitHub](https://github.com/SimengSun/ChapterBreak). ChapterBreak is licenced under Apache 2.0.
* "The Complete Works of William Shakespeare" is public domain. This particular copy has been sourced from [Project Gutenburg](https://www.gutenberg.org/), whose terms of use can be found on their website.   
