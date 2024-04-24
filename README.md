# GoodAI LTM Benchmark (v2)

![GoodAI Logo. A cybernetic owl, which is half robot, half organic, and next to it the company name: GoodAI](reporting/templates/GoodAI_logo.png "GoodAI Research s.r.o.")

This repository contains the code and data to replicate our experiments regarding the Long-Term Memory (LTM) abilities of conversational agents. The benchmark was originally released jointly with [a blogpost](https://www.goodai.com/introducing-goodai-ltm-benchmark/), check it out for obtaining more information about the benchmark and the related research goals.

As part of our research efforts in the area of continual learning, we are open-sourcing this benchmark for testing agents’ ability to perform tasks involving the advanced use of the memory over very long conversations. Among others, we evaluate the agent’s performance on tasks that require dynamic upkeep of memories or integration of information over long periods of time.

We are open-sourcing:
 * The living GoodAI LTM Benchmark (this repository).
 * Our [LTM agents](model_interfaces/).
 * Our experiment [data](data/tests/Benchmark%202%20-%2010k%20Filler/definitions) and [results](data/tests/Benchmark%202%20-%2010k%20Filler/results).

## Running the Benchmarks

These tests require python 3.10 or higher.

First, set your `OPENAI_API_KEY`, and optionally `ANTHROPIC_API_KEY` environment variables and clone the repository: 
```bash
git clone git@github.com:GoodAI/goodai-ltm-benchmark.git
```

The file `run_benchmark.py` can be executed by giving it a configuration `.yml` file using `-c` (examples are located in `./configurations/`), an agent using `-a` (see below), and optionally a limit for the size of the context with `-m`.

For example, to run a benchmark on the GPT4-turbo LLM with a context size of 4096 tokens:

```bash
python run_benchmark.py -c ./configurations/<configuration_name>.yml \
                        -a gpt-4-turbo -m 4096
```

This will generate a set of test specifications if there is not one already, and start to produce result files, one for each test. The result files will be located at `./tests/<benchmark_name>/results/<agent_name>/`.

At the end of testing, an HTML report will be generated in `data/reports` which will give a detailed breakdown of the tests run, responses, and evaluations. It will be given a name of the form `<time stamp> - Detailed Report - <run_name> - <agent_name>.html`.

## Agents

The agents currently implemented in this repository are the ones shown below. For implementing your own agent, please see the more detailed instructions [here](model_interfaces/README.md).

```text
# OpenAI models
gpt-3.5-turbo       # GPT3.5
gpt-4-1106          # GPT4-turbo preview
gpt-4-turbo         # GPT4-turbo-2024-04-09

# Anthropic Models (200k context)
claude-2.1          # Claude 2.1
claude-3-haiku      # Claude 3 Haiku
claude-3-sonnet     # Claude 3 Sonnet
claude-3-opus       # Claude 3 Opus

# Models with timestamped messages
ts-<model>          # Any of the above OpenAI or Anthropic models 

# GoodAI LTM models
# Variants:
#   1. semantic retrieval + query generation + JSON scratchpad
#   2. semantic retrieval
#   3. semantic retrieval + text scratchpad
# Optional model ID to use as core LLM
# Example: ltm_agent_1(claude-3-opus-20240229)
ltm_agent_<variant>[(<model>)]

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

The configuration files used in the different versions of the benchmark can be found in `configurations/published_benchmarks`, in which `<x>k` denotes the memory span in thousands of tokens. For each of the benchmarks under a single version, we keep the scripts and needles the same, but we increase the amount of filler tokens owing to the larger memory span. Older configurations from previous releases can be found in `published_benchmarks/legacy`. These configuration files are compatible only with their corresponding releases and their operation is described in the readmes for those releases.


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
restaurant
sallyanne
shopping
spy_meeting
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


## Benchmark 3 - 04/2024

### Benchmark 3 - 120k Memory Span

| Model               | Context Tokens | Score / 11 | Time (m) | Cost ($) | LTM Score (tokens) |
|---------------------|---------------:|-----------:|---------:|---------:|-------------------:|
| GPT-3.5-turbo       |          16384 |          0 |        7 |     1.33 |                  0 |
| Claude 3 Opus       |         200000 |        7.7 |      518 |   476.00 |             869014 |
| GPT-4 2024-04-09    |         128000 |        5.1 |       51 |   215.86 |            1081773 |
| LTMAgent 1 (GPT-4)  |          16384 |        5.1 |      567 |    89.36 |             548641 |
| LTMAgent 2 (GPT-4)  |          16384 |        4.6 |       85 |    62.50 |             420210 |
| LTMAgent 1 (Claude) |          16384 |        5.2 |      308 |   158.24 |             548641 |


### Benchmark 3 - 200k Memory Span

| Model               | Context Tokens | Score / 11 | Time (m) | Cost ($) | LTM Score (tokens) |
|---------------------|---------------:|-----------:|---------:|---------:|-------------------:|
| Claude 3 Opus       |         200000 |        4.9 |      338 |   502.00 |             866157 |
| GPT-4 2024-04-09    |         128000 |        3.9 |       47 |   222.62 |             635842 |
| LTMAgent 1 (GPT-4)  |          16384 |        5.2 |      326 |    87.78 |             868573 |
| LTMAgent 2 (GPT-4)  |          16384 |        5.6 |    125.8 |    71.33 |             948850 |
| LTMAgent 1 (Claude) |          16384 |          6 |    342.8 |   149.53 |            1000353 |


### Benchmark 3 - 500k Memory Span

| Model               | Context Tokens | Score / 11 | Time (m) | Cost ($) | LTM Score (tokens) |
|---------------------|---------------:|-----------:|---------:|---------:|-------------------:|
| Claude 3 Opus       |         200000 |        3.3 |    324.3 |      527 |            1421359 |
| GPT-4 2024-04-09    |         128000 |          1 |     48.7 |   223.16 |             263189 |
| LTMAgent 1 (GPT-4)  |          16384 |        3.1 |     1240 |   174.93 |            1393652 |
| LTMAgent 2 (GPT-4)  |          16384 |        4.3 |      307 |   106.39 |            1785526 |
| LTMAgent 1 (Claude) |          16384 |        4.9 |    523.3 |   230.27 |            2041299 |

## Previous versions

- [Benchmark 1](https://github.com/GoodAI/goodai-ltm-benchmark/tree/v1-benchmark) (02/2024)
- [Benchmark 2](https://github.com/GoodAI/goodai-ltm-benchmark/tree/v2-benchmark) (03/2024)

## Licence and usage
This project is licensed under the MIT License - see the LICENSE file for details. Use of this software requires attribution to the original author and project, as detailed in the license.

Some datasets use data generated by GPT, so those specific tests are unsuitable for commercial purposes.



## Acknowledgements
* The filler is drawn from the [TriviaQA dataset](https://github.com/mandarjoshi90/triviaqa) which is licenced under Apache 2.0.
* The data for the SallyAnne dataset (labelled `data/tomi_data/`) was generated using [this code](https://github.com/kayburns/tom-qa-dataset) implementing the paper [Evaluating Theory of Mind in Question Answering](https://arxiv.org/abs/1808.09352), which is currently (as of 22/01/2024) unlicenced.
* The ChapterBreak dataset is described in the paper [ChapterBreak: A Challenge Dataset for Long-Range Language Models](https://arxiv.org/abs/2204.10878) and the repository is found on [GitHub](https://github.com/SimengSun/ChapterBreak). ChapterBreak is licenced under Apache 2.0.
* "The Complete Works of William Shakespeare" is public domain. This particular copy has been sourced from [Project Gutenburg](https://www.gutenberg.org/), whose terms of use can be found on their website.   
