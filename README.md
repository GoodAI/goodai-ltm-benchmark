# GoodAI LTM Benchmark (v3.5)

![GoodAI Logo. A cybernetic owl, which is half robot, half organic, and next to it the company name: GoodAI](reporting/templates/GoodAI_logo.png "GoodAI Research s.r.o.")

This repository contains the code and data to replicate our experiments regarding the Long-Term Memory (LTM) abilities of conversational agents. This is the 2<sup>nd</sup> version of our LTM Benchmark. Check out [our blogpost](https://www.goodai.com/introducing-goodai-ltm-benchmark/) for more information about the benchmark and the related research goals.

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

For example, to run the 1k blogpost benchmark on the GPT4-turbo-preview LLM with a context size of 4096 tokens:

```bash
python run_benchmark.py -c ./configurations/published_benchmarks/benchmark-v3-500k.yml \
                        -a gpt-4-turbo -m 4096
```

This will generate a set of test specifications if there is not one already, and start to produce result files, one for each test. The result files will be located at `./tests/Benchmark 1 - 1k Filler/results/GPTChatSession - gpt-4-1106-preview - 4096/`.

At the end of testing, an HTML report will be generated in `data/reports` which will give a detailed breakdown of the tests run, responses, and evaluations. It will be given a name of the form `<time stamp> - Detailed Report - <run_name> - <agent_name>.html`.

## Agents

The agents that have been specifically implemented in this repository are the ones shown below. For implementing your own agent, please see the more detailed instructions [here](model_interfaces/README.md).

```text
# OpenAI models
gpt-3.5-turbo       # GPT3.5
gpt-4-turbo         # latest GPT4-turbo
gpt-4o              # latest GPT4o

# Anthropic Models (200k context)
claude-2.1          # Claude 2.1
claude-3-haiku      # Claude 3 Haiku
claude-3-sonnet     # Claude 3 Sonnet
claude-3-opus       # Claude 3 Opus

# Google Gemini  (1.5M-2M context)
gemini              # Gemini 1.5 Pro

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

In addition, we also support LLMs that are supported by [litellm](https://www.litellm.ai/). To use external providers through litellm (e.g for [together.ai](https://www.together.ai/)) you should set your api key in either a `.env` file or as an environment variable:
```bash
export TOGETHERAI_API_KEY=sk-...
```
And then call your agent in the form `<api>/<author>/<model>`. For example:
```bash
python run_benchmark.py -c ./configurations/published_benchmarks/benchmark-v3-500k.yml \
                        -a together_ai/meta-llama/Llama-3-70b-chat-hf -m 8000
```


## Configurations

The configuration files used in the different versions of the benchmark can be found in `configurations/published_benchmarks`, in which `<x>k` denotes the memory span in thousands of tokens. For each of the benchmarks under a single version, we keep the scripts and needles the same, but we increase the amount of filler tokens owing to the larger memory span. Older configurations from previous releases can be found in `published_benchmarks/legacy`. These configuration files are compatible only with their corresponding releases and their operation is described in the readmes for those releases.


## Datasets

The datasets that are implemented for this benchmark can be found in `./datasets/`. Briefly, they are:

```
chapterbreak
colours
jokes
locations_directions
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


## Benchmark 3.5 - 06/2024

### Benchmark 3 - Isolated (No Memory Span)

| Model                      | Context Tokens | Score / 11 | Time (m) | Cost ($) |
|----------------------------|---------------:|-----------:|---------:|---------:|
| Mixtral-8x7B Instruct 0.1  |          32768 |          5 |    10.25 |     0.15 | 
| Mixtral-8x22B Instruct 0.1 |          65536 |        4.9 |       11 |     0.61 |
| Llama 3 70B Instruct       |           8000 |        8.2 |      8.8 |     0.13 | 
| GPT-3.5-turbo              |          16384 |        4.1 |        6 |     0.13 | 
| GPT-4 Turbo                |         128000 |        7.9 |     18.5 |     6.94 | 
| GPT-4o                     |         128000 |        7.6 |        8 |     3.08 | 
| Claude 3 Opus              |         200000 |        8.3 |       41 |    15.28 | 
| Gemini 1.5 Pro             |        2000000 |        7.4 |       58 |      --- |
| LTMAgent 1 (Llama 3 70B)   |           8000 |        8.4 |       26 |     0.65 |
| LTMAgent 1 (GPT-4-turbo)   |          16384 |        9.2 |     68.3 |     9.81 |
| LTMAgent 1 (Claude)        |          16384 |        8.7 |     99.5 |     0.52 | 


### Benchmark 3 - 2k Memory Span (Without ChapterBreak)

| Model                      | Context Tokens | Score / 10 | Time (m) | Cost ($) |
|----------------------------|---------------:|-----------:|---------:|---------:|
| Mixtral-8x7B Instruct 0.1  |          32768 |        1.4 |      7.5 |     0.08 | 
| Mixtral-8x22B Instruct 0.1 |          65536 |        5.6 |     97.2 |     0.93 |
| Llama 3 70B Instruct       |           8000 |        1.9 |      4.5 |     0.08 | 
| GPT-3.5-turbo              |          16384 |        4.7 |      8.1 |     0.31 | 
| GPT-4 Turbo                |         128000 |        6.6 |      5.5 |     8.29 | 
| GPT-4o                     |         128000 |        5.9 |      4.8 |     4.55 | 
| Claude 3 Opus              |         200000 |        7.8 |     41.8 |    19.19 | 
| Gemini 1.5 Pro             |        2000000 |        6.5 |       55 |      --- |
| LTMAgent 1 (Llama 3 70B)   |           8000 |        6.9 |     22.9 |      1.2 |
| LTMAgent 1 (GPT-4-turbo)   |          16384 |        6.3 |       99 |    17.34 |
| LTMAgent 1 (Claude)        |          16384 |        7.5 |     90.8 |     0.38 | 



### Benchmark 3 - 32k Memory Span

| Model                      | Context Tokens | Score / 11 | Time (m) | Cost ($) |
|----------------------------|---------------:|-----------:|---------:|---------:|
| Mixtral-8x7B Instruct 0.1  |          32768 |        0.1 |        9 |     0.06 | 
| Mixtral-8x22B Instruct 0.1 |          65536 |        0.0 |       18 |     0.93 |
| Llama 3 70B Instruct       |           8000 |        0.2 |     10.8 |     0.06 | 
| GPT-3.5-turbo              |          16384 |        0.1 |      5.5 |     0.06 | 
| GPT-4 Turbo                |         128000 |        4.8 |     18.5 |    77.74 | 
| GPT-4o                     |         128000 |        4.6 |       15 |    38.38 | 
| Claude 3 Opus              |         200000 |        6.7 |    133.5 |   215.42 | 
| Gemini 1.5 Pro             |        2000000 |        6.4 |       39 |      --- |
| LTMAgent 1 (Llama 3 70B)   |           8000 |          5 |     43.7 |     2.50 |
| LTMAgent 1 (GPT-4-turbo)   |          16384 |        5.2 |    171.9 |    61.46 |
| LTMAgent 1 (Claude)        |          16384 |          5 |    173.2 |     0.68 | 


### Benchmark 3 - 120k Memory Span

| Model                      | Context Tokens | Score / 11 | Time (m) | Cost ($) |
|----------------------------|---------------:|-----------:|---------:|---------:|
| Mixtral-8x7B Instruct 0.1  |          32768 |        0.1 |      7.7 |     0.06 | 
| Mixtral-8x22B Instruct 0.1 |          65536 |        0.1 |     21.1 |     1.12 |
| Llama 3 70B Instruct       |           8000 |        0.2 |      9.4 |     0.06 | 
| GPT-3.5-turbo              |          16384 |        0.0 |        6 |     1.33 | 
| GPT-4 Turbo                |         128000 |        5.8 |       49 |   215.86 | 
| GPT-4o                     |         128000 |        5.5 |       32 |   108.22 | 
| Claude 3 Opus              |         200000 |        7.4 |      519 |   476.68 | 
| Gemini 1.5 Pro             |        2000000 |        7.0 |      --- |      --- |
| LTMAgent 1 (Llama 3 70B)   |           8000 |        4.7 |     86.5 |     3.10 |
| LTMAgent 1 (GPT-4-turbo)   |          16384 |        5.0 |    567.5 |    89.36 |
| LTMAgent 1 (Claude)        |          16384 |        5.7 |    307.5 |   158.24 | 


### Benchmark 3 - 200k Memory Span

| Model                      | Context Tokens | Score / 11 | Time (m) | Cost ($) |
|----------------------------|---------------:|-----------:|---------:|---------:|
| Mixtral-8x7B Instruct 0.1  |          32768 |        0.1 |      8.7 |     0.04 | 
| Mixtral-8x22B Instruct 0.1 |          65536 |        0.1 |     14.5 |     1.21 |
| Llama 3 70B Instruct       |           8000 |        0.2 |      8.0 |     0.06 | 
| GPT-3.5-turbo              |          16384 |        0.0 |      5.0 |     0.06 | 
| GPT-4 Turbo                |         128000 |        3.9 |    45.17 |   222.62 | 
| GPT-4o                     |         128000 |        5.2 |    35.75 |   111.80 | 
| Claude 3 Opus              |         200000 |        5.4 |   338.43 |   502.28 | 
| Gemini 1.5 Pro             |        2000000 |        8.0 |       76 |      --- |
| LTMAgent 1 (Llama 3 70B)   |           8000 |        5.6 |   126.87 |     3.89 |
| LTMAgent 1 (GPT-4-turbo)   |          16384 |        5.3 |   326.22 |    87.78 |
| LTMAgent 1 (Claude)        |          16384 |        6.4 |   342.83 |   149.53 | 


### Benchmark 3 - 500k Memory Span

| Model                      | Context Tokens | Score / 11 | Time (m) | Cost ($) |
|----------------------------|---------------:|-----------:|---------:|---------:|
| Mixtral-8x7B Instruct 0.1  |          32768 |        0.1 |      8.5 |     0.07 | 
| Mixtral-8x22B Instruct 0.1 |          65536 |        0.1 |       44 |     1.15 |
| Llama 3 70B Instruct       |           8000 |        0.2 |     11.5 |     0.06 | 
| GPT-3.5-turbo              |          16384 |        0.0 |      6.5 |     0.06 | 
| GPT-4 Turbo                |         128000 |        1.0 |       48 |   223.16 | 
| GPT-4o                     |         128000 |        0.9 |       38 |   111.49 | 
| Claude 3 Opus              |         200000 |        3.4 |   324.35 |   527.86 | 
| Gemini 1.5 Pro             |        2000000 |        5.3 |     82.5 |      --- |
| LTMAgent 1 (Llama 3 70B)   |           8000 |        4.8 |   250.23 |     6.13 |
| LTMAgent 1 (GPT-4-turbo)   |          16384 |        3.1 |  1240.30 |   174.93 |
| LTMAgent 1 (Claude)        |          16384 |        4.9 |   528.37 |   230.27 | 

## Previous versions

- [Benchmark 1](https://github.com/GoodAI/goodai-ltm-benchmark/tree/v1-benchmark) (02/2024)
- [Benchmark 2](https://github.com/GoodAI/goodai-ltm-benchmark/tree/v2-benchmark) (03/2024)
- [Benchmark 3](https://github.com/GoodAI/goodai-ltm-benchmark/tree/v3-benchmark) (04/2024)

## Licence and usage
This project is licensed under the MIT License - see the LICENSE file for details. Use of this software requires attribution to the original author and project, as detailed in the license.

Some datasets use data generated by GPT, so those specific tests are unsuitable for commercial purposes.



## Acknowledgements
* The filler is drawn from the [TriviaQA dataset](https://github.com/mandarjoshi90/triviaqa) which is licenced under Apache 2.0.
* The data for the SallyAnne dataset (labelled `data/tomi_data/`) was generated using [this code](https://github.com/kayburns/tom-qa-dataset) implementing the paper [Evaluating Theory of Mind in Question Answering](https://arxiv.org/abs/1808.09352), which is currently (as of 22/01/2024) unlicenced.
* The ChapterBreak dataset is described in the paper [ChapterBreak: A Challenge Dataset for Long-Range Language Models](https://arxiv.org/abs/2204.10878) and the repository is found on [GitHub](https://github.com/SimengSun/ChapterBreak). ChapterBreak is licenced under Apache 2.0.
* "The Complete Works of William Shakespeare" is public domain. This particular copy has been sourced from [Project Gutenburg](https://www.gutenberg.org/), whose terms of use can be found on their website.   
