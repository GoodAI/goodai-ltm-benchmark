# Runner

The entrypoint for running the benchmarks is `run_benchmark.py`:
```bash
python run_benchmark.py -c ./configurations/blogpost_tests/benchmark-1k.yml \
                        -a gpt-4-1106 -m 4096
```

Use the option `-i` to run the benchmark tests without any kind of interleaving or filler messages, which is similar to how other standard benchmarks run. Run the command passing `--help` to see more options.

## Initialisation and resuming

When a run starts, the configuration file is read and the agent initialised. If the configuration file describes an existent set of test definitions, then the user will be prompted if they wish to reuse the generated tests.

If the user elects to regenerate, or if there is no set to reuse, then the relevant `DatasetInterface` classes are instantiated and their `generate_examples()` methods are called to produce the required number of `TestExample` objects. Otherwise, the `TestExample` objects will be loaded from disk.

Once the `TestExample` objects are collected, the system will perform a check for whether it can resume the test. If not, then the tests will start from scratch. If the tests are to be resumed, the master log is read and followed by the system in order to _fast-forward_ the run state to the last moment recorded in the log. Once this is done, all tests are synchronized to the saved state and the running of the tests resumes from the exact point where it was left last time.

## Scheduling

Tests are arranged in a gantt-like chart. Running tests coordinate with the scheduling system to allow it switch tasks regularly. At any given point, only compatible tests are running, which means that messages from one test will not spoil another test's execution. The system will always favour restarting waiting tests over starting new ones, but if it needs to do so a survey of the current running tests is performed and if this test would interfere with any of the running tests, then the system will choose another test to start instead. For example the tests `names` and `colours` do not interfere, as the names and favourite colours of a person are not mutually exclusive. However, `names` and `name_list` will interfere with each other, as their test examples have different names that they will supply to the agent. By default, tests interfere with themselves, which results in the system being unable to interleave two repetitions of the same test. Such repetitions will need to run sequentially.

Each test generates a `TestResult` object which is initialised when a TestExample first starts. The scheduler will pick a test and run it until the test decides to wait. A test can wait for up to two things: Tokens, and time. When a test waits, control is ceded to another test. If all tests are waiting, then the time will be forwarded, or filler tokens will be given to the agent.

The benchmark uses the [time-machine](https://pypi.org/project/time-machine/) package to spoof the current time from the perspective of the current python process. Note that this won't have effect on agents that run externally to the machine running the benchmark. In such cases, the options are to slightly alter the benchmark's code in order to either disable time jumps in favour of actual waits, or implement some form of signaling mechanism that mirrors the time jumps on the agent side.

For tests waiting on tokens, we supply filler tokens in the form of questions and answers from the [TriviaQA](https://github.com/mandarjoshi90/triviaqa) dataset. To keep reactivity up in this case of monologuing to the agent, we ask it to supply the answers in the form of a JSON list. We don't do anything with this reply, it's just to give the agent a meaningful thing to do while this filling process is ongoing. In the case of LLMs, because replies are so predictable and in order to save tokens, we add this reply directly to the LLM's context instead of having the LLM generate it. The system always provides the expected filler replies, in case the agent side decides to implement a similar approach.

## Evaluation

By default, a `TestExample` object has a script, which contains both the setup and the questions. Whenever a question is asked, the answer is logged, and once the script is finished, all of these questions and answers are evaluated.

An alternative method is used for any `TestExample` object that specifies a condition which should hold true for the rest of the conversation. See `ProspectiveMemoryDataset` for an example of this, where the agent has to append some quote to the n<sup>th</sup> reply after the question is asked. Callbacks are run after each step of a test, no matter which test it is, and they get the entire log of the conversation so far. If needed, a callback can deregister itself so that it is no longer called.

Another exception to this are the `DynamicExample` tests, which generate the script, questions and evaluation dynamically and on the fly.

Once a `TestExample` object has been evaluated, the TestResult object is updated and saved to disk.

## Logging

Every message and event during the execution of a benchmark is logged. The system keeps a global log (the one that is used for resuming), and additionally test-specific logs which are the `TestResult` objects. Logs are saved to disk after every agent's response, which ensures a safe recovery after any unexpected issue.

