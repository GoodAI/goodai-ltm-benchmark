# Runner

The entrypoint for running the benchmarks is `run_benchmark.py`:
```bash
python run_benchmark.py -c ./configurations/blogpost_tests/benchmark-1k.yml \
                        -a gpt-4-1106 -m 4096
```

## Initialisation

When a run starts, the configuration file is read and the agent initialised. If the configuration file describes an existent set of test definitions, then the user will be prompted if they wish to reuse the generated tests.

If the user elects to regenerate, or if there is no set to reuse, then the relevant `DatasetInterface` classes are instantiated and their `generate_examples()` methods are called to produce the required number of `TestExample` objects.


## Scheduling

Once the `TestExample` objects are collected, the scheduler creates groups of tests. A test group is a collection of non-interfering tests, for example the tests `names` and `colours` do not interfere, as the names and favourite colours of a person are not mutually exclusive. However, `names` and `name_list` will interfere with each other, as their test examples have different names that they will supply to the agent. By default, tests interfere with themselves, so a typical group will take one `TestExample` object from each dataset.

Each test generates a `TestResult` object which is initialised when a TestExample first starts. When running a group, the scheduler will pick a test and run it until the test decides to wait. A test can wait for up to two things: Tokens, and time. When a test waits, control is ceded to another test. If all tests are waiting, then the time will be forwarded, or filler tokens will be given to the agent.

The benchmark uses the [time-machine](https://pypi.org/project/time-machine/) package to spoof the current time from the perspective of the current python process. 

For tests waiting on tokens, we supply filler tokens in the form of questions and answers from the [TriviaQA](https://github.com/mandarjoshi90/triviaqa) dataset. To keep reactivity up in this case of monologuing to the agent, we ask it to supply the answers in the form of a JSON list. We don't do anything with this reply, it's just to give the agent a meaningful thing to do while this filling process is ongoing. 

## Evaluation

A `TestExample` object has a script, which contains both the setup and the questions. Whenever a question is asked, the answer is logged, and once the script is finished, all of these questions and answers are evaluated.

An alternative method is used for any `TestExample` object that specifies a condition which should hold true for the rest of the conversation. See `ProspectiveMemoryDataset` for an example of this, where the agent has to append some quote to the n<sup>th</sup> reply after the question is asked. Callbacks are run after each step of a test, no matter which test it is, and they get the entire log of the conversation so far. If needed, a callback can deregister itself so that it is no longer called. 

Once a `TestExample` object has been evaluated, the TestResult object is updated and saved. When all the tests in a group have finished, the agent is reset to clear both the context and current memory, and testing of the next group starts. 
