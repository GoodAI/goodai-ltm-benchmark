# Datasets


## Adding your own
There are interfaces and functionalities provided for you to implement your own datasets. New datasets should inherit from `DatasetInterface` (at `dataset_interfaces/interface.py`) or `GPTGenerated` (`dataset_interfaces/gpt_generated.py`).

Broadly speaking, a `TestExample` consists of a script, some expected responses, methods to evaluate responses from the agent, and what the information gap should be between statements in the script.


### DatasetInterface
If inheriting from `DatasetInterface` You should implement the following methods:
- `generate_examples()` which will return a list of `TestExample` objects.
- `evaluate_correct` which will take a list of questions, responses and expected answers for one test, and return the score given, the maximum score achievable for the test, and a list of strings explaining why each score was (or was not) given.

### GPTGenerated
If inheriting from `GPTGenerated` You should write a JSON file containing
a prompt which will inform GPT-3.5-turbo what to do in order to generate the information, questions and answers.

An example of the JSON for the `delayed_recall` test:
```json
{
  "content":"Generate a comprehensive list of ten distinct facts spanning various subjects (e.g., history, science, arts) pertaining to a fictional world.",
  "question":"Generate a series of questions to evaluate the understanding of each distinct fact from the content prompt above. There needs to be ten questions.",
  "answer":"For each of the generated questions based on the generated content, provide the key information that would make for a desired answer. There needs to be ten answers."
}
```

As a word of caution: The generation process for these questions and answers are not always accurate. Some questions can end up being too general, in that they do not ask about a particular fact, but rather a wide commonsense term.
We have found that generating totally fictional facts (about a fictional place) and processes (preparing to use a fictional piece of technology) works best, otherwise the model can give general commonsense advice which is marked correct.

### Dynamic Tests

Instead of having a fixed script, dynamic tests are generated on the fly, and they can react to the agent's responses and incorporate them into the test. In order to define a dataset with dynamic tests, your dataset class will need to inherit from `DynamicDataset`, and you will need to define custom test examples by inheriting from `DynamicExample`. There are some things that you must do differently in the case of dynamic tests:

- Set a `max_score` for the test.
- Implement `action_iter`. It is a Python generation function, which is expected to yield one `TestAction` object at a time until the test has completed. You should also update `score` according to how the test goes.

Additionally, there are some helper methods available in the base class:

- `say` Returns a `SendMessageAction` object. If `question` is set to `True`, the message will be appended to the test's script.
- `wait` Returns a `WaitAction` object.
- `ask_llm` Calls an OpenAI LLM (by default `gpt-3.5-turbo`) and returns a text response. It also registers the cost of the call as part of the benchmark management costs.

Finally, we encourage you to take a look at the [restaurant task](restaurant.py) for a complete example of how to implement a dynamic test.

### Evaluation

(First read the [runner documentation](../runner/README.md) to understand how test running and evaluations work) 

To evaluate a `TestExample` from your dataset (not dynamic), you should implement `evaluate_correct()` which will get lists  of the questions asked, the answers given by the agent, and the answers that were expected.  Returning a tuple of `score, max_score, [reasons]`. There is a path to use GPT4 to evaluate the answers relative to the expected answers with `evaluate_correct_gpt()`.

If `evaluate_correct()` is not suitable for your use case, you can create a callback by implementing `continual_evaluation_callback()` instead. This will be run in lieu of `evaluate_correct()` and must set `example.finished=True` in one of its code paths.

For example usages of each of these implementations, see:
* `datasets/colours.py  (evaluate_correct()`)
* `datasets/prospective_memory.py (continual_evaluation_callback())` 
* `datasets/delayed_recall.py (GPTGenerated, evaluate_correct_gpt())`
