# Models

(First see the [runner](../runner/README.md) documentation to understand how the tests are run and the part that the models play)

A model wraps a conversational agent, so that the benchmark system can talk to it. The model's implementation details don't matter, as long as it implements a very simple interface.

A model should implement the `ChatSession` interface found at `model_interfaces/interface.py` The important methods that you are required to implement are:

* `reply(message)` which will return a `str` response to the input `message` and update the `costs_usd` attribute of the class. 
* `reset()` which will clear any trace of the interactions since the start of the chat session.
* `load()` which loads the model.
* `save()` which saves data from the model.

Agents are expected to be stateful and able to persist. State management actions must take place as part of the `reply(message)` method, and persistence checkpoints must be correctly managed by the methods `load()` and `save()`, in order to ensure a correct behaviour during the resume of a benchmark run.

It is expected that all calls to `reply(message)` incur in some cost, and therefore an exception will be raised if `costs_usd` does not increase from call to call. However, if your agent is cost-free or if you determine that the costs are negligible, you can set `is_local=True` and the costs will not be checked.

If your agent is configurable, you are expected to implement `@property name`, which will be used by our benchmarking system to generate a unique agent ID. Agent IDs are expected to be equal only if the agent implementation and configuration are identical, and they are key for storing different configurations' results.

Every agent has its own restrictions with respect to how long the messages can be. We have set it to a reasonable default of 1,000 tokens, but you can change this default, and let the runner know what your agent's maximum message length is by setting `max_message_size`.

Tokenization methods vary significantly between implementations, and the length of filler messages, as well as the span of tests, depend on this. The system will use by default a byte-pair encoding tokenizer, but we encourage you to use your own tokenizer by implementing the `token_len(text)` method of the interface, in order to ensure a fair comparison against other agents.