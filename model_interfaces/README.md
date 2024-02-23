# Models

(First see the [runner](../runner/README.md) documentation to understand how the tests are run and the part that the models play)


A model should implement the `ChatSession` interface found at `model_interfaces/interface.py` The two important methods that you are required to implement are:

* `reply(message)` which will return a `str` response to the input `message` and update the `costs_usd` attribute of the class. 
* `reset()` which will clear the memory and context completely.
* `load()` which loads the model.
* `save()` which saves data from the model.

If the agent is not able to be persistent for some reason, you should set `reset_policy = ResetPolicy.HARD`, which will reset any in-progress tests upon any restart of the tests. A soft reset will resume tests that are in progress, so if your model is not persistent, then it will be potentially penalised for forgetting important information.

It is expected that all calls to `reply(message)` incur in some cost, and therefore an exception will be raised if `costs_usd` does not increase from call to call. However, if your agent is cost-free or if you determine that the costs are negligible, you can set `is_local=True` and the costs will not be checked.

If your agent is configurable, you are expected to implement `@property name`, which will be used by our benchmarking system to generate a unique agent ID. Agent IDs are expected to be equal only if the agent implementation and configuration are identical, and they are key for storing different configuration's results.

Every agent has its own restrictions with respect to how long the messages can be. We have set it to a reasonable default of 1,000 tokens, but you can change this default, and let the runner know what your agent's maximum message length is by setting `max_message_size`.