s_ltm_template_queries_info = """
{user_info_description}

== New user question/statement ==
{user_content}
==

Based on the information above, your task is to provide a new scratchpad
containing your model of the world, particularly information the user expects
you to keep track of, or facts relating to the user.

The updated scratchpad should be compact. Avoid storing unimportant general knowledge 
you're already aware of. Capture information provided by the user without
omitting important details. Exercise judgment in determining if new information replaces,
augments or removes existing information.

Important: Avoid removing prior information you're keeping track of if it's possibly
still relevant.

Structure information in a way that will facilitate its interpretation 
going forward, e.g. by using outlines. Be creative.

Explain your careful reasoning in regards to how the scratchpad should be updated. In 
particular, determine if the user has provided information relating to themselves, or 
information they may be expecting you to keep track of.

Then write the scratchpad content with ```txt and ``` delimiters. Your reasoning and
scratchpad content follows:
"""
