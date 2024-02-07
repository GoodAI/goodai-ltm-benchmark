cltm_template_queries_info = """
{user_info_description}

== New user question/statement ==
{user_content}
==

Based on prior user information and the above user question/statement, your task is to provide 
(1) semantic queries for searching archived conversation history that may be relevant to reply 
to the user, and (2) a new user object with updated information provided by the user, such as 
facts about themselves or information they are expecting you to keep track of.

The updated user object should be compact. Avoid storing unimportant general knowledge 
you're already aware of. At the same time, it's important to preserve prior information 
you're keeping track of for the user. Capture information provided by the user without
omitting important details. Exercise judgment in determining if new information overrides, 
deletes or augments existing information. Property names should be descriptive.

The search queries you produce should be compact reformulations of the user question/statement,
taking context into account. The purpose of the queries is accurate information retrieval. 
Search is purely semantic. Time-based queries are unsupported.

Write JSON in the following format:

{{
    "reasoning": string, // Your careful reasoning about how the user object should be updated. In particular, does the user query/statement contain information relating to the user or something they may expect you to keep track of?
    "user": {{ ... }}, // An updated user object containing attributes, facts, world models
    "queries": array, // An array of strings: 1 or 2 descriptive search phrases
}}
"""  # noqa: E501
