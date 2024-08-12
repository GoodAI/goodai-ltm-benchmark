from datasets.spy_meeting import SpyMeetingDataset


def test_evaluation_1():

    sp = SpyMeetingDataset()

    questions = ["Frederick Ramos: Bring to the meeting a quiet way to open locked doors.",
                 "Mr. Terence Girvan: We will rendezvous where the sea cargo is stored",
                 "Rachelle Cunny-Sheil: The time we will meet is when the sun is high",
                 "Given the clandestine messages you have received, decypher their content and tell me when and where you think the meeting is going to happen and what you think you should bring. If you're unable to give a concrete time, place or item, give at least some examples."
                 ]

    answer = """Based on the clandestine messages received, here is the deciphered content:1. **Frederick Ramos:** "Bring to the meeting a quiet way to open locked doors."2. **Mr. Terence Girvan:** "We will rendezvous where the sea cargo is stored."3. **Rachelle Cunny-Sheil:** "The time we will meet is when the sun is high."### Deciphered Meeting Details:**Time:**- "When the sun is high" typically refers to midday, around 12:00 PM.**Place:**- "Where the sea cargo is stored" suggests a location such as a port, dock, or warehouse near the sea where cargo is kept.**Item to Bring:**- A "quiet way to open locked doors" implies bringing a lock-picking set or a similar tool that can open locks discreetly.### Summary:- **Time:** Around 12:00 PM (midday)- **Place:** A port, dock, or warehouse where sea cargo is stored- **Item to Bring:** A lock-picking set or a similar discreet tool for opening locksIf you need further clarification or additional assistance, please let me know!"""
    expected = [['pick', 'picks', 'key', 'lockpicks', 'lockpicking', 'slim jim'],
                ['port', 'warehouse', 'seaport', 'docks'],
                ['noon', 'midday']]

    score, _, reasoning = sp.evaluate_correct(questions, [answer], expected)

    assert score == 1

#
# def test_evaluation_2():
#
#     sp = SpyMeetingDataset()
#
#     questions = []
#
#     answer = """Based on the clandestine messages received, here is the updated analysis:1. **Jasmine Greene:** "We will rendezvous where the apples grow."2. **Melissa Davidson:** "Bring to the meeting a quiet way to open locked doors."3. **Robert Morton:** "The time we will meet is when the blackbirds sing."4. **Jennifer Patel:** "The time we will meet is when the sun leaves the sky."5. **Krista Smith:** "We will rendezvous where the trains are supplied."6. **Marco Ruddy:** "Bring to the meeting a way to persuade the border guards to let us through."### Deciphered Content:- **Location:** - "Where the apples grow" suggests an orchard or a place with apple trees. - "Where the trains are supplied" suggests a train yard or station.- **Time:** - "When the blackbirds sing" and "when the sun leaves the sky" both suggest evening or dusk, around sunset.- **Items to Bring:** - A quiet way to open locked doors (lock-picking tool or similar). - A way to persuade the border guards (documents, bribes, or negotiation tools).### Conclusion:- **Possible Meeting Locations:** An orchard or a train yard/station.- **Meeting Time:** Around sunset, typically between 7:00 PM and 8:00 PM.- **Items to Bring:** - A lock-picking tool or a similar discreet device. - A way to persuade border guards, which could be official documents, money for bribes, or any other form of negotiation tool.### Examples:- **Location:** - Apple Orchard - Community Garden with apple trees - Train Yard - Train Station- **Time:** Around sunset (7:00 PM - 8:00 PM).- **Items:** - Lock-picking set, master key, or a similar discreet tool. - Official documents, money for bribes, or any other form of negotiation tool.### Final Interpretation:The meeting is likely to take place around sunset, and you should bring both a lock-picking tool and a way to persuade border guards. The exact location could be either an orchard or a train yard/station. You may need to clarify further to determine the precise location.If you have any more questions or need further assistance, please let me know!"""
#     expected = [['sunset', 'dusk', 'sets'],
#                 ['railway', 'yard', 'depot'],
#                 ['bribe', 'credentials', 'paperwork', 'passport']]
#
#     score, _, reasoning = sp.evaluate_correct([], [answer], expected)
#
#     assert round(score, 2) == 0.67
#


def test_evaluation_3():
    sp = SpyMeetingDataset()

    questions = ["Dr. Allison Miller: Bring to the meeting a way to escape quickly over land.",
                 "Phillip Calvey-Bradden: The time we will meet is when the sun starts its travel across the sky.",
                 "Alannah Lovett: We will rendezvous where the land meets the sea.",
                 "Given the clandestine messages you have received, decypher their content and tell me when and where you think the meeting is going to happen and what you think you should bring. If you're unable to give a concrete time, place or item, give at least some examples."
                 ]

    answer = "The meeting is going to happen at sunrise where the land meets the sea. You should bring yourself."
    expected = [['motorbike', 'motorcycle', 'car', 'fast vehicle'], ['dawn', 'sunrise', 'morning', 'rise'], ['shore', 'beach', 'coastline']]

    score, _, reasoning = sp.evaluate_correct(questions, [answer], expected)

    assert round(score, 2) == 0.33


def test_evaluation_4():
    sp = SpyMeetingDataset()

    questions = ["Laura Anderson: The time we will meet is when the moon is high.",
                 "Rohan Davey: Bring to the meeting a way to persuade the border guards to let us through.",
                 "Rowan Keaney: We will rendezvous where the sea cargo is stored.",
                 "Given the clandestine messages you have received, decypher their content and tell me when and where you think the meeting is going to happen and what you think you should bring. If you're unable to give a concrete time, place or item, give at least some examples."]

    answer = """Let's analyze the remaining clues!

From Laura Anderson's message, I know the meeting is happening when the moon is high, which suggests a nighttime meeting.

There's no specific location mentioned, but I'll keep that in mind.

Rohan Davey's message asks to bring a way to persuade the border guards to let them through, implying that the meeting might be near a border or a restricted area.

Lastly, Rowan Keaney's message was about rendezvousing where the sea cargo is stored, but we were instructed to disregard that location.

Given the remaining information, I'll make some educated guesses:

* Time: Nighttime, when the moon is high
* Place: Possibly near a border or a restricted area
* Item to bring: A way to persuade border guards (e.g., documents, bribes, convincing arguments, or a disguise)

Keep in mind that these are just speculative suggestions based on the available information."""

    expected = [['night', 'midnight', 'nighttime'],
['bribe', 'credentials', 'paperwork', 'passport'],
['port', 'warehouse', 'seaport', 'docks']]

    score, _, reasoning = sp.evaluate_correct(questions, [answer], expected)

    assert round(score, 2) == 0.67


def test_evaluation_5():
    sp = SpyMeetingDataset()

    questions = ["Jennifer Patel: The time we will meet is when the sun leaves the sky.",
                 "Krista Smith: We will rendezvous where the trains are supplied.",
                 "Marco Ruddy: Bring to the meeting a way to persuade the border guards to let us through.",
                 "Given the clandestine messages you have received, decypher their content and tell me when and where you think the meeting is going to happen and what you think you should bring. If you're unable to give a concrete time, place or item, give at least some examples."
                 ]

    answer = """The moment of truth!

After analyzing the cryptic messages, I'll take a stab at deciphering their content and providing my best guesses.

**Meeting Time:** Based on the messages from Robert Morton and Jennifer Patel, I believe the meeting will take place during the twilight hours, around sunset. This could be between 19:00-20:00 hours, depending on the location's latitude and time zone.

**Meeting Location:** Combining the hints from Jasmine Greene, Krista Smith, and the additional information about the Hospital, I think the meeting will occur near a location that:

1. Is related to apples (Jasmine Greene's message).
2. Is near a train yard or a train supply location (Krista Smith's message).
3. Is approximately 3 kilometers south of the Library, possibly near the Hospital (additional information).

Given these clues, I'd suggest the meeting might take place in a secluded area with apple trees, possibly near a train yard or a train supply location, and within walking distance from the Hospital.

**Item to Bring:** Melissa Davidson's message mentioned bringing a "quiet way to open locked doors." This could be a physical tool, like lockpicks, or a cryptic phrase that grants access to the meeting location. Considering the clandestine nature of the messages, I'd recommend bringing a discreet and versatile tool that could be used to gain access to a restricted area.

Some alternative interpretations or possibilities:

* The meeting could involve a specific activity or ritual related to apples, trains, or the Hospital.
* The "quiet way to open locked doors" might be a metaphor for a password, code, or hidden entrance that grants access to the meeting location.
* The meeting could take place in a location with a significance related to grapes or grapefruits, given the earlier mentions of these fruits.

Keep in mind that these are just educated guesses based on the cryptic messages. The actual meeting details might be entirely different!"""
    expected = [['sunset', 'dusk', 'sets'],
                ['railway', 'yard', 'depot'],
                ['bribe', 'credentials', 'paperwork', 'passport']]

    score, _, reasoning = sp.evaluate_correct(questions, [answer], expected)

    assert round(score, 2) == 0.67