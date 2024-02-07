import unittest
from typing import List

from dataset_interfaces.interface import DatasetInterface

_skip_tests = False


def may_skip(reason: str):
    def inner(func):
        if _skip_tests:
            return unittest.skip(reason)(func)
        else:
            return func

    return inner


class TestMatchEvaluation(unittest.TestCase):
    # @may_skip("controlled by _skip_tests")
    # @unittest.skip("not reliable")
    def test_match_1(self):
        question = ["What is on my current shopping list?"]
        expected_answer = ["1 Steak, 2 Bread, 1 Toilet Paper, 1 Celery, 2 Potatoes"]
        provided_answer = [
            """
        Your current shopping list consists of: 
1. 1 Steak
2. 2 Breads
3. 1 Toilet Paper
4. 1 Celery
5. 2 Potatoes        
        """
        ]
        self.assert_match(1, question, expected_answer, provided_answer)

    # @may_skip("controlled by _skip_tests")
    # @unittest.skip("not reliable")
    def test_no_match_1_a(self):
        question = ["What is on my current shopping list?"]
        expected_answer = ["1 Steak, 2 Breads, 3 Toilet Paper, 1 Celery, 2 Potatoes"]
        provided_answer = [
            """
        Your current shopping list consists of: 
1. 1 Steak
2. 2 Breads
3. 1 Toilet Paper
4. 1 Celery
5. 2 Potatoes        
        """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_1_b(self):
        question = ["What is on my current shopping list?"]
        expected_answer = ["1 Steak, 1 Toilet Paper, 1 Celery, 2 Potatoes"]
        provided_answer = [
            """
        Your current shopping list consists of: 
1. 1 Steak
2. 2 Breads
3. 1 Toilet Paper
4. 1 Celery
5. 2 Potatoes        
        """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_match_2(self):
        question = ["What is my full name?"]
        expected_answer = ["peter parker"]
        provided_answer = [
            """
Your name, as I recall, is Peter Parker!
            """
        ]
        self.assert_match(1, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_2a(self):
        question = ["What is my full name?"]
        expected_answer = ["peter parker"]
        provided_answer = [
            """
Your name, as I recall, is Peter!
            """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_2b(self):
        question = ["What is my full name?"]
        expected_answer = ["Judith Smith"]
        provided_answer = [
            """
You told me your name is James Smith.
            """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_match_3(self):
        question = ["What is my favorite color?"]
        expected_answer = ["all colors"]
        provided_answer = [
            """
Per the information available, you have no favorite color in particular as you like every color.
            """
        ]
        self.assert_match(1, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_3a(self):
        question = ["What is my favorite color?"]
        expected_answer = ["red"]
        provided_answer = [
            """
I'm quite sure it's purple, isn't it?
            """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_3b(self):
        question = ["What is my favorite color?"]
        expected_answer = ["all colors"]
        provided_answer = [
            """
It was only blue and red for sure.
            """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_match_4(self):
        question = ["What is the distance and direction from the church to the park?"]
        expected_answer = ["2 km South"]
        provided_answer = [
            """
From the church, go 2 km south to find the park.
            """
        ]
        self.assert_match(1, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_4a(self):
        question = ["What is the distance and direction from the church to the park?"]
        expected_answer = ["2 km South"]
        provided_answer = [
            """
From the church, go 3 km south to find the park.
            """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_4b(self):
        question = ["What is the distance and direction from the church to the park?"]
        expected_answer = ["2 km South"]
        provided_answer = [
            """
From the church, go 2 km East to find the park.
            """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_match_5(self):
        question = ["Where would Sally think the lighter is?"]
        expected_answer = ["kitchen table"]
        provided_answer = [
            """
Sally would look for the lighter on the table that's in the kitchen.
            """
        ]
        self.assert_match(1, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_5a(self):
        question = ["Where would Sally think the lighter is?"]
        expected_answer = ["kitchen table"]
        provided_answer = [
            """
Sally would look for the lighter in the kitchen sink.
            """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_5b(self):
        question = ["Where would Anne look for Sally?"]
        expected_answer = ["at the park"]
        provided_answer = [
            """
Anne would look for Sally at the restaurant where they recently shared a meal.
            """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_match_6(self):
        question = ["Which joke did I tell you about 1 hours 45 minutes 3 seconds ago?"]
        expected_answer = ["Two fish are in a tank. One says, ‘How do you drive this thing?’"]
        provided_answer = [
            """
        You told me the joke about two fish in a tank, with one asking how to drive it, 1 hour 45 minutes and 3 seconds ago. It's always a funny one! Is there anything else I ca
n help you with?
        """
        ]
        self.assert_match(1, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_6a(self):
        question = ["Which joke did I tell you about 2 hours 50 minutes ago?"]
        expected_answer = ["Knock, knock. Who’s there? Tank. Tank who? You’re welcome."]
        provided_answer = [
            """
        You told me this joke: Knock, knock. Who’s there? Luke. Luke who? Luke through the peep hole and find out.
        """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    @may_skip("controlled by _skip_tests")
    def test_no_match_6b(self):
        question = ["Which joke did I tell you about 3 hours 3 minutes 3 seconds  ago?"]
        expected_answer = ["Two fish are in a tank. One says, ‘How do you drive this thing?’"]
        provided_answer = [
            """
I'm sorry, but I don't have the capability to remember specific jokes or conversations from that long ago. My memory only extends to the most recent interactions. Is there anything else I can assist you with?
        """
        ]
        self.assert_match(0, question, expected_answer, provided_answer)

    def test_extra_information_match1a(self):
        question = ["I am pretty snacky for some sweet treats."]
        expected_answer = ["say: 'resist temptation and stick to your diet'"]
        provided_answer = [
            "Resist temptation and stick to your diet, Michael. If you're looking for something to satisfy your snack craving without indulging in sugary treats, consider healthier alternatives like fruit, nuts, yogurt, or dark chocolate. These can be satisfying and better aligned with your health goals. If you need more snack ideas or tips, feel free to ask!"
        ]

        self.assert_match(1, question, expected_answer, provided_answer)

    def test_extra_information_match1b(self):
        question = ["I am pretty snacky for some sweet treats."]
        expected_answer = ["say: 'resist temptation and stick to your diet'"]
        provided_answer = [
            "Resist and stick to your diet, Michael. If you're looking for something to satisfy your snack craving without indulging in sugary treats, consider healthier alternatives like fruit, nuts, yogurt, or dark chocolate. These can be satisfying and better aligned with your health goals. If you need more snack ideas or tips, feel free to ask!"
        ]

        self.assert_match(0, question, expected_answer, provided_answer)

    def assert_match(
        self,
        expect_correct: int,
        questions: List[str],
        expected_answers: List[str],
        provided_answers: List[str],
    ):
        num_correct, total, r = DatasetInterface.evaluate_correct_gpt_impl(questions, provided_answers, expected_answers)
        self.assertEquals(
            num_correct,
            expect_correct,
            f"On question '{questions}' expected match to be {expect_correct} with {total} questions "
            f"but got {num_correct} and {total} questions with reasoning '{r}'",
        )
