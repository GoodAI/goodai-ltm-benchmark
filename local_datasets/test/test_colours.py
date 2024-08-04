import json
from local_datasets.colours import ColourDataset


def test_colours():
    with open('datasets/test/test_cases_colours.json', 'r') as file:
        test_cases = json.load(file)

    dataset = ColourDataset()

    for test_case in test_cases:
        questions = test_case['questions']
        responses = test_case['responses']
        expected_answers = test_case['expected_answers']

        score, max_score, response = dataset.evaluate_correct(questions, responses, expected_answers)

        assert score == max_score, f"Test Case: {test_case}\nExpected: {max_score}\nResult: {score}\nResponse: {response}"
