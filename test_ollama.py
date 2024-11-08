import pytest
import ollama
import deepeval

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset


class OllamaStub:
    def __init__(self, name):
        self._name = name

    def run(self, prompt: str) -> str:
        return ollama.generate(model = self._name, prompt=prompt)['response']
        
# model = OllamaStub('hf.co/RLHF-And-Friends/Llama-3.2-3B-Instruct:F16')
model = OllamaStub('hf.co/RLHF-And-Friends/Llama-3.2-3B-Instruct-DPO-Math:BF16')

inputs = [
    "Hi!",  # greet
    "What is 2 + 2?",  # simple
    "Simplify the algebraic expression `(3x^2 - 4y^3) / (2x)`"  # hard
]

outputs = [model.run(input) for input in inputs] 

test_cases = [
    LLMTestCase(
        input = input,
        actual_output = output
    ) for input, output in zip(inputs, outputs)
]

dataset = EvaluationDataset(test_cases)


@pytest.fixture
def correctness():
    return GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually "
                 "correct based on the input.",
        evaluation_params=[
            LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT
        ],
    )


@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_model(test_case: LLMTestCase, correctness: GEval):
    deepeval.assert_test(test_case, [correctness])

