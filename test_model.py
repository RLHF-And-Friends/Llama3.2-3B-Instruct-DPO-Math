import os

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"

class EvaluatedModel:
    def __init__(self, tokenizer, model):
        self._model = model
        self._tokenizer = tokenizer

    def __call__(self, input: list[str]) -> list[str]:
        model_inputs = self._tokenizer(
            input, return_tensors = "pt", padding=True
        ).to("cuda")
        generated_ids = self._model.generate(**model_inputs)
        output = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return output


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map = 'auto', 
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
)
tokenizer.pad_token = tokenizer.eos_token
model = EvaluatedModel(tokenizer, model)

def test():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually"
                 "correct based on the input.",
        evaluation_params=[
            LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT
        ],
    )

    inputs = [
        "What is 2+2?",
        "Hi!"
    ]
    test_cases = [
        LLMTestCase(
            input=input, 
            actual_output=output
        ) for input, output in zip(inputs, model(inputs))
    ]

    dataset = EvaluationDataset(test_cases=test_cases)
    dataset.evaluate([correctness_metric])
