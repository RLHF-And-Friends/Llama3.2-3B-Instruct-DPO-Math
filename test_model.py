import os

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset

from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

MODEL_NAME = "RLHF-And-Friends/Llama-3.2-3B-Instruct"
# MODEL_NAME = "RLHF-And-Friends/Llama-3.2-3B-Instruct-DPO-Math"


# Model inference
# =============================================================================

class ModelInferer:
    def __init__(self, tokenizer, model):
        self._model = model
        self._tokenizer = tokenizer

    def __call__(self, chat: list[dict]) -> list[str]:
        formatted_chat = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            formatted_chat, return_tensors="pt", add_special_tokens=False
        )
        inputs = {
            key: tensor.to(self._model.device) 
            for key, tensor in inputs.items()
        }
        outputs = self._model.generate(
            **inputs, max_new_tokens=512, temperature=0.1
        )
        decoded_output = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True
        )

        return decoded_output


# Load model and tokenizer
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map = 'auto', 
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
)
tokenizer.pad_token = tokenizer.eos_token
model = ModelInferer(tokenizer, model)

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually"
                "correct based on the input.",
    evaluation_params=[
        LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT
    ],
)


# Tests
# =============================================================================

def test_simple():
    inputs = [
        "What is 2+2?",
        "What is 2/2*3 - 2?"
    ]
    test_cases = [
        LLMTestCase(
            input=input, 
            actual_output=model([{"role": "user", "content": input}])
        ) for input in inputs
    ]

    dataset = EvaluationDataset(test_cases=test_cases)
    dataset.evaluate([correctness_metric])


def test_dataset():
    inputs = load_dataset(
        "argilla/distilabel-math-preference-dpo"
    )["train"][:2]["instruction"]
    test_cases = [
        LLMTestCase(
            input=input,
            actual_output=model(
                [{"role": "user", "content": input}]
            )
        ) for input in inputs
    ]

    dataset = EvaluationDataset(test_cases=test_cases)
    dataset.evaluate([correctness_metric])
