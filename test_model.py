from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset


hard_preference = GEval(
    name = "Hard preference",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    evaluation_steps=[
        "Read the user request from the input carefully to understand the context and requirements.",
        "Compare the responses of Model 1 and Model 2 for relevance, correctness, and clarity based on the user request.",
        "Evaluate the mathematical correctness of any facts or calculations presented in both model responses.",
        "Assign 0 score if the response of Model 1 is better, 5 score if responses are equally good, and 10 score if Model 2 response is better."
    ],
    model="gpt-4o-mini",
    verbose_mode=True,
    threshold=0.7
)

soft_preference = GEval(
    name = "Soft preference",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    evaluation_steps=[
        "Read the user request from the input carefully to understand the context and requirements.",
        "Compare the responses of Model 1 and Model 2 for relevance, correctness, and clarity based on the user request.",
        "Evaluate the mathematical correctness of any facts or calculations presented in both model responses.",
        "Assign 0 score if the response of Model 1 is better, 5 score if responses are equally good, and 10 score if Model 2 response is better."
    ],
    model="gpt-4o-mini",
    verbose_mode=True,
    threshold=0.5
)

comp_dataset = EvaluationDataset()
comp_dataset.add_test_cases_from_csv_file(
    file_path='comp_dataset.csv', 
    input_col_name='prompt', 
    actual_output_col_name='responses',
)

def test_dataset():
    comp_dataset.evaluate([hard_preference, soft_preference])
