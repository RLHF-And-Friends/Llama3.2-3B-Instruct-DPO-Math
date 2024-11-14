from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset


metric = GEval(
    name = "preference",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT
    ],
    criteria='Given the request in the input, determine whether the response '
             'in the actual output is better then the response in the '
             'context. Pay attention to the correctness of math facts and '
             'calculations.'
)

comp_dataset = EvaluationDataset()
comp_dataset.add_test_cases_from_csv_file(
    file_path='comp_dataset.csv', 
    input_col_name='prompt', 
    actual_output_col_name='fine_tuned_response',
    context_col_name='base_response'
)

def test_dataset():
    comp_dataset.evaluate([metric])
