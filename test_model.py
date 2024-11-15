from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset


metric = GEval(
    name = "preference",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    criteria='User request is given in the input. Responses of two models are '
             'given in the actual output in the following form: '
             'Model 1 response:"response"\n\nModel 2 response:"response". '
             'Give score 0 if you like model 1 response more, give score '
             '0.5 if you like responses the same and give score 1 if you like '
             'the model 2 response more. Pay attention to the correctness '
             'of math facts and calculations.',
    model="gpt-4o-mini"
)

comp_dataset = EvaluationDataset()
comp_dataset.add_test_cases_from_csv_file(
    file_path='comp_dataset.csv', 
    input_col_name='prompt', 
    actual_output_col_name='responses',
)

def test_dataset():
    comp_dataset.evaluate([metric])
