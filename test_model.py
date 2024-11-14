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
             'Base model response:"response"\n\nEvaluated model response:"response". '
             'Give a bigger score if you like evaluated models reponse more '
             'and less score it you like base model response more. Pay '
             'attention to the correctness of math facts and calculations.'
)

comp_dataset = EvaluationDataset()
comp_dataset.add_test_cases_from_csv_file(
    file_path='comp_dataset.csv', 
    input_col_name='prompt', 
    actual_output_col_name='responses',
)

def test_dataset():
    comp_dataset.evaluate([metric])
