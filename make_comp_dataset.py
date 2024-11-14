import pandas as pd

from utils import infer_conversational_lm

BASE_MODEL = "RLHF-And-Friends/Llama-3.2-3B-Instruct"
FT_MODEL = "RLHF-And-Friends/Llama-3.2-3B-Instruct-DPO-Math"

BATCH_SIZE = 16

CONCATINATION_TEMPLATE = lambda base_resp, ft_resp: (
    f'Base model response:"{base_resp}"\n\nEvaluated model response:"{ft_resp}"'
)
# =============================================================================

# Load dataset
# -----------------------------------------------------------------------------

dataset = pd.read_csv('val_dataset.csv')
prompts = dataset['prompt']
chats = [[{"role": "user", "content": prompt}] for prompt in prompts]

# Infer base and fine-tuned models
# -----------------------------------------------------------------------------

base_text_responses = infer_conversational_lm(BASE_MODEL, chats, BATCH_SIZE)
ft_text_responses = infer_conversational_lm(FT_MODEL, chats, BATCH_SIZE)

# Make comparison dataset
# -----------------------------------------------------------------------------

concatenated_responses = [
    CONCATINATION_TEMPLATE(base_resp, ft_resp) 
    for base_resp, ft_resp in zip(base_text_responses, ft_text_responses)
]

dataset = pd.DataFrame({
    "prompt": prompts, 
    "responses": concatenated_responses,
})

dataset.to_csv("comp_dataset.csv", index=False)
