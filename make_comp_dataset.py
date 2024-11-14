from tqdm import tqdm

from transformers import pipeline
from datasets import load_dataset

import pandas as pd

BASE_MODEL = "RLHF-And-Friends/Llama-3.2-3B-Instruct"
FT_MODEL = "RLHF-And-Friends/Llama-3.2-3B-Instruct-DPO-Math"

BATCH_SIZE = 16

# =============================================================================

# Load dataset
# -----------------------------------------------------------------------------

dataset = pd.read_csv('val_dataset.csv')
prompts = dataset['prompt']
chats = [[{"role": "user", "content": prompt}] for prompt in prompts]

# Infer base model
# -----------------------------------------------------------------------------
text_generator = pipeline(
    model=BASE_MODEL,
    device_map='auto',
    batch_size=BATCH_SIZE,
    max_new_tokens=512
)

base_responses = []
for idx in tqdm(
    range(0, len(chats), BATCH_SIZE), desc="Base model inference"
):
    batch = chats[idx:idx+BATCH_SIZE]
    base_responses.extend(text_generator(batch))
    
base_text_reponses = [
    response[0]['generated_text'][-1]['content'] for response in base_responses
]

# Infer fine-tuned model
# -----------------------------------------------------------------------------
text_generator = pipeline(
    model=FT_MODEL,
    device_map='auto',
    batch_size = BATCH_SIZE,
    max_new_tokens=512
)

ft_responses = []
for idx in tqdm(
    range(0, len(chats), BATCH_SIZE), desc="Fine-tuned model inference"
):
    batch = chats[idx:idx+BATCH_SIZE]
    ft_responses.extend(text_generator(batch))

ft_text_responses = [
    response[0]['generated_text'][-1]['content'] for response in ft_responses
]

# Make comparison dataset
# -----------------------------------------------------------------------------

dataset = pd.DataFrame({
    "prompt": prompts, 
    "base_response": base_text_reponses, 
    "fine_tuned_response": ft_text_responses
})

dataset.to_csv("comp_dataset.csv", index=False)
