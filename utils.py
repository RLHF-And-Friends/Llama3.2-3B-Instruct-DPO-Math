from tqdm import tqdm

from transformers import pipeline


def infer_conversational_lm(
    model: str,
    chats: list[list[dict]],
    batch_size:int=8,
    max_new_tokens:int=512
) -> list[str]:

    text_generator = pipeline(
        model=model,
        device_map='auto',
        batch_size=batch_size,
        max_new_tokens=max_new_tokens
    )
    responses = []
    for idx in tqdm(
        range(0, len(chats), batch_size), desc=f'{model} inference'
    ):
        batch = chats[idx:idx+batch_size]
        responses.extend(text_generator(batch))
        
    text_reponses = [
        response[0]['generated_text'][-1]['content'] for response in responses
    ]
    
    return text_reponses
        