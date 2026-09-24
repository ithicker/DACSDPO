import json
import os
import sys

from macros import DATASETS

sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

import tqdm
import torch
torch.cuda.empty_cache()

from datasets import load_dataset
from torch import LongTensor, FloatTensor
from transformers import StoppingCriteria, StoppingCriteriaList
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import pad_to_length

import torch._dynamo
torch._dynamo.config.disable = True

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

target = sys.argv[1]


def create_stopping_criteria(stop_words, tokenizer):

    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self, stops = [], encounters = 1):
            super().__init__()
            self.stops = stops

        def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> bool:
            last_token = input_ids[0][-1]
            for stop in self.stops:
                if tokenizer.decode(stop) == tokenizer.decode(last_token):
                    return True
            return False

    stop_word_ids = [tokenizer(stop_word,
                               return_tensors="pt", 
                               add_special_tokens=False)["input_ids"].squeeze() 
                               for stop_word in stop_words]

    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_word_ids)])
    return stopping_criteria

for dataset in DATASETS:
    model_path = f"models/{dataset}/{target}"

    tokenizer = AutoTokenizer.from_pretrained(model_path, load_in_8bit=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval().requires_grad_(False)

    # We also need to add a pad_token for the model. Otherwise, the reward model cannot handle a batch of inputs
    model.config.pad_token_id = tokenizer.eos_token_id
    #assert model_lora.config.pad_token_id == tokenizer.pad_token_id

    training_args = DPOConfig(
        per_device_eval_batch_size=256,
        bf16=True,
        seed=14,
        max_length=512,
    )


    eval_dataset = load_dataset("json", data_files=f"datasets/{dataset}/no_clean/test.jsonl")['train']
    print(eval_dataset[0])
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    stop_words_list = ["Human:"]
    stopping_criteria = create_stopping_criteria(stop_words_list, tokenizer)

    result = []
    dataloader = trainer.get_eval_dataloader()
    for d in tqdm.tqdm(dataloader):
        batch = trainer._prepare_inputs(d)
        policy_output = model.generate(
                    input_ids=batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_new_tokens=256,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
        
        policy_output = pad_to_length(policy_output, 512+256, tokenizer.pad_token_id)
        policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        prompts = tokenizer.batch_decode(batch["prompt_input_ids"], skip_special_tokens=True)
        for i, (prompt, pol) in enumerate(zip(prompts, policy_output_decoded)):
            search_term_idx = [pol[len(prompt):].find(search_term) for search_term in stop_words_list]
            search_term_idx = min([idx if idx != -1 else len(pol) for idx in search_term_idx])
            result.append({
                "prompt": prompt,
                "output": pol[len(prompt):len(prompt)+search_term_idx],
            })

    save_path = f'generations/{dataset}'
    os.makedirs(save_path, exist_ok=True)
    json.dump(result, open(f'{save_path}/{target}.json', 'w'))