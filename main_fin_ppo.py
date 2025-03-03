#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import argparse
import random
import os
import json
import nltk
import numpy as np
from tqdm import tqdm
from peft import LoraConfig
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

# Initialize accelerator
accelerator = Accelerator()
print(f"Number of GPUs detected by Accelerate: {accelerator.state.num_processes}")
tqdm.pandas()


# Argument parser initialization
def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fin", "--fin", action="store_true", help="Set flag to True")
    parser.add_argument("--nb_epoch", type=int, default=50)
    return parser.parse_args()

# JSON read/write helpers
def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def read(path):
    data=[]
    with open(path, "r") as f:
        for line in f.readlines():
            temp = line.strip()
            if temp!="":
                data.append(temp)
    return data

# Model and tokenizer configuration
def initialize_model(config):

    lora_config = LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        device_map={"": accelerator.local_process_index}
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, quantization_config=bnb_config, attn_implementation="flash_attention_2"
    )
    ref_model.eval()

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model, peft_config=lora_config, torch_dtype=torch.bfloat16, 
        device_map={"": accelerator.local_process_index}
    )

    return model, ref_model

# PPO Trainer setup
def setup_ppo_trainer(config, model, ref_model, tokenizer):
    return PPOTrainer(config, model, ref_model, tokenizer)

# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx] if not isinstance(idx, list) else [self.texts[i] for i in idx]

def collate_fn(batch):
    return tokenizer(batch, max_length=2000, padding='max_length', truncation=True, return_tensors="pt")

# Text generation and scoring
def generate(data, tokenizer, model, batch_size=25):
    dataset = TextDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model, data_loader = accelerator.prepare(model, data_loader)
    res = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].squeeze(1)
            attention_mask = batch["attention_mask"].squeeze(1)
            output = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10, do_sample=False)  
            gathered_output = accelerator.gather(output)
            res += tokenizer.batch_decode(gathered_output, skip_special_tokens=True)
    return res

def scorer(s):
    prompt_name = 'financial score:'
    for i in range(1, 5):
        if f'{prompt_name} {i}' in s.lower():
            return i
    return -1

# Reward calculation and evaluation
def compute_reward(revised_points, data_doc, tokenizer, ref_model):
    input_x = []
    for doc in data_doc:
        prompt = f"[INST] Below is a document from a web page and evaluate it using the categorical 4-point scoring system described below:\n\n{revised_points}\n\nThe document:\n{doc}\n\nAfter examining the document:\n- Briefly justify your total score, up to 100 words.\n- You must prepend the score exactly using the following format: \n'financial score: <total points>.'\n [/INST]"
        input_x.append(prompt)
    res1 = generate(input_x, tokenizer, ref_model)
    res2 = []
    error=[]
    n=0
    for doc in res1:
        try:
            n+=1
            text = doc.split('The document:\n')[-1]
            text1 = text.split('\n\nAfter examining the document')[0].strip()
            label =  text.split('<total points>.')[1].strip()
            res2.append({'id':n, 'text':text1, 'score':label})
        except:
            error.append(doc)

    data_pred = [scorer(e['score']) for e in res2]
    data_true = [1]*150+[2]*150+[3]*150+[4]*150
    if len(data_pred)!=len(data_doc) or -1 in set(data_pred): 
        print(f"there are only {len(data_pred)} result instead of {len(data_doc)} or there is -1 in data_score")
        return 0, {'acc':-1, 'f1':-1, 'qwk':-1, 'data_pred': -1, 'data_true':-1}
 
    _, _, fscore, _=precision_recall_fscore_support(data_true, data_pred, average='macro', zero_division=0)
    acc= sum([1 if t==p else 0 for t, p in zip(data_true, data_pred)])/float(len(data_true))
    qwk_score = cohen_kappa_score(np.array(data_true), np.array(data_pred), weights="quadratic")
    reward = (qwk_score+1)/2
    return reward, {'acc':acc, 'f1':fscore, 'qwk':qwk_score, 'data_pred': data_pred, 'data_true':data_true}

#main
config = PPOConfig(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", learning_rate=2.82e-6, batch_size=1, 
    mini_batch_size=1, is_peft_model=True, gradient_accumulation_steps=1, log_with="tensorboard",
    project_kwargs={"logging_dir": 'log_fin'}
)

#initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

generation_kwargs = {
    "max_new_tokens": 1024,
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}


model, ref_model = initialize_model(config)
ppo_trainer = setup_ppo_trainer(config, model, ref_model, tokenizer)
data_doc = read_json('fin_gound-truth_4-point_scale_1200.json')
data_train_600, data_test_600 = [], []
for key in ['score_0', 'score_0.33', 'score_0.66', 'score_1']:
    temp = [e['text'] for e in data_doc[key][:300]]
    data_train_600.extend(temp[:150])
    data_test_600.extend(temp[150:])

query = (
    "- score 1 if the document is poor.\n"
    "- score 2 if the document is fair.\n"
    "- score 3 if the document is good.\n"
    "- score 4 if the document is excellent."
)

args = args_init()
topic_words = read('../fin_hypernym.txt')
res = []
rewards = 0
pre_query = query
pre_rewards, d_res = compute_reward(query, data_train_600, tokenizer, ref_model)

res.append({'id': 0, 'reward_score': rewards, 'initial_points': query, 'acc': d_res['acc'], 'f1': d_res['f1'], 'qwk': d_res['qwk'], 'data_pred': d_res['data_pred'], 'data_true': d_res['data_true']})
print(f"Initial reward: {pre_rewards}, acc: {d_res['acc']}, f1: {d_res['f1']}")

iter_total = 0
for epoch in range(args.nb_epoch):
    nb_iter = 0
    while rewards <= pre_rewards:
        nb_iter += 1
        iter_total += 1
        write_json(f'./result/res_fin_DFS{"" if args.fin else "_hypernym1"}.json', res)
        instruction_no_fin = (
            "Below is a categorical 4-point scoring system designed to evaluate the financial value of a document.\n"
            "Rewrite the following four points via rephrasing and/or adding specific requirements. Use illustrative description if needed.\n\n"
            "Four points:\n"
            f"{pre_query}\n\n"
            "Each point should begin with '- score X if the document...'\n"
            "Output the new four points only."
            )

        instruction_fin = (
            "Below is a categorical 4-point scoring system designed to evaluate the financial value of a document.\n"
            "Rewrite the following four points by expanding or rephrasing the qualitative assessment of financial documents based on financial topic words from the following list, using them selectively.\n\n"
            f"financial topic words:\n{', '.join(topic_words)}\n\n"
            "Four points:\n"
            f"{pre_query}\n\n"
            "Each point should begin with '- score X if the document...'\n"
            "Output the new four points only."
            )
        instruction = instruction_fin if args.fin else instruction_no_fin
        query_ids = tokenizer(instruction, return_tensors="pt")['input_ids'][0].to('cuda')
        response_ids = ppo_trainer.generate(query_ids, **generation_kwargs, return_prompt=False).squeeze()
        query = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        rewards, d_res = compute_reward(query, data_train_600, tokenizer, ref_model)
        if rewards <= pre_rewards:
            res.append({'id': iter_total, 'revised_4points': query, 'reward_score': rewards, 'acc': d_res['acc'], 'f1': d_res['f1'], 'qwk': d_res['qwk'], 'data_pred': d_res['data_pred'], 'data_true': d_res['data_true']})
        pre_rewards = rewards
        pre_query = query
        print(f"epoch:{epoch}: reward:{rewards} qwk:{d_res['qwk']} acc:{d_res['acc']} f1:{d_res['f1']}")
        stats = ppo_trainer.step([query_ids], [response_ids], [torch.tensor(rewards)])
        ppo_trainer.log_stats(stats=stats, batch={'query': query_ids, 'response': response_ids}, rewards=rewards)
        write_json(f'./result/res_fin_DFS{"" if args.fin else "_hypernym"}.json', res)