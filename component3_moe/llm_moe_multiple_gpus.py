#!/usr/bin/env python3

import os
import re
import json
import torch
import random
import logging
import argparse
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)

base_path  = "component0_preprocessing/generated_data"
dataset_name = 'popQA'
target_relation_ids = 'all'
subset_percentage = 0.05

gen_models = [
    "flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl",
    "stable_lm2", "tiny_llama", "MiniCPM",
    "llama2", "mistral", "zephyr"
]
    
retrieval_model = 'ideal'
model_name = gen_models[5]

if model_name in ["flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl"]:
    model_type = 'flant5'
elif model_name in ["stable_lm2", "tiny_llama", "MiniCPM"]:
    model_type = 'slms'
elif model_name in ["llama2", "mistral", "zephyr"]:
    model_type = 'llms'

results_files = [
    {"id": 1, "title": "NoFT/NoRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{dataset_name}_{model_name}_bf_norag_full_results.jsonl"},
    {"id": 2, "title": "NoFT/idealRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{dataset_name}_{model_name}_bf_rag_{retrieval_model}_full_results.jsonl"},
    {"id": 3, "title": "FT/NoRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{dataset_name}_{model_name}_af_norag_peft_results.jsonl"},
    {"id": 4, "title": "FT/idealRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{dataset_name}_{model_name}_af_rag_{retrieval_model}_peft_results.jsonl"},
    # {"title": f"NoFT/dprRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{dataset_name}_{model_name}_bf_rag_dpr_full_results.jsonl"},
    # {"title": f"FT/dprRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{dataset_name}_{model_name}_af_rag_dpr_peft_results.jsonl"},
]

dataset_dir = 'component0_preprocessing/generated_data/{}_costomized'.format(dataset_name)
test_dir = f"{dataset_dir}/test"
output_file = f"{base_path}/{dataset_name}_costomized/results/{dataset_name}_{model_name}_moe_llm_results.jsonl"


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_relations_data():
    subfolders = ['test']
        
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = f"{base_path}/{dataset_name}_costomized/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                relation_id = file.split('.')[0]
                if relation_id not in relation_files:
                    relation_files[relation_id] = []
                relation_files[relation_id].append(os.path.join(subfolder_path, file))    

    # Select relations =================
    if target_relation_ids == "all":
        if dataset_name == 'popQA':
            test_relation_ids = ['22', '91', '97', '106', '164', '182', '218', '257', '292', '422', '472', '484', '526', '533', '560', '639']
        elif dataset_name == 'witQA':
            test_relation_ids = ['17', '19', '22', '25', '27', '36', '50', '57', '58', '69', '86', '106', '123', '136', '140', '149', '162', '184', '344', '452', '462', '641', '674', '1038', '1050', '1376', '1431', '1433', '2012', '2936', '3301', '4647']
        elif dataset_name == 'EQ':
            test_relation_ids = ['17', '19', '20', '26', '30', '36', '40', '50', '69', '106', '112', '127', '131', '136', '159', '170', '175', '176', '264', '276', '407', '413', '495', '740', '800']
    else:
        test_relation_ids = target_relation_ids
    
    test_files = {subfolder: [] for subfolder in subfolders}
    
    for subfolder in subfolders:
        subfolder_path = f"{base_path}/{dataset_name}_costomized/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_id = file.split('.')[0]
                if file_id in test_relation_ids:
                    test_files[subfolder].append(os.path.join(subfolder_path, file))

    print("Selected Relation ID:", test_relation_ids)
    logging.info(f"Selected Relation ID: {test_relation_ids}")

    return test_relation_ids, test_files, relation_files

def load_dataset(test_files):
    test_data = []
    for file in test_files['test']:
        relation_id = file.split('/')[-1].split('.')[0]
        data = load_json_file(file)
        for item in data:
            item['relation_id'] = relation_id
        test_data.extend(data) 
        
    test_subset_size = int(subset_percentage * len(test_data))
    subset_test_data = random.sample(test_data, test_subset_size)    
    test_questions = [(item['query_id'], item['question'], item['pageviews'], item['relation_id']) for item in subset_test_data]
    test_answers = [item['answers'] for item in subset_test_data]
    
    logging.info("Test dataset is loaded.")
    # print("Test dataset is loaded.")
    return test_questions, test_answers

def load_results_data(results_files):
    results_data = {}
    for result_file in results_files:
        file_id = result_file['id']
        with open(result_file['filename'], 'r') as file:
            for line in file:
                result = json.loads(line)
                query_id = result['query_id']
                result_obj = {
                    "file_id": file_id,
                    "result": result
                }
                
                if query_id in results_data:
                    results_data[query_id].append(result_obj)
                else:
                    results_data[query_id] = [result_obj]
    return results_data


def main(args):
    results_data = load_results_data(results_files)
    test_relation_ids, test_files, relation_files = load_relations_data()
    test_questions, test_answers = load_dataset(test_files)
    
    accelerator = Accelerator()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    accelerator.wait_for_everyone()
    start=time.time()
    
    with accelerator.split_between_processes(test_questions) as prompts:
        
        results=dict(outputs=[], num_tokens=0)
        
        # for prompt in prompts:
        for idx, (query_id, query, query_pv, query_relation) in enumerate(tqdm(prompts)):
            prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")
            output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100)[0]

            # remove prompt from output 
            output_tokenized=output_tokenized[len(prompt_tokenized["input_ids"][0]):]

            # store outputs and number of tokens in result{}
            results["outputs"].append( tokenizer.decode(output_tokenized) )
            results["num_tokens"] += len(output_tokenized)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--gpu_num", type=int, default=1)
    
    args = parser.parse_args()
    main(args)