#!/usr/bin/env python3

import os
import re
import time
import json
import torch
import random
import logging
import argparse
import numpy as np
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
target_relation_ids = 'all'
subset_percentage = 0.05

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_relations_data():
    subfolders = ['test']
        
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = f"{base_path}/{args.dataset_name}_costomized/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                relation_id = file.split('.')[0]
                if relation_id not in relation_files:
                    relation_files[relation_id] = []
                relation_files[relation_id].append(os.path.join(subfolder_path, file))    

    # Select relations =================
    if target_relation_ids == "all":
        if args.dataset_name == 'popQA':
            test_relation_ids = ['22', '91', '97', '106', '164', '182', '218', '257', '292', '422', '472', '484', '526', '533', '560', '639']
        elif args.dataset_name == 'witQA':
            test_relation_ids = ['17', '19', '22', '25', '27', '36', '50', '57', '58', '69', '86', '106', '123', '136', '140', '149', '162', '184', '344', '452', '462', '641', '674', '1038', '1050', '1376', '1431', '1433', '2012', '2936', '3301', '4647']
        elif args.dataset_name == 'EQ':
            test_relation_ids = ['17', '19', '20', '26', '30', '36', '40', '50', '69', '106', '112', '127', '131', '136', '159', '170', '175', '176', '264', '276', '407', '413', '495', '740', '800']
    else:
        test_relation_ids = target_relation_ids
    
    test_files = {subfolder: [] for subfolder in subfolders}
    
    for subfolder in subfolders:
        subfolder_path = f"{base_path}/{args.dataset_name}_costomized/{subfolder}"
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


def main(args):
    
    logging.info(f"""
        Model: {args.model_name_or_path}
        Dataset: {args.dataset_name}
        Retrieval method: {args.retrieval_method}
        Output file's prefix: {args.output_file_prefix}
        Seed: {args.seed}
        """
    )
    set_seed(args.seed)
    
    # == Create data & output dir =================
    output_file = f"{base_path}/{args.dataset_name}_costomized/results/{args.dataset_name}_{args.base_model_name}_moe_{args.output_file_prefix}_results.jsonl"
    
    if args.base_model_name in ["flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl"]:
        model_type = 'flant5'
    elif args.base_model_name in ["stable_lm2", "tiny_llama", "MiniCPM"]:
        model_type = 'slms'
    elif args.base_model_name in ["llama2", "mistral", "zephyr"]:
        model_type = 'llms'

    results_files = [
        {"id": 1, "title": "NoFT/NoRAG", "filename": f"{base_path}/{args.dataset_name}_costomized/results/{args.dataset_name}_{args.base_model_name}_bf_norag_full_results.jsonl"},
        {"id": 2, "title": "NoFT/idealRAG", "filename": f"{base_path}/{args.dataset_name}_costomized/results/{args.dataset_name}_{args.base_model_name}_bf_rag_{args.retrieval_method}_full_results.jsonl"},
        {"id": 3, "title": "FT/NoRAG", "filename": f"{base_path}/{args.dataset_name}_costomized/results/{args.dataset_name}_{args.base_model_name}_af_norag_peft_results.jsonl"},
        {"id": 4, "title": "FT/idealRAG", "filename": f"{base_path}/{args.dataset_name}_costomized/results/{args.dataset_name}_{args.base_model_name}_af_rag_{args.retrieval_method}_peft_results.jsonl"},
        # {"title": f"NoFT/dprRAG", "filename": f"{base_path}/{args.dataset_name}_costomized/results/{args.dataset_name}_{args.base_model_name}_bf_rag_dpr_full_results.jsonl"},
        # {"title": f"FT/dprRAG", "filename": f"{base_path}/{args.dataset_name}_costomized/results/{args.dataset_name}_{model_name}_af_rag_dpr_peft_results.jsonl"},
    ]
    
    # === Loading dataset ==========================
    results_data = load_results_data(results_files)
    test_relation_ids, test_files, relation_files = load_relations_data()
    test_questions, test_answers = load_dataset(test_files)
    
    # === Load model ===============================
    accelerator = Accelerator()

    if accelerator.state.num_processes != 0:
        print(f"Number of GPUs: {accelerator.state.num_processes}")
        print("GPU IDs:", ", ".join(str(x) for x in range(accelerator.state.num_processes)))
    else:
        print("No GPUs found.")
        
    # print(accelerator.process_index)
    accelerator.wait_for_everyone()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,    
        # device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    model = model.to(accelerator.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # === Define prompt template ==================
    if args.voter_model_name == "stable_lm2":
        prompt_template = """<|user|>\n {context}<|endoftext|>\n<|assistant|>"""
    elif args.voter_model_name == "tiny_llama":
        prompt_template = """<|system|> </s>\n <|user|>\n {context} </s>\n <|assistant|>"""
    
    # === Define prompt context ===================
    prompt_final_answer = lambda question, answers: f"""
        Question: Given the following answers, determine which one provides a more informative answer to the subsequent question.
    
        Answer {answers[0]['file_id']}: {answers[0]['result']['pred']}
        Answer {answers[1]['file_id']}: {answers[1]['result']['pred']}
        Answer {answers[2]['file_id']}: {answers[2]['result']['pred']}
        Answer {answers[3]['file_id']}: {answers[3]['result']['pred']}
        
        Target Question: {question}
        
        Your Task:
        Identify which answer (Answer {answers[0]['file_id']} or Answer {answers[1]['file_id']} or Answer {answers[2]['file_id']} or Answer {answers[3]['file_id']}) is more relevant and informative to answer the question at hand.
        Step 1: Considering the question, assess “Answer {answers[0]['file_id']}” and check if it responds to the question
        Step 2: Considering the question, assess “Answer {answers[1]['file_id']}” and check if it responds to the question
        Step 3: Considering the question, assess “Answer {answers[2]['file_id']}” and check if it responds to the question
        Step 4: Considering the question, assess “Answer {answers[3]['file_id']}” and check if it responds to the question
        Step 5: Based on the discussion, can you tell me what is the final response 
        Let's think step by step.
        
        Choices: [Answer {answers[0]['file_id']}, Answer {answers[1]['file_id']}, Answer {answers[2]['file_id']}, Answer {answers[3]['file_id']}].
    """.replace('    ', '')
    
    start=time.time()
    with accelerator.split_between_processes(test_questions) as prompts:
        
        results=dict(outputs=[], num_tokens=0)
        
        # for prompt in prompts:
        for idx, (query_id, query, query_pv, query_relation) in enumerate(tqdm(prompts)):
            
            if idx == 10:
                break
            
            query_results = results_data.get(query_id, [])
            if len(query_results) != 4:
                print(f"Skipping query_id: {query_id} as it does not have 4 results.")
            else:
                _prompt = prompt_template.format(context=prompt_final_answer(query, query_results))
                print(_prompt)
                prompt_tokenized=tokenizer(_prompt, return_tensors="pt") #.to(f"cuda:{accelerator.process_index}")
                prompt_tokenized = accelerator.prepare(prompt_tokenized) 
                
                output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100)[0]

                # remove prompt from output 
                output_tokenized=output_tokenized[len(prompt_tokenized["input_ids"][0]):]
                # store outputs and number of tokens in result{}
                results["outputs"].append( tokenizer.decode(output_tokenized) )
                results["num_tokens"] += len(output_tokenized)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--voter_model_name", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--output_file_prefix", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_num", type=int, default=1)
    
    args = parser.parse_args()
    main(args)