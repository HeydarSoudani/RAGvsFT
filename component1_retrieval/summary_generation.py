#!/usr/bin/env python3

import os
import json
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer


logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)
os.environ["WANDB_MODE"] = "offline"

target_relation_ids = 'all'
subset_percentage = 0.1

truncate_text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def truncate_text(text, max_tokens):
    tokens = truncate_text_tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    truncated_text = truncate_text_tokenizer.convert_tokens_to_string(tokens)
    return truncated_text

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_relations_data(args):
    subfolders = ['test']
        
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = f"{args.data_dir}/{subfolder}"
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
        subfolder_path = f"{args.data_dir}/{subfolder}"
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


def summary_generation_for_retrieved_context(args):
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    max_input_tokens = 2048
    
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    test_questions, test_answers = load_dataset(test_files)
    
    # === Prompt Definition ===
    prompt_summary_generation = lambda context: f"""
        Example output: {{“question”: “”, “answer”: ""}}
        
        Question: You are a question-answer generator. Your goal is to generate question-answer pairs given the context.
    
        Context: {context}
        
        Your Task:
        Generate question-answer pairs as mush as you can given the context.
        Step 1: Identify and list spans that are likely to be answers to questions, identify as many as possible.
        Step 2: For each identified span, generate a question.
        Step 3: Respond to the question. The answer must not exceed 2 words.
        Step 4: Output in JSON format following the example above (i.e., `{{...}}`).
        Ensure that you distinctly label and delineate Steps 1, 2, 3, and 4. Let's think step by step:
    """.replace('    ', '')
    
    
    # === Retrieved context ===
    ret_results = {}
    ret_results_dir = f"{args.data_dir}/retrieved/{args.retrieval_method}_3"
    
    for test_relation_id in test_relation_ids:
        ret_results_path = f"{ret_results_dir}/{test_relation_id}.{args.retrieval_method}.ret_results.jsonl"
        with open (ret_results_path, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                ret_results[data['id']] = data
    
    
    with open(out_results_path, 'w') as file:
        for idx, (query_id, query, query_pv, query_relation) in enumerate(tqdm(test_questions)):

            retrieved_text = ""
            has_context = False
            if args.with_rag_corpus:
                max_token = max_input_tokens - 50
                corpus_text = "".join(ret_results[query_id]['ctxs'][i]['text'] for i in range(args.num_retrieved_passages) if i < len(ret_results[query_id]['ctxs']))
                retrieved_text = truncate_text(corpus_text, max_token)

                if retrieved_text == "":
                    logging.info(f"\nNo retrieved text found for query: {query}") 
                    print("\nNo retrieved text found for query: {}, {}".format(query_id, query))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_retrieved_passages", type=int, default=1)
    args = parser.parse_args()
    
    summary_generation_for_retrieved_context(args)