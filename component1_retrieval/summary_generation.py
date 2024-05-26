#!/usr/bin/env python3

import re
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

def _extract_json_part(new_pt):
    """Extract the last PT from the user's response
    E.g., "Output: [first] (some text) [second] (more text)" --> "[second]"
    """
    new_pt = re.sub(r'\n+', ' ', new_pt).strip()
    matches = re.findall(r'(\{.*?\})', new_pt) # The latter part handle the case '{"text": "The user has no allergies", "preference_name": "allergies", "turn_number": 2}'
    new_pt = matches[-1] if matches else ''

    return new_pt

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
    
    output_file = f'component0_preprocessing/generated_data/{args.dataset_name}_costomized/highlight_results/all.jsonl'
    os.makedirs(f'component0_preprocessing/generated_data/{args.dataset_name}_costomized/highlight_results', exist_ok=True)
    
    pipe = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    max_input_tokens = 2048
    
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    test_questions, test_answers = load_dataset(test_files)
    
    # === Prompt Definition ===
    prompt_highlight_generation = lambda query, context: f"""
        Example output: {{“sentences”: []}}
        I will give you a query and a passage. You should find the most relevant sentences from the passage to answer the query.

        Relations: Occupation, Genre, Capital of, Religion, Producer, Country, Place of birth, Father, Mother, Capital, Color, Author, Director, Screenwriter, Sport, Composer

        Query: {query}

        Passage: {context}

        Your Task: Extract sentences that possibly contain answer for the question.
        Step 1: Categorize the type of query in the relations mentioned. Specify which relation the query is about.
        Step 2: Based on the query and its relation, extract sentences that contain information about the query answer.
        Step 3: Output in JSON format following the example above (i.e., `{{...}}`).
        Ensure that you distinctly label and delineate Steps 1, 2, and 3. Let's think step by step:
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
    
    
    with open(output_file, 'w') as file:
        for idx, (query_id, query, query_pv, query_relation) in enumerate(tqdm(test_questions)):

            if idx == 10:
                break

            retrieved_text = ""
            has_context = False
            if args.with_rag_corpus:
                max_token = max_input_tokens - 50
                corpus_text = "".join(ret_results[query_id]['ctxs'][i]['text'] for i in range(args.num_retrieved_passages) if i < len(ret_results[query_id]['ctxs']))
                retrieved_text = truncate_text(corpus_text, max_token)

                if retrieved_text == "":
                    logging.info(f"\nNo retrieved text found for query: {query}") 
                    print("\nNo retrieved text found for query: {}, {}".format(query_id, query))
              
            if retrieved_text != "":
                _prompt = [
                    { "role": "system", "content": ""},
                    { "role": "user", "content": prompt_highlight_generation(query=query, context=retrieved_text)}
                ]
                prompt = pipe.tokenizer.apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
                
                outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
                new_pt = outputs[0]["generated_text"]
                new_pt = new_pt.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[1].strip()
                new_pt = _extract_json_part(new_pt)
                
                item = {
                    "query_id": query_id,
                    "question": query,
                    "pageviews": query_pv,
                    "highlighted_text": new_pt["sentences"],
                }
                file.write(json.dumps(item) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_retrieved_passages", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    
    summary_generation_for_retrieved_context(args)