#!/usr/bin/env python3

import os
import re
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

base_path  = "component0_preprocessing/generated_data"
target_relation_ids = 'all'
subset_percentage = 1.0

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

def get_pred_values(query_id, file_list):
    # Dictionary to store results with filename as key and pred value as value
    results = {}

    # Loop over each file in the list
    for file_name in file_list:
        # Open and read the JSONL file
        with open(file_name, 'r') as file:
            # Read each line (each line is a JSON object)
            for line in file:
                # Convert JSON string to a Python dictionary
                data = json.loads(line)
                
                # Check if the current dictionary has the query_id we're looking for
                if data.get('query_id') == query_id:
                    # Extract the 'pred' value
                    pred_value = data.get('pred', 'Not available')
                    # Store the result
                    results[file_name] = pred_value
                    break  # Assuming only one matching queryId per file
            else:
                # If no match was found in the entire file, mark as not found
                results[file_name] = 'queryId not found in this file'

    return results

def extract_number(text):
    match = re.search(r'\d+', text)
    if match:
        return int(match.group(0))
    else:
        return None

def load_chunked_data(args):
    file_path = f"component3_moe/naive_method/{args.dataset_name}_chunked/part_{args.chunk_index}.json"
    test_questions = []
    test_answers = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            test_questions.append(tuple(item[:-1]))
            test_answers.append(item[-1])
    
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
    
    # == Create data & output dir ===========================
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
    results_data = load_results_data(results_files)
    
    # === Loading dataset ==========================
    # test_relation_ids, test_files, relation_files = load_relations_data()
    # test_questions, test_answers = load_dataset(test_files)
    test_questions, test_answers = load_chunked_data(args)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    pipe = pipeline(
        task="text-generation",
        model=args.model_name_or_path,
        tokenizer=tokenizer,
        max_new_tokens = 8
    )
    
    # # Example output: {{“question”: “”, “answer”: “”, “resource”: ””}}
    # Step 6: Output in JSON format according to the example above (ie `{{...}}`). The resource indicates the answer number on which the final answer is derived.
    # Ensure that you distinctly label and delineate Steps 1, 2, 3, 4, and 5. Let's think step by step.
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
    
    # prompt_final_answer = lambda question, answers: f"""
    # Question: Given the following answers, determine which one provides a more informative answer to the subsequent question.
    
    # Answer {answers[0]['file_id']}: {answers[0]['result']['pred']}
    # Answer {answers[1]['file_id']}: {answers[1]['result']['pred']}
    # Answer {answers[2]['file_id']}: {answers[2]['result']['pred']}
    # Answer {answers[3]['file_id']}: {answers[3]['result']['pred']}
    
    # Target Question: {question}
    
    # Your Task:
    # Identify which answer (Answer {answers[0]['file_id']} or Answer {answers[1]['file_id']} or Answer {answers[2]['file_id']} or Answer {answers[3]['file_id']}) is more relevant and informative to answer the question at hand.
    # Choices: [Answer {answers[0]['file_id']}, Answer {answers[1]['file_id']}, Answer {answers[2]['file_id']}, Answer {answers[3]['file_id']}].
    # Do not exceed 2 words.
    
    # Answer:
    # """.replace('    ', '')

    accuracy = []
    with open(output_file, 'w') as file:
        for idx, (query_id, query, query_pv, query_relation) in enumerate(tqdm(test_questions)):
            
            if idx == 10:
                break
        
            query_results = results_data.get(query_id, [])
            if len(query_results) == 4:
            
                _prompt = [
                    { "role": "system", "content": ""},
                    { "role": "user", "content": prompt_final_answer(query, query_results)}
                ]
            
                prompt = pipe.tokenizer.apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
                outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
                output = outputs[0]["generated_text"]
                result = output.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[1].strip()
                
                # logging.info(query)
                # logging.info(result)
                
                final_answer = extract_number(result)
                if final_answer == None:
                    print(f'output is: {result}')
                    final_answer = 4
    
                pred = query_results[int(final_answer)-1]['result']['pred']
                
                is_correct = False
                for pa in test_answers[idx]:
                        if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                            is_correct = True
                accuracy.append(is_correct)
                
                if idx < 10 or idx % 200 == 0:
                    logging.info('\n')
                    logging.info(f"Prompt: {result}")
                    logging.info(f"Query: {query}")
                    logging.info(f"Pred: {pred}")
                    logging.info(f"resource: {final_answer}"),
                    logging.info(f"Labels: {test_answers[idx]}")
                    logging.info(f"Final decision: {is_correct}")
                    logging.info('====')
                    # print('\n')
                    # print(f"Prompt: {new_pt}")
                    # print(f"Query: {query}")
                    # print(f"Pred: {pred}")
                    # print(f"resource: {final_ans['resource']}")
                    # print(f"Labels: {test_answers[idx]}")
                    # print(f"Final decision: {is_correct}")
                    # print('====')
                
                item = {
                    "query_id": query_id,
                    "question": query,
                    "possible_answers": test_answers[idx],
                    "pred": pred,
                    "resource": final_answer,
                    "is_correct": is_correct,
                    "pageviews": query_pv
                }
                file.write(json.dumps(item) + '\n')
    
    acc = sum(accuracy) / len(accuracy)
    logging.info(f"Accuracy: {acc * 100:.2f}%")
    print(f"Accuracy: {acc * 100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--output_file_prefix", type=str)
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()
    main(args)