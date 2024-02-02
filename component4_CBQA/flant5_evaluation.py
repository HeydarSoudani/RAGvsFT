#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from peft import PeftConfig, PeftModel
import torch
import argparse, os, json
import numpy as np
import random
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[
        # logging.FileHandler("app.log"),
        logging.StreamHandler()
    ])

os.environ["WANDB_MODE"] = "offline"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda:0'
prompt_prefix = "Answer the question : "
dataset_name = 'popQA' # [TQA, popQA, EQ]
completion_template_wo_ans = "Q: {} A:"
completion_template_with_ans = "Q: {} A: {}"
dev_split = 0.1
with_peft = False
with_fs = True
with_rag = True
training_style = 'qa' # ['clm', 'qa']
# target_relation_ids = 'all'
target_relation_ids = ["91"]
# target_relation_ids = ["91", "106", "22", "182"]
file_prefix="bf_rag"

subset_percentage = 0.01
if dataset_name == "TQA":
    num_relations = 1
else:
    num_relations = 15

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

def load_json_files(relation_files, split_name):
    json_data = {}
    for relation_id, files in relation_files.items():
        split_file = next((file for file in files if split_name in file), None)
        with open(split_file, 'r') as file:
            json_data[relation_id] = json.load(file)
    return json_data

def format_example(example, dataset_name):
    if dataset_name in ['EQ', 'popQA']:
        return example['question'], example['answers']
    elif dataset_name == 'TQA':
        return example['Question'], example['Answer']['NormalizedAliases']

def create_few_shot_examples(relation_id, data, num_samples):
    few_shot_examples = []
    sampled_examples = random.sample(data[relation_id], min(num_samples, len(data[relation_id])))
    for example in sampled_examples:
        question, answers = format_example(example, dataset_name)
        completion = completion_template_with_ans.format(question, random.choice(answers))
        few_shot_examples.append(completion)
    return few_shot_examples

def load_relations_data(args):
    
    if dataset_name == "TQA":
        subfolders = ['dev']
    else:
        subfolders = ['test']
        
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = os.path.join(args.data_dir, subfolder)
        for file in os.listdir(subfolder_path):
            relation_id = file.split('.')[0]
            if relation_id not in relation_files:
                relation_files[relation_id] = []
            relation_files[relation_id].append(os.path.join(subfolder_path, file))    

    # Select one relation =================
    # test_relation_id = random.choice(list(relation_files.keys()))
    if target_relation_ids == "all":
        test_relation_ids = ["22", "218", "91", "257", "182", "164", "526", "97", "533", "639", "472", "106", "560", "484", "292", "422"]
    else:
        test_relation_ids = target_relation_ids
    
    test_files = {subfolder: [] for subfolder in subfolders}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(args.data_dir, subfolder)
        for file in os.listdir(subfolder_path):
            file_id = file.split('.')[0]
            if file_id in test_relation_ids:
                test_files[subfolder].append(os.path.join(subfolder_path, file))

    print("Selected Relation ID:", test_relation_ids)
    logging.info(f"Selected Relation ID: {test_relation_ids}")
    # print("Selected Files:", test_files)

    return test_relation_ids, test_files, relation_files

def load_dataset(test_files):
    if dataset_name in ['popQA', 'EQ']:        
        test_data = []
        # test_data = load_json_file(test_files['test'])
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

def load_model(args, with_peft=False):
    if with_peft:
        pass
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            # device_map={"": 0}
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    model.to(device)
    model.eval()
    logging.info("Model and tokenizer are loaded")
    
    return model, tokenizer


def main(args):
    logging.info(f"\\
                Model: {args.model_name_or_path} \n \\
                PEFT: {with_peft} \n \\
                RAG: {with_rag} \n \\
                Few-shot input: {with_fs} \n \\
                output file's prefix: {file_prefix}"
    )
    set_seed(42)
    
    logging.info("Inferencing ...")
    model, tokenizer = load_model(args)
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    test_questions, test_answers = load_dataset(test_files)
    
    # == Create results dir ==================================
    out_results_dir = f"{args.output_result_dir}/results"
    os.makedirs(out_results_dir, exist_ok=True)
    model_name = args.model_name_or_path.split('/')[-1]
    str_rels = "all" if target_relation_ids == "all" else '_'.join(test_relation_ids)
    out_results_path = f"{out_results_dir}/{str_rels}.{model_name}.{file_prefix}_results.jsonl"
    
    if with_rag:
        ret_results = []
        ret_results_dir = f"{args.data_dir}/retrieved"
        
        for test_relation_id in test_relation_ids:
            ret_results_path = f"{ret_results_dir}/{test_relation_id}.ret_results.jsonl"
            with open (ret_results_path, 'r') as file:
                ret_results.extend([json.loads(line) for line in file])

    loaded_json_data = load_json_files(relation_files, 'test' if dataset_name in ['EQ', 'popQA'] else 'dev')
    num_samples_per_relation = 1
    max_new_tokens=15
    accuracy = []
    
    # with open(out_results_path, 'w') as file:
    for idx, (query_id, query, query_pv, query_relation) in enumerate(tqdm(test_questions)):
        if idx == 3:
            break
        
        prompt = prompt_prefix + query
        inpts = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inpts,
                # generation_config=generation_config,
                # input_ids=inpts["input_ids"],
                # attention_mask=inpts["attention_mask"],
                # pad_token_id=tokenizer.eos_token_id,
                # max_new_tokens=max_new_tokens,
                # num_beams=1,
                # do_sample=False
            )
            text = tokenizer.decode(gen[0])
            print(text)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_result_dir", type=str)
    
    args = parser.parse_args()
    main(args)