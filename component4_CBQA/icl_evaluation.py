#!/usr/bin/env python3

import argparse, os, json
import numpy as np
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM


os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda:0'
dataset_name = 'popQA' # [TQA, popQA, EQ]
completion_template_wo_ans = "Q: {} A:"
completion_template_with_ans = "Q: {} A: {}"
with_peft = False
with_fs = True

def set_seed(seed):
    """Set the seed for reproducibility in PyTorch, NumPy, and Python."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={"": 0},
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    return model, tokenizer

def load_relations_data(args):
    
    if dataset_name == "TQA":
        num_relations = 1
        subfolders = ['dev']
    else:
        num_relations = 15
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
    # selected_relation_id = random.choice(list(relation_files.keys()))
    selected_relation_id = "106"
    selected_files = {}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(args.data_dir, subfolder)
        for file in os.listdir(subfolder_path):
            if file.startswith(selected_relation_id):
                selected_files[subfolder] = os.path.join(subfolder_path, file)

    print("Selected Relation ID:", selected_relation_id)
    print("Selected Files:", selected_files)
    
    # Other relations ===============
    if dataset_name in ['EQ', 'popQA']:
        relation_files.pop(selected_relation_id)
    
    selected_relations = random.sample(relation_files.keys(), num_relations)

    return relation_files, selected_relations, selected_files

def load_dataset(selected_files):
    subset_percentage = 1.0
    
    if dataset_name in ['popQA', 'EQ']:        
        test_data = load_json_file(selected_files['test'])
        test_subset_size = int(subset_percentage * len(test_data))
        subset_test_data = random.sample(test_data, test_subset_size)    
        test_questions = [(item['query_id'], item['question'], item['pageviews']) for item in subset_test_data]
        test_answers = [item['answers'] for item in subset_test_data]
    
        return test_questions, test_answers

def create_few_shot_examples(
    relation_files,
    selected_relations,
    num_samples,
    split_name
):
    few_shot_examples = []
    
    for relation_id in selected_relations:
        files = relation_files[relation_id]
        split_file = next((file for file in files if split_name in file), None)
        data = load_json_file(split_file)
        
        sampled_examples = random.sample(data, min(num_samples, len(data)))
        for example in sampled_examples:
            
            if dataset_name in ['EQ', 'popQA']:
                question = example['question']
                answers = example['answers']
            elif dataset_name == 'TQA':
                question = example['Question']
                answers = example['Answer']['NormalizedAliases']
            
            completion = completion_template_with_ans.format(question, random.choice(answers))
            few_shot_examples.append(completion)
    return few_shot_examples

def inference_on_testset(
    model,
    tokenizer,
    test_questions,
    test_answers,
    relation_files,
    selected_relations,
    device,
    args,
    with_fs,
    prefix="bf"
):
    results_dir = f"{args.data_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    result_file_path = f"{results_dir}/106.{args.model_name_or_path.split('/')[-1]}.{prefix}_results.jsonl"
    
    num_samples_per_relation = 1
    model.eval()
    max_new_tokens=15
    accuracy = []
    
    with open(result_file_path, 'w') as file:
    
        for idx, (query_id, query, query_pv) in enumerate(test_questions):
                    
            if with_fs:
                few_shot_examples = create_few_shot_examples(
                    relation_files,
                    selected_relations,
                    num_samples_per_relation,
                    'test' if dataset_name in ['EQ', 'popQA'] else 'dev'
                )
                np.random.shuffle(few_shot_examples)
                few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"
            else:
                few_shot_examples_text = "\n\n"
            
            retrieved_text = ""
            
            prompt = few_shot_examples_text + retrieved_text + "\n\n" + completion_template_wo_ans.format(query)
            
            inpts = tokenizer(prompt, return_tensors="pt").to(device)
            inpt_decoded = tokenizer.decode(inpts["input_ids"][0, :])
            
            with torch.no_grad():
                gen = model.generate(
                    input_ids=inpts["input_ids"],
                    attention_mask=inpts["attention_mask"],
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
            text = tokenizer.decode(gen[0])
            
            # print(text)
            
            pred = text[2+len(prompt):]
            # pred = pred[5:]
            pred = pred.split("\n")[0]

            # if idx % 15 == 0:
            print('Query: {}'.format(query))
            print('Pred: {}'.format(pred))
            print('Labels: {}'.format(test_answers[idx]))
                # print('\n\n')
            
            is_correct = False
            for pa in test_answers[idx]:
                if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                    is_correct = True
            accuracy.append(is_correct)
            print('Final decision: {}'.format(is_correct))
            print('====')
            
            # Write to file
            item = {
                "query_id": query_id,
                "question": query,
                "possible_answers": test_answers[idx],
                "pred": pred,
                "is_correct": is_correct,
                "pageviews": query_pv
            }
            file.write(json.dumps(item) + '\n')

        acc = sum(accuracy) / len(accuracy)
        print(f"Accuracy: {acc * 100:.2f}%")


def data_preparing():
    pass

def main(args):
    
    set_seed(42)
    model, tokenizer = load_model(args)
    
    relation_files, selected_relations, selected_files = load_relations_data(args)
    test_questions, test_answers = load_dataset(selected_files)
    
    inference_on_testset(
        model,
        tokenizer,
        test_questions,
        test_answers,
        relation_files,
        selected_relations,
        device,
        args,
        with_fs,
        prefix="bf"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    
    args = parser.parse_args()
    main(args)