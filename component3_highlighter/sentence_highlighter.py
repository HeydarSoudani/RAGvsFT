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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)
os.environ["WANDB_MODE"] = "offline"

target_relation_ids = 'all'

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

def load_model(args):
    
    if args.llm_model_name in ["llama3", "llama2", "mistral", "zephyr", "stable_lm2", "tiny_llama", "MiniCPM"]:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map={"":0}, # Load the entire model on the GPU 0
            # device_map='auto',
            trust_remote_code=True
        )
    elif args.llm_model_name == "flant5":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            # load_in_8bit=True,
            # device_map={"": 0}
            # device_map="auto"
        )
        
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    if args.llm_model_name in ["llama2", "mistral", "zephyr", "stable_lm2", "tiny_llama", "MiniCPM"]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    model.eval()
    logging.info("Model and tokenizer are loaded")
    
    return model, tokenizer

def main(args):
    logging.info(f"""
        Model: {args.model_name_or_path}
        Dataset: {args.dataset_name}
        Retrieval method: {args.retrieval_method}
        Seed: {args.seed}
        """
    )
    print(f"""
        Model: {args.model_name_or_path}
        Dataset: {args.dataset_name}
        Retrieval method: {args.retrieval_method}
        Seed: {args.seed}
        """
    )
    set_seed(args.seed)
    
    # == Create data & output dir ===========================
    args.data_dir = f"component0_preprocessing/generated_data/{args.dataset_name}_costomized"
    retrieved_passage_dir = f"{args.data_dir}/retrieved_passage/{args.retrieval_method}_3"
    highlighted_sentences_dir = f"{args.data_dir}/highlighted_sentences/{args.retrieval_method}_3"
    os.makedirs(f'{args.data_dir}/highlighted_sentences', exist_ok=True)
    os.makedirs(highlighted_sentences_dir, exist_ok=True)
    
    # === Prompt Definition ===
    prompt_highlight_generation = lambda query, context: f"""
        Example output: {{"relation": "",“sentences”: []}}
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
    
    if args.llm_model_name == "zephyr":
        prompt_template = """<|system|> </s>\n <|user|>{context}</s>\n <|assistant|>"""
    elif args.llm_model_name == "llama3":
        prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>{context}<|eot_id|>"""
        # <|start_header_id|>assistant<|end_header_id|>{{ model_answer_1 }}<|eot_id|>
    
    # === Load model and tokenizer ==========================
    model, tokenizer = load_model(args)
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    max_input_tokens = 2048
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # === Retrieved context =================================
    # === Loop over relations ===============================
    # for test_relation_id in test_relation_ids:
    test_relation_id = test_relation_ids[11]
    
    ret_results_path = f"{retrieved_passage_dir}/{test_relation_id}.{args.retrieval_method}.ret_results.jsonl"
    output_file = f"{highlighted_sentences_dir}/{test_relation_id}.{args.retrieval_method}.set_highlighted.jsonl"
    logging.info(f"Processing {test_relation_id}...")
    # print(f"Processing {test_relation_id}...")
    
    rel_data = []
    with open (ret_results_path, 'r') as in_file:
        for idx, line in enumerate(tqdm(in_file)):
            data = json.loads(line.strip())
            rel_data.append(data)
            
    with open(output_file, 'w') as out_file:
        for idx, item in enumerate(tqdm(rel_data)):
            
            # if idx == 5:
            #     break
            
            # data = json.loads(line.strip())
            query_id, query, ctxs = item['id'], item['question'], item['ctxs']
            
            retrieved_text = ""
            max_token = max_input_tokens - 50
            
            if args.passage_concatenation == "concatenate":
                
                corpus_text = "".join(ctxs[i]['text'] for i in range(args.num_retrieved_passages) if i < len(ctxs))
                retrieved_text = truncate_text(corpus_text, max_token)  
                if retrieved_text != "":
                    prompt = prompt_template.format(context=prompt_highlight_generation(query=query, context=retrieved_text))
                    n_max_trial = 5
                    for i in range(n_max_trial):
                        try:
                            result = pipe(prompt, max_new_tokens=1024)[0]['generated_text']
                            break
                        except Exception as e:
                            print(f"Try #{i+1} for Query: {query_id}")
                            print('Error message:', e)
                    
                    if args.llm_model_name in ['zephyr']:
                        pred = result.split("<|assistant|>")[1].strip()
                    elif args.llm_model_name in ['llama3']:
                        pred = result[len(prompt):]
                    
                    pred = _extract_json_part(pred)
                    
                    if idx < 3 or idx % 200 == 0:
                        logging.info('\n')
                        logging.info(f"Prompt: {prompt}")
                        logging.info(f"Query: {query}")
                        logging.info(f"highlighted sentence: {pred}"),
                        logging.info('====')
                        # print('\n')
                        # print(f"Prompt: {prompt}")
                        # print(f"Query: {query}")
                        # print(f"highlighted passage: {pred}"),
                        # print('====')
                    
                    item = {
                        "query_id": query_id,
                        "question": query,
                        "sentences": [{
                            "ret_rank": 1,
                            "sentence": pred
                        }],
                    }
                    out_file.write(json.dumps(item) + '\n')
                
                else:
                    logging.info(f"\nNo retrieved text found for query: {query}") 
                    print("\nNo retrieved text found for query: {}, {}".format(query_id, query))

            elif args.passage_concatenation == "separate":
                
                highlighted_passages = []
                for ret_idx in range(args.num_retrieved_passages):
                    if ret_idx < len(ctxs):
                        retrieved_text = truncate_text(ctxs[ret_idx]['text'], max_token)
                        if retrieved_text != "":
                            prompt = prompt_template.format(context=prompt_highlight_generation(query=query, context=retrieved_text))
                            
                            n_max_trial = 5
                            result = None
                            for i in range(n_max_trial):
                                try:
                                    result = pipe(prompt, max_new_tokens=1024)[0]['generated_text']
                                    break
                                except Exception as e:
                                    print(f"Try #{i+1} for Query: {query_id}")
                                    print('Error message:', e)
                            
                            if result != None:
                            
                                if args.llm_model_name in ['zephyr']:
                                    pred = result.split("<|assistant|>")[1].strip()
                                elif args.llm_model_name in ['llama3']:
                                    pred = result[len(prompt):]
                                pred = _extract_json_part(pred)
                                
                                if idx < 2 or idx % 200 == 0:
                                    logging.info('\n')
                                    logging.info(f"Prompt: {prompt}")
                                    logging.info(f"Query: {query}")
                                    logging.info(f"highlighted sentence: {pred}"),
                                    logging.info('====')
                                    # print('\n')
                                    # print(f"Prompt: {prompt}")
                                    # print(f"Query: {query}")
                                    # print(f"highlighted passage: {pred}"),
                                    # print('====')
                                
                                highlighted_passages.append({
                                    "ret_rank": ret_idx,
                                    "sentence": pred
                                })
                            
                            else:
                                highlighted_passages.append({
                                    "ret_rank": ret_idx,
                                    "sentence": ""
                                })
                        
                        else:
                            logging.info(f"\nNo retrieved text found for query: {query_id}, {query}, Ret_rank: {ret_idx}") 
                            print("\nNo retrieved text found for query: {}, {}, Ret_rank: {}".format(query_id, query, ret_idx))
                
                item = {
                    "query_id": query_id,
                    "question": query,
                    "sentences": highlighted_passages,
                }
                out_file.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--llm_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--num_retrieved_passages", type=int, default=1)
    parser.add_argument("--passage_concatenation", type=str, default="concatenate", choices=["concatenate", "separate"])
    parser.add_argument("--retrieval_method", type=str, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    
    main(args)