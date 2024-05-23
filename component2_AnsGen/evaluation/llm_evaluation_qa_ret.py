#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import pipeline
from peft import PeftConfig, PeftModel
import argparse, os, json
import numpy as np
import random
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)
os.environ["WANDB_MODE"] = "offline"

print("Available GPUs:", torch.cuda.device_count())
device = 'cuda:0'
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

truncate_text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def truncate_text(text, max_tokens):
    tokens = truncate_text_tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    truncated_text = truncate_text_tokenizer.convert_tokens_to_string(tokens)
    return truncated_text

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
    if args.with_peft:
        config = PeftConfig.from_pretrained(args.model_name_or_path)
        
        if args.llm_model_name in ["llama2", "mistral", "zephyr", "stable_lm2", "tiny_llama", "MiniCPM"]:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map={"":0},
                trust_remote_code=True
            )
        elif args.llm_model_name == "flant5":
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                config.base_model_name_or_path,
                load_in_8bit=True,
                # device_map={"":0}
            )
            
        model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
        model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True
        )
        
    else:
        if args.llm_model_name in ["llama2", "mistral", "zephyr", "stable_lm2", "tiny_llama", "MiniCPM"]:
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
    # if args.llm_model_name == 'llama2':
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model.eval()
    logging.info("Model and tokenizer are loaded")
    
    return model, tokenizer


def main(args):
    args.data_dir = f"component0_preprocessing/generated_data/{args.dataset_name}_costomized"
    
    set_seed(args.seed)
    prompt_template_w_context = """<User>\n Context: {context}\n Question: {question}\n <AI>"""
    prompt_template_wo_context = """<User>\n Question: {question}\n <AI>"""
    
    model, tokenizer = load_model(args)
    test_relation_ids, test_files, relation_files = load_relations_data(args)

    # == Loading the retrieval results (corpus) ==============
    if args.with_rag:
        ret_results = []
        ret_results_dir = f"{args.data_dir}/retrieved/{args.retrieval_method}"
        
        for test_relation_id in test_relation_ids:
            ret_results_path = f"{ret_results_dir}/{test_relation_id}.{args.retrieval_method}.ret_results.jsonl"
            with open (ret_results_path, 'r') as file:
                ret_results.extend([json.loads(line) for line in file])

    # == Loading the retrieval results (qa_pairs) ============
    data_path = 'component0_preprocessing/generated_data/popQA_costomized/retrieved_qa_pairs/ideal/526.retrieved_qa_pairs.jsonl'
    qa_pairs = {}
    with open(data_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            qa_pairs[data['query_id']] = data

    # === Load data ===
    relation_id = '526'
    relation_name = "Director"
    with open('analysis/on_false_results/compare_ft_wo_ft.json', 'r') as file:
        data = json.load(file)
    test_data = data[relation_name]
    
    max_new_tokens = 40
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens = max_new_tokens
    )
    
    for idx, test_item in enumerate(tqdm(test_data)):
        
        if idx == 20:
            break
        
        query_id = test_item['query_id']
        query = test_item['question']
        
        retrieved_text = ""
        has_context = False
        if args.with_rag:
            for ret_result in ret_results:
                if ret_result['id'] == query_id:
                    retrieved_text = truncate_text(ret_result['ctxs'][0]['text'], 490)
                    break
            if retrieved_text == "":
                logging.info('\n')
                logging.info(f"No retrieved text found for query: {query}") 
                print('\n')
                
                print("No retrieved text found for query: {}, {}".format(query_id, query))
                prompt = prompt_template_wo_context.format(question=query)                
            else:
                
                qa_pairs_data = qa_pairs[query_id]['relevant_train_questions']
                qa_pairs_text = ""
                if len(qa_pairs_data) > 0:
                    for qa_pair in qa_pairs_data:
                        qa_pairs_text += f"{qa_pair['question']} {qa_pair['answers'][0]}\n"
                
                retrieved_text += f"\n{qa_pairs_text}"
                prompt = prompt_template_w_context.format(context=retrieved_text, question=query)        
            
        else:
            prompt = prompt_template_wo_context.format(question=query)
        

        result = pipe(prompt)[0]['generated_text']
        pred = result.split("<AI>")[1].strip()
        print('\n')
        print(f"Prompt: {prompt}")
        print(f"Question: {query}")
        print(f"Prediction: {pred}")
     


if __name__ == "__main__":
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--llm_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_file_pre_prefix", type=str)
    parser.add_argument("--with_peft", type=str2bool, default=False)
    parser.add_argument("--with_rag", type=str2bool, default=False)
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()
    main(args)