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
    handlers=[
        # logging.FileHandler("app.log"),
        logging.StreamHandler()
    ])
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

def load_relations_data(args):
    subfolders = ['test']
        
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = os.path.join(args.data_dir, subfolder)
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                relation_id = file.split('.')[0]
                if relation_id not in relation_files:
                    relation_files[relation_id] = []
                relation_files[relation_id].append(os.path.join(subfolder_path, file))    

    # Select one relation =================
    # test_relation_id = random.choice(list(relation_files.keys()))
    if target_relation_ids == "all":
        if args.dataset_name == 'popQA':
            test_relation_ids = ["22", "218", "91", "257", "182", "164", "526", "97", "533", "639", "472", "106", "560", "484", "292", "422"]
        elif args.dataset_name == 'witQA':
            test_relation_ids = ['58', '2936', '17', '1376', '674', '50', '22', '1431', '123', '1038', '3301', '140', '36', '136', '86', '69', '1433', '4647', '25', '452', '2012', '106', '27', '184', '1050', '162', '149', '641', '57', '462', '344', '19']   
        elif args.dataset_name == 'EQ':
            test_relation_ids = ['17', '19', '20', '26', '36', '40', '50', '69', '106', '112', '127', '131', '136', '159', '170', '175', '176', '264', '276', '407', '413', '495', '740', '800']
    
    else:
        test_relation_ids = target_relation_ids
    
    test_files = {subfolder: [] for subfolder in subfolders}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(args.data_dir, subfolder)
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_id = file.split('.')[0]
                if file_id in test_relation_ids:
                    test_files[subfolder].append(os.path.join(subfolder_path, file))

    print("Selected Relation ID:", test_relation_ids)
    logging.info(f"Selected Relation ID: {test_relation_ids}")
    # print("Selected Files:", test_files)

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

def load_model(args):
    if args.with_peft:
        config = PeftConfig.from_pretrained(args.model_name_or_path)
        
        if args.llm_model_name in ["llama2", "mistral", "zephyr", "tiny_llama", "MiniCPM"]:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map={"":0} # Load the entire model on the GPU 0
            )
        elif args.llm_model_name == "flant5":
            model = AutoModelForSeq2SeqLM.from_pretrained(
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
        if args.llm_model_name in ["llama2", "mistral", "zephyr", "tiny_llama", "MiniCPM"]:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map={"":0}, # Load the entire model on the GPU 0
                trust_remote_code=True
            )
        elif args.llm_model_name == "flant5":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                device_map={"": 0}
                # device_map="auto"
            )
            
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True)

    if args.llm_model_name in ["llama2", "mistral", "zephyr", "tiny_llama", "MiniCPM"]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    if args.llm_model_name == 'llama2':
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model.eval()
    logging.info("Model and tokenizer are loaded")
    
    return model, tokenizer

def main(args):
    
    # == Create data & output dir ===========================
    args.data_dir = "component0_preprocessing/generated_data/{}_costomized".format(args.dataset_name)
    args.output_result_dir = "component0_preprocessing/generated_data/{}_costomized".format(args.dataset_name)
    
    # == Create results dir and file ========================
    out_results_dir = f"{args.output_result_dir}/results"
    os.makedirs(out_results_dir, exist_ok=True)
    
    rag_part = "rag" if args.with_rag else "norag"
    peft_part = "peft" if args.with_peft else "full"
    if args.with_rag:
        file_prefix = f"{args.dataset_name}_{args.llm_model_name}_{args.output_file_pre_prefix}_{rag_part}_{args.retrieval_method}_{peft_part}"
    else:
        file_prefix = f"{args.dataset_name}_{args.llm_model_name}_{args.output_file_pre_prefix}_{rag_part}_{peft_part}"
    
    out_results_path = f"{out_results_dir}/{file_prefix}_results.jsonl"
    
    logging.info(f"""
        Model: {args.model_name_or_path}
        Dataset: {args.dataset_name}
        PEFT: {args.with_peft}
        RAG: {args.with_rag}
        Few-shot input: {args.with_fs}
        Retrieval method {args.retrieval_method}
        Output file's prefix: {file_prefix}
        """
    )
    set_seed(42)
    
    ### === Parameters per model
    # 1) Llama2 & tiny_llama & Mistral
    if args.llm_model_name in ["llama2", "tiny_llama", "mistral"]:
        prompt_template_w_context = """
            <s>You are an Answer Generator system. Your goal is to provide one-entity responses to questions, drawing upon either the context provided or your own stored knowledge.\n
            [INST]\n
            Context: {context}\n
            Question: {question}\n
            [/INST]"""
        prompt_template_wo_context = """
            <s>You are an Answer Generator system. Your goal is to provide one-entity responses to questions, drawing upon either the context provided or your own stored knowledge.\n
            [INST]\n
            Question: {question}\n
            [/INST]"""

    # 2) Zephyr
    elif args.llm_model_name == "zephyr":
        prompt_template_w_context = """
            <|system|>You are an Answer Generator system. Your goal is to provide one-entity responses to questions, drawing upon either the context provided or your own stored knowledge.\n
            <|user|>\n 
            Context: {context}\n
            Question: {question}\n
            <|assistant|>\n""" 

        prompt_template_wo_context = """
            <|system|>You are an Answer Generator system. Your goal is to provide one-entity responses to questions, drawing upon either the context provided or your own stored knowledge.\n
            <|user|>\n 
            Question: {question}\n
            <|assistant|>\n""" 
    
    # 3) MiniCPM
    elif args.llm_model_name == "MiniCPM":
        prompt_template_w_context = """
            <User> You are an Answer Generator system. Your goal is to provide one-entity responses to questions, drawing upon either the context provided or your own stored knowledge.\n
            Context: {context}\n
            Question: {question}\n
            <AI>"""
        prompt_template_wo_context = """
            <User> You are an Answer Generator system. Your goal is to provide one-entity responses to questions, drawing upon either the context provided or your own stored knowledge.\n
            Question: {question}\n
            <AI>"""
    
    # 4) FlanT5 family
    elif args.llm_model_name == "flant5":
        prompt_template_w_context = """Context: {context} \n Based on the provided context, answer the question: {question}"""
        prompt_template_wo_context = """Answer the question: {question}"""
        
    
    logging.info("Inferencing ...")
    model, tokenizer = load_model(args)
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    test_questions, test_answers = load_dataset(test_files)
    
    # == Loading the retrieval results =======================
    if args.with_rag:
        ret_results = []
        ret_results_dir = f"{args.data_dir}/retrieved/{args.retrieval_method}"
        
        for test_relation_id in test_relation_ids:
            ret_results_path = f"{ret_results_dir}/{test_relation_id}.{args.retrieval_method}.ret_results.jsonl"
            with open (ret_results_path, 'r') as file:
                ret_results.extend([json.loads(line) for line in file])
    
    # == Loop over the test questions ========================
    accuracy = []
    max_new_tokens = 790 if args.with_rag else 300
    
    if args.llm_model_name in ["llama2", "mistral", "zephyr", "tiny_llama", "MiniCPM"]:
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens = max_new_tokens
        )
    elif args.llm_model_name == "flant5":
        pipe = pipeline( 
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens = max_new_tokens
        ) 
        
    
    with open(out_results_path, 'w') as file:
        for idx, (query_id, query, query_pv, query_relation) in enumerate(tqdm(test_questions)):
            
            # if idx == 10:
            #     break
            
            retrieved_text = ""
            has_context = False
            if args.with_rag:
                for ret_result in ret_results:
                    if ret_result['id'] == query_id:
                        retrieved_text = truncate_text(ret_result['ctxs'][0]['text'], 490) + "\n\n"
                        break
                if retrieved_text == "":
                    logging.info('\n')
                    logging.info(f"No retrieved text found for query: {query}") 
                    print('\n')
                    
                    print("No retrieved text found for query: {}, {}".format(query_id, query))
                    prompt = prompt_template_wo_context.format(question=query)                
                else:
                    prompt = prompt_template_w_context.format(context=retrieved_text, question=query)
                    has_context = True                  
            else:
                prompt = prompt_template_wo_context.format(question=query)                
                    
            
            result = pipe(prompt)[0]['generated_text']
            
            # print("pr: {}".format(prompt))
            # print("rs: {}".format(result))
            
            if args.llm_model_name in ["llama2", "tiny_llama", "mistral"]:
                pred = result.split("[/INST]")[1].strip()
            elif args.llm_model_name == 'zephyr':
                pred = result.split("<|assistant|>")[1].strip()
            elif args.llm_model_name == 'MiniCPM':
                pred = result.split("<AI>")[1].strip()
            elif args.llm_model_name == 'flant5':
                pred = result
            
            is_correct = False
            for pa in test_answers[idx]:
                    if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                        is_correct = True
            accuracy.append(is_correct)
            
            if idx % 300 == 0:
                logging.info('\n')
                logging.info(f"Prompt: {prompt}")
                logging.info(f"Query: {query}")
                logging.info(f"Has context: {has_context}"),
                logging.info(f"Pred: {pred}")
                logging.info(f"Labels: {test_answers[idx]}")
                logging.info(f"Final decision: {is_correct}")
                logging.info('====')
            # print('\n')
            # print(f"Query: {query}")
            # print(f"Has context: {has_context}"),
            # print(f"Pred: {pred}")
            # print(f"Labels: {test_answers[idx]}")
            # print(f"Final decision: {is_correct}")
            # print('====')
            
            item = {
                "query_id": query_id,
                "question": query,
                "possible_answers": test_answers[idx],
                "pred": pred,
                "is_correct": is_correct,
                "has_context": has_context,
                "pageviews": query_pv
            }
            file.write(json.dumps(item) + '\n')
    acc = sum(accuracy) / len(accuracy)
    logging.info(f"Accuracy: {acc * 100:.2f}%")
    print(f"Accuracy: {acc * 100:.2f}%")
    
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
    parser.add_argument("--with_fs", type=str2bool, default=False)
    parser.add_argument("--with_rag", type=str2bool, default=False)
    parser.add_argument("--retrieval_method", type=str)
    
    args = parser.parse_args()
    main(args)
    
    