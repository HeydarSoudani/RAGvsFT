#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import BitsAndBytesConfig
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
dataset_name = 'popQA' # [TQA, popQA, EQ]
completion_template_wo_ans = "Q: {} A:"
completion_template_with_ans = "Q: {} A: {}"
dev_split = 0.1
training_style = 'qa' # ['clm', 'qa']
target_relation_ids = 'all'
# target_relation_ids = ["91"]
# target_relation_ids = ["91", "106", "22", "182"]


subset_percentage = 1.0
num_relations = 1 if dataset_name == "TQA" else 15

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

truncate_text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def truncate_text(text, max_tokens):
    
    tokens = truncate_text_tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    truncated_text = truncate_text_tokenizer.convert_tokens_to_string(tokens)
    return truncated_text
    
def load_model(args):
    
    if args.with_peft:
        
        config = PeftConfig.from_pretrained(args.model_name_or_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            # device_map={"": 0},
            quantization_config=bnb_config
        )
        
        model = PeftModel.from_pretrained(model, args.model_name_or_path, device_map={"":0})
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            # trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            # device_map={"": 0},
        )
        tokenizer = AutoTokenizer.from_pretrained(
           "facebook/opt-350m",
        #    args.model_name_or_path,
           trust_remote_code=True
        )
    model.to(device)
    model.eval()
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # print("Model and tokenizer are loaded")
    logging.info("Model and tokenizer are loaded")
    
    return model, tokenizer

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

def retrieved_file_preparing(
    corpus_file, # json file
    queries_file,# json file
    qrels_file,  # json file
    out_file     # jsonl file
):
    queries = {}
    with open(queries_file, 'r') as qfile:
        data = json.load(qfile)
        for item in data:
            queries[item['query_id']] = item['question']
    
    corpus = {}
    with open(corpus_file, 'r') as cfile:
        data = json.load(cfile)
        for item in data:
            corpus[item['doc_id']] = item['content']
    
    with open(qrels_file, 'r') as qr_file, open(out_file, 'w') as ofile:
        data = json.load(qr_file)
        for idx, item in enumerate(data):
            doc_id = item['doc_id']
            query_id = item['query_id']
            context = corpus.get(doc_id, "No context found")
            question = queries.get(query_id, "No question found")
            
            combined_obj = {
                "id": query_id,
                "question": question,
                "ctxs": [{
                    "id": doc_id,
                    "text": context,
                    "hasanswer": True if item['score'] == 1 else False
                }],
            }
            ofile.write(json.dumps(combined_obj) + "\n")

def main(args):
    
    pre_prefix = "af"
    rag_part = "rag" if args.with_rag else "norag"
    peft_part = "peft" if args.with_peft else "full"
    if args.with_rag:
        file_prefix = f"{pre_prefix}_{rag_part}_{args.retrieval_method}_{peft_part}"
    else:
        file_prefix = f"{pre_prefix}_{rag_part}_{peft_part}"
    
    logging.info(f"""
        Model: {args.model_name_or_path}
        PEFT: {args.with_peft}
        RAG: {args.with_rag}
        Few-shot input: {args.with_fs}
        Retrieval method: {args.retrieval_method}
        Output file's prefix: {file_prefix}
        """
    )    
    set_seed(42)

    # == Creating retrieval files -> Only for the first time
    corpus_dir = f"{args.data_dir}/corpus"
    queries_dir = f"{args.data_dir}/test"
    qrels_dir = f"{args.data_dir}/qrels"
    output_dir = f"{args.data_dir}/retrieved"
    os.makedirs(output_dir, exist_ok=True)
        
    # for qrels_file_name in os.listdir(qrels_dir):
    #     relation_id = qrels_file_name.split('.')[0]
    #     qrels_file = f"{qrels_dir}/{qrels_file_name}"
    #     corpus_file = f"{corpus_dir}/{relation_id}.corpus.json"
    #     queries_file = f"{queries_dir}/{relation_id}.test.json"
    #     out_file = f"{output_dir}/{relation_id}.ret_results.jsonl"
    
    #     retrieved_file_preparing(
    #         corpus_file,
    #         queries_file,
    #         qrels_file,
    #         out_file
    #     )
    ### ====================================================== 
    
    model, tokenizer = load_model(args)
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    test_questions, test_answers = load_dataset(test_files)
    
    logging.info("Inferencing ...")
    
    # == Create results dir ==================================
    out_results_dir = f"{args.output_result_dir}/results"
    os.makedirs(out_results_dir, exist_ok=True)
    model_name = args.model_name_or_path.split('/')[-1]
    str_rels = "all" if target_relation_ids == "all" else '_'.join(test_relation_ids)
    out_results_path = f"{out_results_dir}/{str_rels}.{model_name}.{file_prefix}_results.jsonl"

    if args.with_rag:
        ret_results = []
        ret_results_dir = f"{args.data_dir}/retrieved/{args.retrieval_method}"
        
        for test_relation_id in test_relation_ids:
            ret_results_path = f"{ret_results_dir}/{test_relation_id}.{args.retrieval_method}.ret_results.jsonl"
            with open (ret_results_path, 'r') as file:
                ret_results.extend([json.loads(line) for line in file])

    loaded_json_data = load_json_files(relation_files, 'test' if dataset_name in ['EQ', 'popQA'] else 'dev')
    num_samples_per_relation = 1
    max_new_tokens=15
    accuracy = []
    
    generation_config = GenerationConfig(
        num_beams=4,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        early_stopping=True,
        decoder_start_token_id=0,
        # eos_token_id=tokenizer.eos_token_id,
        # pad_token=model.config.pad_token_id,
        # num_beams=4,
    )
    
    with open(out_results_path, 'w') as file:
        for idx, (query_id, query, query_pv, query_relation) in enumerate(tqdm(test_questions)):
            
            # if idx == 3:
            #     break
            # print(f"Q: {query} is processing ...")
            # logging.info(f"Q: {query} is processing ...")
            
            few_shot_examples_text = ""
            if args.with_fs:
                few_shot_examples = []
                keys_to_sample = [key for key in relation_files.keys() if key != query_relation]
                fewshot_relations = random.sample(keys_to_sample, num_relations)
                for relation_id in fewshot_relations:
                    sampled_examples = random.sample(loaded_json_data[relation_id], min(num_samples_per_relation, len(loaded_json_data[relation_id])))
                    for example in sampled_examples:
                        question, answers = format_example(example, dataset_name)
                        completion = completion_template_with_ans.format(question, random.choice(answers))
                        few_shot_examples.append(completion)
                random.shuffle(few_shot_examples)
                few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"
                
            retrieved_text = ""
            if args.with_rag:
                for ret_result in ret_results:
                    if ret_result['id'] == query_id:
                        retrieved_text = truncate_text(ret_result['ctxs'][0]['text'], 512) + "\n\n"
                        break
                if retrieved_text == "":
                    logging.info(f"No retrieved text found for query: {query}")
                    # print("No retrieved text found for query: {}".format(query))
            
            prompt = few_shot_examples_text + retrieved_text + completion_template_wo_ans.format(query)
            inpts = tokenizer(prompt, return_tensors="pt").to(device)
            # inpt_decoded = tokenizer.decode(inpts["input_ids"][0, :])
            # print(f"Prompt: {prompt}")
            
            with torch.no_grad():
                gen = model.generate(
                    **inpts,
                    generation_config=generation_config,
                    # input_ids=inpts["input_ids"],
                    # attention_mask=inpts["attention_mask"],
                    # pad_token_id=tokenizer.eos_token_id,
                    # max_new_tokens=max_new_tokens,
                    # num_beams=1,
                    # do_sample=False
                )
            text = tokenizer.decode(gen[0])
            pred = text[2+len(prompt):]
            pred = pred.split("\n")[0]
            
            is_correct = False
            for pa in test_answers[idx]:
                if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                    is_correct = True
            accuracy.append(is_correct)
            
            if idx % 500 == 0:
                logging.info('\n')
                logging.info(f"Query: {query}")
                logging.info(f"Pred: {pred}")
                logging.info(f"Labels: {test_answers[idx]}")
                logging.info(f"Final decision: {is_correct}")
                logging.info('====')
            
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
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_result_dir", type=str)
    parser.add_argument("--with_peft", type=str2bool, default=False)
    parser.add_argument("--with_fs", type=str2bool, default=False)
    parser.add_argument("--with_rag", type=str2bool, default=False)
    parser.add_argument("--retrieval_method", type=str)
    
    args = parser.parse_args()
    main(args)