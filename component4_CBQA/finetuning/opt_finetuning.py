#!/usr/bin/env python3

import random
import requests
from io import BytesIO
from zipfile import ZipFile
from datasets import Dataset, DatasetDict
import argparse, os, json
from itertools import chain
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
import evaluate
import logging

from huggingface_hub import HfFolder, HfApi

import numpy as np
import torch

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[
        # logging.FileHandler("app.log"),
        logging.StreamHandler()
    ])

os.environ["WANDB_MODE"] = "offline"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Available GPUs:", torch.cuda.device_count())
device = 'cuda:0'
completion_template_wo_ans = "Q: {} A:"
completion_template_with_ans = "Q: {} A: {}"
dataset_name = 'popQA' # [TQA, popQA, EQ]
training_style = 'qa' # ['clm', 'qa']
target_relation_ids = 'all'
# target_relation_ids = ["91"]
# target_relation_ids = ["91", "106", "22", "182"]
subset_percentage = 1.0
num_relations = 1 if dataset_name == "TQA" else 15
generation_method = "prompting" # ["pipeline", "prompting"]

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

def load_model(args):
    
    if not args.with_peft:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            # device_map={"": 0},
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            # device_map={"": 0},
        )
        model.config.use_cache = False # From ft llama with qlora ...
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        
        def print_trainable_parameters(model):
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )

        # v1: From ft llama with qlora ...
        # alpha = 16
        # dropout = 0.1
        # r = 64
        
        # v2
        # lora_alpha = 32
        # lora_dropout = 0.05
        # r = 8 

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)

        print_trainable_parameters(model)
        model.print_trainable_parameters()
        
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'left'
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False
    )
    
    return model, tokenizer, data_collator

def load_relations_data(args):
    
    # If you need to download EQ or TQA dataset
    if not os.path.isdir(args.data_dir):
        if dataset_name == "TQA":
            url = "https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz"     # For TQA dataset
        
        url = "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip" # For EQ dataset
        response = requests.get(url)
        zip_file = ZipFile(BytesIO(response.content))
        zip_file.extractall("entity_questions_dataset")
        zip_file.close()
    
    subfolders = ['train', 'dev', 'test'] 
    relation_files = {}
    for subfolder in subfolders:
        # subfolder_path = os.path.join(args.data_dir, subfolder)
        subfolder_path = f"{args.data_dir}/{generation_method}/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                relation_id = file.split('.')[0]
                if relation_id not in relation_files:
                    relation_files[relation_id] = []
                relation_files[relation_id].append(os.path.join(subfolder_path, file))    

    # Select one relation =================
    if target_relation_ids == "all":
        test_relation_ids = ["22", "218", "91", "257", "182", "164", "526", "97", "533", "639", "472", "106", "560", "484", "292", "422"]
    else:
        test_relation_ids = target_relation_ids
    
    test_files = {subfolder: [] for subfolder in subfolders}
    
    for subfolder in subfolders:
        # subfolder_path = os.path.join(args.data_dir, subfolder)
        subfolder_path = f"{args.data_dir}/{generation_method}/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_id = file.split('.')[0]
                if file_id in test_relation_ids:
                    test_files[subfolder].append(os.path.join(subfolder_path, file))

    print("Selected Relation ID:", test_relation_ids)
    print("Selected Files:", test_files)

    return test_relation_ids, test_files, relation_files
    
def load_dataset_qa(tokenizer, test_files):
    
    train_data = []
    for file in test_files['train']:
    # for file in test_files['test']:
        train_data.extend(load_json_file(file))    
    dev_data = []
    for file in test_files['dev']:
    # for file in test_files['test']:
        dev_data.extend(load_json_file(file))    
    
    # train_data = load_json_file(test_files['train'])
    # dev_data = load_json_file(test_files['dev'])

    train_subset_size = int(subset_percentage * len(train_data))
    subset_train_data = random.sample(train_data, train_subset_size)
    # dev_subset_size = int(subset_percentage * len(dev_data))
    dev_subset_size = int(subset_percentage * len(dev_data))
    subset_dev_data = random.sample(dev_data, dev_subset_size)

    if dataset_name in ['EQ', 'popQA']:
      train_questions = [item['question'] for item in subset_train_data]
      train_answers = [item['answers'] for item in subset_train_data]
      val_questions = [item['question'] for item in subset_dev_data]
      val_answers = [item['answers'] for item in subset_dev_data]
    else:
      train_questions = [item['Question'] for item in subset_train_data]
      train_answers = [item['Answer']['NormalizedAliases'] for item in subset_train_data]
      val_questions = [item['Question'] for item in subset_dev_data]
      val_answers = [item['Answer']['NormalizedAliases'] for item in subset_dev_data]
    
    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "question": train_questions,
            "possible_answers": train_answers
        }),
        'dev': Dataset.from_dict({
            "question": val_questions,
            "possible_answers": val_answers
        })
    })
    print(raw_dataset)
    
    def qa_tokenize_function(examples):
        input_prompts = []
        len_dataset = len(examples['question'])
    
        for idx in range(len_dataset):
            input_prompts.extend(
                [completion_template_with_ans.format(examples['question'][idx], pa) for pa in examples['possible_answers'][idx]]
            )    
        model_inputs = tokenizer(input_prompts)
        return model_inputs
    
    tokenized_train_datasets = raw_dataset.map(
        qa_tokenize_function,
        batched=True,
        remove_columns=["question", "possible_answers"],
        desc="Running tokenizer on dataset",
    )
    print(tokenized_train_datasets)
    
    # === Print a sample of dataset
    input_text = tokenizer.decode(
        tokenized_train_datasets['train'][0]["input_ids"],
        skip_special_tokens=True
    )
    print(input_text)
    # label_text = tokenizer.decode(tokenized_train_datasets['train'][0]["labels"], skip_special_tokens=True)
    # print(label_text)
    
    return tokenized_train_datasets

def load_dataset_corpus(tokenizer, test_relation_id, test_files):
    subset_percentage = 1.0
    max_pos_embeddings = 1024
    
    corpus_path = f"{args.data_dir}/corpus/{test_relation_id}.corpus.json"
    with open(corpus_path, 'r') as in_file:
        corpus_data = json.load(in_file)
    corpus_content_data = [item["content"] for item in corpus_data]
    
    random.shuffle(corpus_content_data)
    split_index = int(len(corpus_content_data) * dev_split)
    dev_content = corpus_content_data[:split_index]
    train_content = corpus_content_data[split_index:]
    
    train_subset_size = int(subset_percentage * len(train_content))
    subset_train_data = random.sample(train_content, train_subset_size)
    dev_subset_size = int(subset_percentage * len(dev_content))
    subset_dev_data = random.sample(dev_content, dev_subset_size)
    
    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "text": subset_train_data,
        }),
        'dev': Dataset.from_dict({
            "text": subset_dev_data,
        })
    })
    print(raw_dataset)
    
    column_names = list(raw_dataset["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    
    block_size = tokenizer.model_max_length
    if block_size > max_pos_embeddings:
        print(
            f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
        )
        if max_pos_embeddings > 0:
            block_size = min(1024, max_pos_embeddings)
        else:
            block_size = 1024
    
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
    
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_train_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    
    ### === Test part =================================         
    if dataset_name in ['popQA', 'EQ']:  
        test_data = load_json_file(test_files['test'])
        test_subset_size = int(subset_percentage * len(test_data))
        subset_test_data = random.sample(test_data, test_subset_size)    
        test_questions = [(item['query_id'], item['question'], item['pageviews']) for item in subset_test_data]
        test_answers = [item['answers'] for item in subset_test_data]
    
    return tokenized_train_datasets, (test_questions, test_answers)

def load_training_args(args):
    output_dir = os.path.join(args.output_model_dir, args.repo_name.split('/')[-1])
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit", # "paged_adamw_8bit"
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.lr,
        max_grad_norm=0.3,
        warmup_ratio=0,
        lr_scheduler_type="linear",
        report_to=[],
        
        fp16=True,

        save_strategy="epoch",
        save_total_limit=4,
        # save_steps=1,
        # logging_steps=1,
        # fp16=True,
        # max_steps=200,
        # load_best_model_at_end=False,
    )
    return training_arguments

def main(args):
    logging.info(f"""
        Model: {args.model_name_or_path}
        PEFT: {args.with_peft}
        version: {args.version}
    """)
    args.repo_name = "HeydarS/{}_{}_v{}".format(
        args.model_name_or_path.split('/')[-1],
        'peft' if args.with_peft else 'full',
        args.version
    )
    
    set_seed(42)
    model, tokenizer, data_collator = load_model(args)
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    if training_style == 'qa':
        tokenized_train_datasets = load_dataset_qa(tokenizer, test_files)
    elif training_style == 'clm':
        tokenized_train_datasets = load_dataset_corpus(
            tokenizer,
            test_relation_ids,
            test_files
        )
    training_arguments = load_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_train_datasets["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    
    print("Fine-tuning ....")
    save_model_dir = os.path.join(args.output_model_dir, args.repo_name.split('/')[-1])
    trainer.train()
    model.save_pretrained(save_model_dir)
    model.push_to_hub(args.repo_name, token=True)
    print("Fine-tuning is done.")

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
    parser.add_argument("--output_model_dir", type=str)
    parser.add_argument("--output_result_dir", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--with_peft", type=str2bool, default=False)
    parser.add_argument("--version", default=1, type=int)
    
    args = parser.parse_args()
    main(args)
