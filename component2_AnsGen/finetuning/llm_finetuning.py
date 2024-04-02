#!/usr/bin/env python3

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets
from huggingface_hub import notebook_login
from huggingface_hub import HfFolder
# import evaluate

import logging
import os, json, argparse
import numpy as np
import random
import nltk

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[
        # logging.FileHandler("app.log"),
        logging.StreamHandler()
    ])

os.environ["WANDB_MODE"] = "offline"
nltk.download("punkt", quiet=True)

print("Available GPUs:", torch.cuda.device_count())
device = 'cuda:0'
dataset_name = 'popQA' # [TQA, popQA, EQ]
target_relation_ids = 'all'
subset_percentage = 1.0
num_relations = 1 if dataset_name == "TQA" else 15

### === Parameters per model
# 1) Llama
batch_size = 32
fp16 = True
bf16 = False
prompt_template = """<s>
        You are an Answer Generator system. Your goal is to provide concise responses to questions, drawing upon either the context provided or your own stored knowledge.\n
        [INST]\n
        Question: {}\n
        [/INST]\n
        Answer: {}
        </s>"""

# 2) Mistral
batch_size = 4
fp16=False
bf16=False
prompt_template = """<s>
        You are an Answer Generator system. Your goal is to provide concise responses to questions, drawing upon either the context provided or your own stored knowledge.\n
        [INST]\n
        Question: {}\n
        [/INST]\n
        Answer: {}
        </s>"""

# 3) Zephyr
batch_size = 4
fp16=False
bf16=False
prompt_template = """<|system|>\n
        You are an Answer Generator system. Your goal is to provide concise responses to questions, drawing upon either the context provided or your own stored knowledge.\n
        <|user|>\n 
        Question: {}\n
        <|assistant|>\n
        Answer: {}
        """


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
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
            device_map={"": 0}
        )
        model.config.use_cache = False
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            device_map={"": 0}
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        
        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64
        if args.llm_model_name == 'llama':
            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM"
            )
        elif args.llm_model_name == 'mistral':
            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                r=lora_r,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.llm_model_name == 'mistral':
        tokenizer.padding_side = "right"
    
    return model, tokenizer, peft_config

def load_relations_data(args): 
    subfolders = ['train', 'dev', 'test']    
    relation_files = {}
    
    for subfolder in subfolders:
        # subfolder_path = os.path.join(args.data_dir, subfolder)
        subfolder_path = f"{args.data_dir}/{args.generation_method}/{subfolder}"
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
        subfolder_path = f"{args.data_dir}/{args.generation_method}/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_id = file.split('.')[0]
                if file_id in test_relation_ids:
                    test_files[subfolder].append(os.path.join(subfolder_path, file))

    print("Selected Relation ID:", test_relation_ids)
    print("Selected Files:", test_files)

    return test_relation_ids, test_files, relation_files     

def load_dataset_qa(test_files):
    train_data = []
    for file in test_files['train']:
    # for file in test_files['test']:
        train_data.extend(load_json_file(file))    
    dev_data = []
    for file in test_files['dev']:
    # for file in test_files['test']:
        dev_data.extend(load_json_file(file)) 
    
    train_subset_size = int(subset_percentage * len(train_data))
    subset_train_data = random.sample(train_data, train_subset_size)
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

    train_data = [
        prompt_template.format(question, train_answers[i])
        for i, question in enumerate(train_questions)
    ]
    
    val_data = [
        prompt_template.format(question, val_answers[i])
        for i, question in enumerate(val_questions)
    ]

    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "text": train_data
        }),
        'dev': Dataset.from_dict({
            "text": val_data
        })
    })
    print(raw_dataset)
    return raw_dataset

def load_training_args(args):
    save_model_dir = os.path.join(args.output_model_dir, args.repo_name.split('/')[-1])
        
    training_arguments = TrainingArguments(
        output_dir=save_model_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=fp16,
        bf16=fp16,
        
        report_to="wandb",
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id=args.repo_name.split('/')[-1],
        hub_token=HfFolder.get_token()
    )
    
    return training_arguments

def main(args):
    logging.info(f"""
        Model: {args.model_name_or_path}
        PEFT: {args.with_peft}
        Version: {args.version}
    """)
    args.repo_name = "HeydarS/{}_{}_v{}".format(
        args.model_name_or_path.split('/')[-1],
        'peft' if args.with_peft else 'full',
        args.version
    )
    set_seed(42)
    
    model, tokenizer, peft_config = load_model(args)
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    raw_dataset = load_dataset_qa(test_files)
    
    training_arguments = load_training_args(args)
    
    max_seq_length = 512 # 2048
    trainer = SFTTrainer(
        model=model,
        train_dataset=raw_dataset['train'],
        eval_dataset=raw_dataset['dev'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    
    print("Fine-tuning ....")
    save_model_dir = os.path.join(args.output_model_dir, args.repo_name.split('/')[-1])
    # trainer.train(resume_from_checkpoint=True)
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
    parser.add_argument("--llm_model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--generation_method", type=str)
    parser.add_argument("--output_model_dir", type=str)
    parser.add_argument("--output_result_dir", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--with_peft", type=str2bool, default=False)
    parser.add_argument("--version", default=1, type=int)
    
    args = parser.parse_args()
    main(args)

