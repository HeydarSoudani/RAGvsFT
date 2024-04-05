#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from datasets import Dataset, DatasetDict
from huggingface_hub import HfFolder
import torch

import os, json, argparse
import numpy as np
import logging
import random
# import nltk

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[
        # logging.FileHandler("app.log"),
        logging.StreamHandler()
    ])
os.environ["WANDB_MODE"] = "offline"
# nltk.download("punkt", quiet=True)

print("Available GPUs:", torch.cuda.device_count())
device = 'cuda:0'
target_relation_ids = 'all'
subset_percentage = 1.0

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

def load_relations_data(args): 
    subfolders = ['train', 'dev', 'test']    
    
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = f"{args.data_dir}/{args.generation_method}/{subfolder}"
        
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
        subfolder_path = f"{args.data_dir}/{args.generation_method}/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_id = file.split('.')[0]
                if file_id in test_relation_ids:
                    test_files[subfolder].append(os.path.join(subfolder_path, file))

    print("Selected Relation ID:", test_relation_ids)
    logging.info(f"Selected Relation ID: {test_relation_ids}")

    return test_relation_ids, test_files, relation_files     

def load_dataset_qa(test_files):
    
    # === Load all data ========================
    train_data = []
    for file in test_files['train']:
        train_data.extend(load_json_file(file))    
    
    dev_data = []
    for file in test_files['dev']:
        dev_data.extend(load_json_file(file)) 
    
    # === Select subset of data ================
    train_subset_size = int(subset_percentage * len(train_data))
    subset_train_data = random.sample(train_data, train_subset_size)
    dev_subset_size = int(subset_percentage * len(dev_data))
    subset_dev_data = random.sample(dev_data, dev_subset_size)

    train_questions = [item['question'] for item in subset_train_data]
    train_answers = [item['answers'] for item in subset_train_data]
    val_questions = [item['question'] for item in subset_dev_data]
    val_answers = [item['answers'] for item in subset_dev_data]

    # === Apply prompt template ================
    train_data = [
        args.prompt_template.format(question=question, answer=train_answers[i])
        for i, question in enumerate(train_questions)
    ]
    
    val_data = [
        args.prompt_template.format(question=question, answer=val_answers[i])
        for i, question in enumerate(val_questions)
    ]

    # === Convert to dataset ===================
    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "text": train_data
        }),
        'dev': Dataset.from_dict({
            "text": val_data
        })
    })
    print(raw_dataset)
    logging.info("Train & Dev datasets are loaded.")
    
    return raw_dataset

def load_model(args):
    if args.with_peft:
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
        
        if args.llm_model_name == 'llama2':
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
        elif args.llm_model_name in ["zephyr", "tiny_llama", "MiniCPM"]:
            pass
               
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map={"": 0}
        )
        model.config.use_cache = False
         
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    
    # == Exeptions for some models
    tokenizer.pad_token = tokenizer.eos_token
    if args.llm_model_name == 'mistral':
        tokenizer.padding_side = "right"
    
    return model, tokenizer, peft_config

def load_training_args(args):        
    training_arguments = TrainingArguments(
        output_dir=args.save_model_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.fp16,
        
        report_to="wandb",
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id=args.repo_name.split('/')[-1],
        hub_token=HfFolder.get_token()
    )
    
    return training_arguments

def main(args):
    # == Create data & output dir ===========================
    args.data_dir = f"component0_preprocessing/generated_data/{args.dataset_name}_costomized"
    args.output_model_dir = f"component2_AnsGen/models/{args.dataset_name}"
    os.makedirs(args.output_model_dir, exist_ok=True)
    
    # == Define model output dir ============================
    training_approach = "peft" if args.with_peft else "full"
    args.repo_name = f"HeydarS/{args.llm_model_name}_{args.dataset_name}_{training_approach}_v{args.version}"
    args.save_model_dir = os.path.join(args.output_model_dir, args.repo_name.split('/')[-1])
    
    logging.info(f"""
        Base Model: {args.model_name_or_path}
        Output Model Dir: {args.save_model_dir}
    """)
    set_seed(42)
    
    # == Parameters per model ==============================
    if args.llm_model_name == 'llama2':
        args.batch_size = 32
        args.fp16 = True
        args.bf16 = False
    elif args.llm_model_name in ['mistral', 'zephyr']:
        args.batch_size = 4
        args.fp16=False
        args.bf16=False
    elif args.llm_model_name in ['tiny_llama', 'MiniCPM']:
        pass
    
    if args.llm_model_name in ["llama2", "tiny_llama", "mistral"]:
        args.prompt_template = """
            <s>You are an Answer Generator system. Your goal is to provide concise responses to questions, drawing upon either the context provided or your own stored knowledge.\n
            [INST]\n
            Question: {question}\n
            [/INST]\n
            Answer: {answer}
            </s>"""
    elif args.llm_model_name == 'zephyr':
        args.prompt_template = """
            <|system|>You are an Answer Generator system. Your goal is to provide concise responses to questions, drawing upon either the context provided or your own stored knowledge.\n
            <|user|>\n 
            Question: {question}\n
            <|assistant|>\n
            Answer: {answer}
            """
    elif args.llm_model_name == "MiniCPM":
        args.prompt_template = """
        <User> You are an Answer Generator system. Your goal is to provide one-entity responses to questions, drawing upon either the context provided or your own stored knowledge.\n
        Question: {question}\n
        <AI>\n
        Answer: {answer}
        """
    
    # == Load data & model ==================================
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
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    model.save_pretrained(args.save_model_dir)
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
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--generation_method", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--with_peft", type=str2bool, default=False)
    parser.add_argument("--version", default=1, type=int)
    
    args = parser.parse_args()
    main(args)


