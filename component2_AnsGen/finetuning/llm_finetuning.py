#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, DatasetDict
from huggingface_hub import HfFolder
import torch

import os, json, argparse
import numpy as np
import logging
import random

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
        args.prompt_template.format(question=question, answer=train_answers[i][0])
        for i, question in enumerate(train_questions)
    ]
    
    val_data = [
        args.prompt_template.format(question=question, answer=val_answers[i][0])
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
    print("\n")
    print(raw_dataset['train'][0]['text'])
    logging.info("Train & Dev datasets are loaded.")
    
    return raw_dataset

def load_model(args):
    if args.with_peft:
        
        if args.llm_model_name in ['llama2', "tiny_llama"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            peft_config = LoraConfig(
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
            model.config.use_cache = False
            model.config.pretraining_tp = 1
            
        elif args.llm_model_name == 'mistral':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            peft_config = LoraConfig(
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"],
            )   
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                # load_in_4bit=True,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model.config.use_cache = False
            model.config.pretraining_tp = 1
            model.gradient_checkpointing_enable()
            
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
               
        elif args.llm_model_name in ["zephyr", "stable_lm2"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant= False,
            )
            
            peft_config = LoraConfig(
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                r=args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=['up_proj', 'base_layer', 'down_proj'],
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                # load_in_4bit=True,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model.config.use_cache = False
            model.config.pretraining_tp = 1
            model.gradient_checkpointing_enable()
            
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
            
        elif args.llm_model_name in ["MiniCPM"]:
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
    # = Exeptions for some models
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if args.llm_model_name in ['mistral', 'zephyr']:
        tokenizer.add_eos_token = True
        tokenizer.add_bos_token, tokenizer.add_eos_token
    
    return model, tokenizer, peft_config

def load_training_args(args):
    training_arguments = TrainingArguments(
        output_dir=args.save_model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        optim=args.optim,
        fp16=args.fp16,
        bf16=args.bf16,
        # max_grad_norm=args.max_grad_norm,
        # warmup_ratio=args.warmup_ratio,
        # group_by_length=args.group_by_length,
        # lr_scheduler_type=args.lr_scheduler_type,
        # weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
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
    if args.llm_model_name in ['llama2', 'mistral', 'zephyr', 'tiny_llama', 'stable_lm2']:
        args.lora_alpha = 16
        args.lora_dropout = 0.1
        args.lora_r = 64
        args.epochs = 2
        args.batch_size = 4
        args.gradient_accumulation_steps = 1
        args.optim = "paged_adamw_32bit"
        args.lr = 2e-4
        args.fp16 = False
        args.bf16 = False
        args.weight_decay=0.001
        args.max_grad_norm=0.3
        args.warmup_ratio=0.03
        args.group_by_length=True
        args.lr_scheduler_type="constant"
    
    elif args.llm_model_name in ['MiniCPM']:
        pass
    
    if args.llm_model_name in ["llama2", "mistral"]:
        args.prompt_template = """<s>[INST] <<SYS>><</SYS>> \n Question: {question} \n[/INST] Answer: {answer} </s>"""
    
    elif args.llm_model_name in ["zephyr", "tiny_llama"]:
        args.prompt_template = """<|system|> </s>\n <|user|> Question: {question}</s>\n <|assistant|> Answer: {answer} </s>"""
    
    elif args.llm_model_name == "stable_lm2":
        args.prompt_template = """<|user|>\n Question: {question}<|endoftext|>\n<|assistant|>\n Answer: {answer}<|endoftext|>"""
    
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
    
    if args.llm_model_name in ["llama2", "mistral", "tiny_llama"]:
        max_seq_length = None
    elif args.llm_model_name in ["zephyr", "stable_lm2"]:
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
        packing=False
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
    parser.add_argument("--with_peft", type=str2bool, default=False)
    parser.add_argument("--version", default=1, type=int)
    
    args = parser.parse_args()
    main(args)


