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
training_style = 'qa' # ['clm', 'qa']
target_relation_ids = 'all'
subset_percentage = 1.0
num_relations = 1 if dataset_name == "TQA" else 15

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
            # device_map={"": 0}
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
            trust_remote_code=True
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        
        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, peft_config

