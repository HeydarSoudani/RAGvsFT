#!/usr/bin/env python3

import argparse, os, json
import numpy as np
import random
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback

os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda:0'
dataset_name = 'popQA' # [TQA, popQA, EQ]
completion_template_wo_ans = "Q: {} A:"
completion_template_with_ans = "Q: {} A: {}"
dev_split = 0.1
with_peft = True
with_fs = True
# with_rag = False
training_style = 'qa' # ['clm', 'qa']
# target_relation_ids = ["106", "22", "182"]
# target_relation_ids = ["22", "218", "91", "257", "182", "164", "526", "97", "533", "639", "472", "106", "560", "484", "292", "422"]
target_relation_ids = ["91"]

if dataset_name == "TQA":
    num_relations = 1
else:
    num_relations = 15

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

def load_model(args):
    if not with_peft:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    else:
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
    
    return model, tokenizer

def load_training_args(args):
    repo_name = ""
    output_dir = ""
    logging_dir = ""
    per_device_train_batch_size = 8
    
    training_arguments = TrainingArguments(
        output_dir=output_dir,          # output directory
        num_train_epochs=10,              # set number of epochs
        per_device_train_batch_size=per_device_train_batch_size,   # set batch size for training
        per_device_eval_batch_size=4,    # set batch size for evaluation
        warmup_steps=0,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=logging_dir,            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )
    
    return training_arguments
    


def main(args):
    repo_name = ""
    output_dir = ""
    set_seed(42)
    
    model, tokenizer = load_model(args, with_peft=with_peft)
    
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    tokenized_train_datasets, (test_questions, test_answers) = load_dataset_qa(
        tokenizer,
        test_files
    )
    
    training_arguments = load_training_args(args)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # callbacks=[eval_callback]
    )
    
    trainer.train()
    model.save_pretrained(save_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_model_dir", type=str)
    parser.add_argument("--output_result_dir", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--version", default=1, type=int)
    
    args = parser.parse_args()
    main(args)
