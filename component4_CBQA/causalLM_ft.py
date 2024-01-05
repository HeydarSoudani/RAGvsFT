import argparse, json, os
import torch
from datasets import Dataset, DatasetDict

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, get_scheduler
from accelerate import Accelerator

from torch.utils.data import DataLoader
from torch.optim import AdamW
import collections
import numpy as np
import math
from tqdm.auto import tqdm

def group_texts(examples):
    block_size = 128
    
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Running on the GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
        print("Running on the mps")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        # load_in_8bit=True,
        # device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    ### === Load dataset =====================
    with open(args.corpus_path, 'r') as input_file:
        corpus_data = [json.loads(line)['contents'] for line in input_file]
    dataset = DatasetDict({
        'train': Dataset.from_dict({
            "text": corpus_data,
        }),
        'validation': Dataset.from_dict({
            "text": corpus_data,
        })
    })
    print(dataset)
    
    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    
    ### === Training (v1) ======================
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)
    model_save_path = os.path.join(args.model_output_dir, args.output_filename)
    os.makedirs(model_save_path, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=model_save_path,
        evaluation_strategy = "epoch",
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        # logging_dir='/content/drive/MyDrive/RAGvsFT/CLM_ft/logs',
        # push_to_hub=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--model_output_dir", type=str)
    parser.add_argument("--model_output_filename", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    
    args = parser.parse_args()
    main(args)
