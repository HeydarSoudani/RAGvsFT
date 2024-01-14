#!/usr/bin/env python3

import argparse, os, json

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import Dataset, DatasetDict
from huggingface_hub import HfFolder, HfApi
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

os.environ["WANDB_MODE"] = "offline"
token = "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"
HfFolder.save_token(token)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main(args):
    # model_name = "facebook/opt-125m"
    # repo_name = "HeydarS/opt-125m-lora-v2"
    api = HfApi()
    
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)
    model_save_path = os.path.join(args.model_output_dir, args.model_output_filename)
    os.makedirs(model_save_path, exist_ok=True)
    

    # === Model loading ==========
    model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True)
    model = prepare_model_for_int8_training(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    # === Dataset ==============
    corpus_path = "/content/drive/MyDrive/RAGvsFT/data_bm25/corpus/corpus_splitted.jsonl"
    with open(args.corpus_path, 'r') as input_file:
        corpus_data = [json.loads(line)['contents'] for line in input_file]

    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "text": corpus_data,
        })
    })
    data = raw_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
    print(data)
    
    # === Start trining =============
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            warmup_steps=500,
            max_steps=8000,
            learning_rate=1e-5,
            fp16=True,
            logging_steps=400,
            output_dir=args.repo_name,
            report_to="none"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    
    # api.upload_model(repo_id=args.repo_name, token=token, model=model)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    model.push_to_hub(args.repo_name, token=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("--repo_name", type=str)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--model_output_dir", type=str)
    parser.add_argument("--model_output_filename", type=str)
    parser.add_argument("--epochs", default=1.0, type=float)
    
    args = parser.parse_args()
    main(args)