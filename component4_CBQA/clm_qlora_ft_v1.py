#!/usr/bin/env python3

import argparse, os, json
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
from huggingface_hub import HfFolder, HfApi
from trl import SFTTrainer

os.environ["WANDB_MODE"] = "offline"
token = "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"
HfFolder.save_token(token)

def main(args):
    # === Define output folder ===
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)
    model_save_path = os.path.join(args.model_output_dir, args.model_output_filename)
    os.makedirs(model_save_path, exist_ok=True)
    
    # === Model loading ==========
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    # )
    
    # ref: https://medium.com/@bnjmn_marie/lightweight-inference-with-large-language-models-using-qlora-335a3f029229
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # === Apply PEFT ==============
    # lora_alpha = 16
    # lora_dropout = 0.1
    # lora_r = 64
    
    # V2, https://towardsdatascience.com/qlora-fine-tune-a-large-language-model-on-your-gpu-27bed5a03e2b
    lora_alpha = 8
    lora_dropout = 0.05
    lora_r = 32
    
    # V3: On OPT-6.7b, https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing 
    # lora_alpha = 32
    # lora_dropout = 0.05
    # lora_r = 16
    
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["query_key_value"]
    ) 
    
    # === Dataset =================
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
    
    # === Training process =========
    max_seq_length = 512
    training_arguments = transformers.TrainingArguments(
        # per_device_train_batch_size=16,
        # gradient_accumulation_steps=4,
        # num_train_epochs=args.epochs,
        # warmup_steps=500,
        # max_steps=3000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=2,
        max_steps=20,
        # logging_steps=400,
        # warmup_ratio = 0.03,
        # max_grad_norm=0.3,
        learning_rate=2e-4,
        fp16=True,
        optim="paged_adamw_8bit",
        output_dir=args.repo_name,
        report_to="none"
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=data['train'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    
    device = next(model.parameters()).device
    print(device)

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module.to(dtype=torch.float32, device=device)  # specify both dtype and device
             
    # === Save and upload ==========
    trainer.train()
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
    
    
