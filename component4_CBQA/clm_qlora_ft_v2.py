#!/usr/bin/env python3

import argparse, os, json, logging
import torch
import torch.nn as nn
import bitsandbytes as bnb
from itertools import chain
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
from huggingface_hub import HfFolder, HfApi
from trl import SFTTrainer

os.environ["WANDB_MODE"] = "offline"
token = "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"
HfFolder.save_token(token)
logger = logging.getLogger(__name__)

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

# ref: https://medium.com/@bnjmn_marie/lightweight-inference-with-large-language-models-using-qlora-335a3f029229
def main(args):
    max_pos_embeddings = 1024
    max_train_samples = 100
    
    # === Define output folder ===
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)
    model_save_path = os.path.join(args.model_output_dir, args.model_output_filename)
    os.makedirs(model_save_path, exist_ok=True)
    
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"":0},
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.05,
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        inference_mode=False
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # === Dataset =================
    corpus_path = "/content/drive/MyDrive/RAGvsFT/data_bm25/corpus/corpus_splitted.jsonl"
    with open(args.corpus_path, 'r') as input_file:
        corpus_data = [json.loads(line)['contents'] for line in input_file]

    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "text": corpus_data,
        })
    })
    
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
        logger.warning(
            f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
        )
        if max_pos_embeddings > 0:
            block_size = min(1024, max_pos_embeddings)
        else:
            block_size = 1024
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    train_dataset = lm_datasets["train"]
    if max_train_samples is not None:
        max_train_samples = min(len(train_dataset), max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    
    print(train_dataset)
    print(train_dataset[0])
    # ==============
    
    
    # max_seq_length = 512
    # training_arguments = transformers.TrainingArguments(
    #     per_device_train_batch_size=16,
    #     gradient_accumulation_steps=8,
    #     warmup_steps=500,
    #     max_steps=3000,
    #     learning_rate=2e-4,
    #     fp16=True,
    #     logging_steps=400,
    #     output_dir=args.repo_name,
    #     optim="paged_adamw_8bit",
    #     report_to="none"
    # )
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     peft_config=peft_config,
    #     dataset_text_field="text",
    #     max_seq_length=max_seq_length,
    #     tokenizer=tokenizer,
    #     args=training_arguments,
    # )
    
    # device = next(model.parameters()).device
    # print(device)

    # trainer.train()
    
    # # === Save and upload ==========
    # model.save_pretrained(model_save_path)
    # # tokenizer.save_pretrained(model_save_path)
    # model.push_to_hub(args.repo_name, token=True)
  
  
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
