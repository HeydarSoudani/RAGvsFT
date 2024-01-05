#!/usr/bin/env python3

import argparse, json, os, math
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import default_data_collator
from transformers import TrainingArguments, Trainer, get_scheduler


from torch.utils.data import DataLoader
from torch.optim import AdamW
import collections
import numpy as np
from tqdm.auto import tqdm

os.environ["WANDB_MODE"] = "offline"

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
    
    ### === Model introduction ============== 
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)
    model_save_path = os.path.join(args.model_output_dir, args.model_output_filename)
    os.makedirs(model_save_path, exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=True,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    ### === Load dataset =====================
    with open(args.corpus_path, 'r') as input_file:
        corpus_data = [json.loads(line)['contents'] for line in input_file]
    train_texts, val_texts = train_test_split(corpus_data, test_size=0.1)
    dataset = DatasetDict({
        'train': Dataset.from_dict({
            "text": train_texts,
        }),
        'validation': Dataset.from_dict({
            "text": val_texts,
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
    
    ### ========================================
    ### === Training (v1) ======================
    
    # training_args = TrainingArguments(
    #     output_dir=model_save_path,
    #     evaluation_strategy = "epoch",
    #     num_train_epochs=args.epochs,
    #     learning_rate=2e-5,
    #     weight_decay=0.01,
    #     # logging_dir='/content/drive/MyDrive/RAGvsFT/CLM_ft/logs',
    #     # push_to_hub=True,
    # )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=lm_datasets["train"],
    #     eval_dataset=lm_datasets["validation"],
    # )
    # trainer.train()
    
    # eval_results = trainer.evaluate()
    # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    ### ========================================
    ### === Training (v2) ======================
    
    # === Dataset
    batch_size = 64
    train_dataloader = DataLoader(
        lm_datasets["train"],
        shuffle=True,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        lm_datasets["validation"],
        batch_size=batch_size
    )
    
    # === training parameters
    optimizer = AdamW(model.parameters(), lr=2e-5)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_training_steps,
    )
    
    # === training loop
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(args.epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(lm_datasets["validation"])]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
        
        # Save model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(model_save_path, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--model_output_dir", type=str)
    parser.add_argument("--model_output_filename", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    
    args = parser.parse_args()
    main(args)
