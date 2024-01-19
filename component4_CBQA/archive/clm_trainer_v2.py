#!/usr/bin/env python3

import argparse, json, os, logging
import math
import os
from itertools import chain
import evaluate
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)

os.environ["WANDB_MODE"] = "offline"
# check_min_version("4.37.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


def main(args):
    # === Parameters
    seed = 0
    max_train_samples = None
    max_eval_samples = None
    max_pos_embeddings = 128
    
    # === 
    set_seed(seed)
    
    with open(args.corpus_path, 'r') as input_file:
        corpus_data = [json.loads(line)['contents'] for line in input_file]
    train_texts, val_texts = train_test_split(corpus_data, test_size=0.1)
    raw_datasets = DatasetDict({
        'train': Dataset.from_dict({
            "text": train_texts,
        }),
        'validation': Dataset.from_dict({
            "text": val_texts,
        })
    })
    print(raw_datasets)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output

    tokenized_datasets = raw_datasets.map(
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
    
    eval_dataset = lm_datasets["validation"]
    if max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)
    
    
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)
    model_save_path = os.path.join(args.model_output_dir, args.model_output_filename)
    os.makedirs(model_save_path, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_steps=2000,
        # eval_steps=200,
        # logging_steps=200,
        # gradient_accumulation_steps=8,
        # weight_decay=0.1,
        # # warmup_steps=100,
        # lr_scheduler_type="cosine",
        # learning_rate=5e-4,
        # fp16=True,
        # push_to_hub=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    
    # Training
    train_result = trainer.train(resume_from_checkpoint = True)
    trainer.save_model()
    metrics = train_result.metrics
    max_train_samples = (
        max_train_samples if max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Evaluation 
    metrics = trainer.evaluate()
    max_eval_samples = max_eval_samples if max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    # Save
    kwargs = {"finetuned_from": args.model_name_or_path, "tasks": "text-generation"}
    if args.dataset_name is not None:
        kwargs["dataset_tags"] = args.dataset_name
    trainer.create_model_card(**kwargs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", required=True)
    parser.add_argument("-d", "--dataset_name", required=True)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--model_output_dir", type=str)
    parser.add_argument("--model_output_filename", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    
    args = parser.parse_args()
    main(args)

