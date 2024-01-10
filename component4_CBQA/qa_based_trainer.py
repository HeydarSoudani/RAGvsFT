#!/usr/bin/env python3

import argparse, json, os
import torch
import evaluate
from datasets import Dataset
from transformers import OPTForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare_data(tokenizer, corpus_path, max_length=32):
    
    texts = []
    with open(corpus_path, 'r') as file:
        for line in file:
            json_line = json.loads(line.strip())
            texts.append("Q: {} A: {}".format(json_line["question"], json_line["possible_answers"][0]))

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    
    # print(encodings.data["input_ids"][0])
    # print(tokenizer.decode(encodings.data["input_ids"][0], skip_special_tokens=True))
    # print(tokenizer.decode(encodings[0]['ids'], skip_special_tokens=True))
    # print(encodings['ids'].shape)
    # print(encodings[0])
    # dataset = torch.utils.data.Dataset(encodings)
    dataset = Dataset.from_dict(encodings)
    
    return dataset


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = OPTForCausalLM.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    dataset = prepare_data(tokenizer, args.data_path)
    train_test_split = dataset.train_test_split(test_size=0.1) # 10% for validation
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']
    
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)
    model_save_path = os.path.join(args.model_output_dir, args.model_output_filename)
    os.makedirs(model_save_path, exist_ok=True)
    
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # output_ids = [token_id for token_id in labels[0, :] if token_id != -100]
        
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1, preds.shape[-1])
        # preds = preds[:, :-1].reshape(-1)
        
        # print(labels.shape)
        # print(preds.shape)
    
        return metric.compute(predictions=preds, references=labels)
    
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_output_dir", type=str)
    parser.add_argument("--model_output_filename", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    
    
    args = parser.parse_args()
    main(args)
