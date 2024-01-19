#!/usr/bin/env python3

import argparse, os, json, logging
import random
import torch
import torch.nn as nn
import bitsandbytes as bnb
from itertools import chain
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset, DatasetDict
import evaluate
from sklearn.model_selection import train_test_split
from huggingface_hub import HfFolder, HfApi
import pandas as pd


os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def accuracy_by_exact_match(pred, possible_answers):
    is_correct = False
    genread_has_answer = False
    for pa in possible_answers:
        if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
            is_correct = True
    return is_correct


def train(args):
    device = 'cuda:0'
    repo_name = "HeydarS/{}-qlora".format(args.model_name_or_path.split('/')[-1])

    ### === Defining model ====================
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={"": 0},
        quantization_config=bnb_config,
        # trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    
    ### === Introducing PEFT to the model ==== 
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
    model.print_trainable_parameters()
    
    ### === Defining tokenizer ===============
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    ### === Defining Dataset =================
    max_pos_embeddings = 128
    max_train_samples = None
    max_eval_samples = None
    max_test_samples = None
    block_size = tokenizer.model_max_length
    
    with open(args.train_data_path, 'r') as input_file:
        corpus_data = [
        (json.loads(line)['question'], json.loads(line)['possible_answers'][0]) for line in input_file
        ]
    
    with open(args.test_data_path, 'r') as input_file:
        test_data = [
            (json.loads(line)['question'], json.loads(line)['possible_answers'][0]) for line in input_file
        ]
    
    train_corpus, val_corpus = train_test_split(corpus_data, test_size=0.1)
    
    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "qa": train_corpus,
        }),
        'validation': Dataset.from_dict({
            "qa": val_corpus,
        }),
        'test': Dataset.from_dict({
            "qa": test_data,
        })
    })
    
    column_names = list(raw_dataset["train"].features)
    text_column_name = "qa" if "qa" in column_names else column_names[0]
    
    def qa_tokenize_function(examples):
        num_shots = 2
        input_max_length = 128
        output_max_length = 4

        inputs = []
        outputs = []
        for i, current_query in enumerate(examples[text_column_name]):
            prompt = ""
            if num_shots != 0:
                selection_pool = [q for j, q in enumerate(examples[text_column_name]) if j != i]
                selected_pairs = random.sample(selection_pool, num_shots)
                prompt = "\n".join([f"Q: {pair[0]} A: {pair[1]}" for pair in selected_pairs])
            prompt += f"\nQ: {current_query[0]} A:"

            inputs.append(prompt)
            outputs.append(current_query[1])

        model_inputs = tokenizer(
            inputs,
            padding = True,
            truncation = True,
            max_length = input_max_length
        )

        labels = tokenizer(
            outputs,
            padding = True,
            truncation=True,
            max_length=output_max_length,
        )
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_datasets = raw_dataset.map(
        qa_tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = tokenized_datasets["train"]
    if max_train_samples is not None:
        max_train_samples = min(len(train_dataset), max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = tokenized_datasets["validation"]
    if max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    test_dataset = tokenized_datasets["test"]
    if max_test_samples is not None:
        max_test_samples = min(len(test_dataset), max_test_samples)
        test_dataset = test_dataset.select(range(max_test_samples))    

    print(tokenized_datasets)
    
    # = Print a sample of dataset
    input_text = tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=True)
    label_text = tokenizer.decode(train_dataset[0]["labels"], skip_special_tokens=True)

    print(input_text)
    print('\n')
    print(label_text)
    # = ==
    
    # === Training process ==============
    output_dir = os.path.join(args.output_dir, args.repo_name.split('/')[-1])
    
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-3,
        max_grad_norm=0.3,
        warmup_ratio=0,
        lr_scheduler_type="linear",
        report_to=[],
        
        save_strategy="epoch",
        save_total_limit=2,
        # save_steps=1,
        # logging_steps=1,
        # fp16=True,
        # max_steps=200,
        # load_best_model_at_end=False,
    )
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    
    trainer.train()
    # model.save_pretrained(model_save_path)
    model.push_to_hub(args.repo_name, token=True)
    

def test(args):
    device = 'cuda:0'
    
    ### === Loading model & tokenizer (After FT) =====
    args.repo_name = "component4_CBQA/models/opt-1.3b_qlora_v2/checkpoint-500"
    # args.repo_name = "component4_CBQA/models/opt-1.3b_qlora_v2/checkpoint-1500"
    
    lora_config = LoraConfig.from_pretrained(args.repo_name)
    model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path,
        device_map={"": 0},
    )
    
    model = get_peft_model(model, lora_config)
    model = PeftModel.from_pretrained(model, args.repo_name)
    model.eval()
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
    ### === Loading model & tokenizer (ZS setting) ===
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path,
    #     device_map={"": 0},
    # )
    # model.eval()
    # model.to(device)
    
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model_name_or_path,
    #     trust_remote_code=True
    # )
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ### === Loading data and Defining params ===
    num_shots = 15
    model_max_length=2048
    max_new_tokens = 15
    max_inpt_tokens = tokenizer.model_max_length if model_max_length is None else model_max_length

    with open(args.test_data_path, 'r') as file:
        data = [json.loads(line) for line in file]
    
    ### === Test loop ==========================
    knowledge = pd.read_csv(args.knowledge_input_file, sep="\t")
    
    
    accuracy = []
    for i, current_query in enumerate(data):

        selection_pool = [row for j, row in enumerate(knowledge) if row["prop_id"] != 106]
        
        
        # selection_pool = [q for j, q in enumerate(data) if j != i]
        selected_pairs = random.sample(selection_pool, num_shots)

        prompt = "\n".join([f"Q: {pair['question']} A: {pair['possible_answers'][0]}" for pair in selected_pairs])
        prompt += f"\nQ: {current_query['question']} A:"

        # Feed to the model
        inpts = tokenizer(prompt, return_tensors="pt").to(device)
        gen = model.generate(
            input_ids=inpts.input_ids[:, -(max_inpt_tokens - max_new_tokens):],
            attention_mask=inpts.attention_mask[:, -(max_inpt_tokens - max_new_tokens):],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            # do_sample=True, # more creative, use very low frequent vocab
            # top_p=0.5, # High value makes model more creative, less makes it more safe
        )

        text = tokenizer.decode(gen[0])
        actual_prompt = tokenizer.decode(inpts.input_ids[0, -(max_inpt_tokens - max_new_tokens):])

        pred = text[len(actual_prompt):]
        # pred = pred[1:]
        # pred = pred.split("\n")[0]
        # pred = pred.split(",")[0]
        if pred.startswith("\n\n"):
            pred = pred[2:]
        pred = pred.split("\n")[0]

        print("Pred:{}, Labels: {}".format(pred, current_query['possible_answers']))
        is_correct = accuracy_by_exact_match(pred, current_query['possible_answers'])
        accuracy.append(is_correct)

    print(sum(accuracy) / len(accuracy))

def main(args):
    
    args.repo_name = "HeydarS/{}_qlora_v{}".format(args.model_name_or_path.split('/')[-1], args.version)
    print(args.repo_name)
    
    # train(args)
    test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--version", default=1, type=int)
    
    args = parser.parse_args()
    main(args)
