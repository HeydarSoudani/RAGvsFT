#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset
from huggingface_hub import notebook_login
from huggingface_hub import HfFolder
import evaluate
import torch

import os, json, argparse
import numpy as np
import random
import nltk

os.environ["WANDB_MODE"] = "offline"
nltk.download("punkt", quiet=True)

print("Available GPUs:", torch.cuda.device_count())
device = 'cuda:0'
prompt_prefix = "Answer the question : "
dataset_name = 'popQA' # [TQA, popQA, EQ]
dev_split = 0.1
with_peft = True
training_style = 'qa' # ['clm', 'qa']
target_relation_ids = 'all'
# target_relation_ids = ["91"]
# target_relation_ids = ["91", "106", "22", "182"]

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

def load_model(args, with_peft=False):
    if not with_peft:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            # device_map={"": 0}
        )
        # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            return_dict=True,
            trust_remote_code=True
            # device_map={"":0},
        )
        model = prepare_model_for_kbit_training(model)
        
        def print_trainable_parameters(model):
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )
        
        peft_config = LoraConfig(
            lora_alpha=16,
            r=32,
            lora_dropout=0.05,
            target_modules=["q", "v"],
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            # inference_mode=False
        )
        model = get_peft_model(model, peft_config)

        print_trainable_parameters(model)
        model.print_trainable_parameters()
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    return model, tokenizer, data_collator
         
def load_relations_data(args):
    
    if dataset_name == "TQA":
        subfolders = ['train', 'dev']
    else:
        subfolders = ['train', 'dev', 'test']
    
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = os.path.join(args.data_dir, subfolder)
        for file in os.listdir(subfolder_path):
            relation_id = file.split('.')[0]
            if relation_id not in relation_files:
                relation_files[relation_id] = []
            relation_files[relation_id].append(os.path.join(subfolder_path, file))    

    # Select one relation =================
    if target_relation_ids == "all":
        test_relation_ids = ["22", "218", "91", "257", "182", "164", "526", "97", "533", "639", "472", "106", "560", "484", "292", "422"]
    else:
        test_relation_ids = target_relation_ids
    
    test_files = {subfolder: [] for subfolder in subfolders}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(args.data_dir, subfolder)
        for file in os.listdir(subfolder_path):
            file_id = file.split('.')[0]
            if file_id in test_relation_ids:
                test_files[subfolder].append(os.path.join(subfolder_path, file))

    print("Selected Relation ID:", test_relation_ids)
    print("Selected Files:", test_files)

    return test_relation_ids, test_files, relation_files     
        
def load_dataset_qa(tokenizer, test_files):
    
    train_data = []
    # for file in test_files['train']:
    for file in test_files['test']:
        train_data.extend(load_json_file(file))    
    dev_data = []
    for file in test_files['test']:
    # for file in test_files['dev']:
        dev_data.extend(load_json_file(file)) 
    
    train_subset_size = int(subset_percentage * len(train_data))
    subset_train_data = random.sample(train_data, train_subset_size)
    dev_subset_size = int(subset_percentage * 0.1 * len(dev_data))
    subset_dev_data = random.sample(dev_data, dev_subset_size)

    if dataset_name in ['EQ', 'popQA']:
        train_questions = [item['question'] for item in subset_train_data]
        train_answers = [item['answers'] for item in subset_train_data]
        val_questions = [item['question'] for item in subset_dev_data]
        val_answers = [item['answers'] for item in subset_dev_data]
    else:
        train_questions = [item['Question'] for item in subset_train_data]
        train_answers = [item['Answer']['NormalizedAliases'] for item in subset_train_data]
        val_questions = [item['Question'] for item in subset_dev_data]
        val_answers = [item['Answer']['NormalizedAliases'] for item in subset_dev_data]

    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "question": train_questions,
            "possible_answers": train_answers
        }),
        'dev': Dataset.from_dict({
            "question": val_questions,
            "possible_answers": val_answers
        })
    })
    print(raw_dataset)
    
    # === Get max length =====
    tokenized_inputs = concatenate_datasets([raw_dataset['train'], raw_dataset['dev']]).map(
    lambda x: tokenizer(x["question"], truncation=True), batched=True, remove_columns=["question", "possible_answers"]
    )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([raw_dataset["train"], raw_dataset["dev"]]).map(
        lambda x: tokenizer(x["possible_answers"][0], truncation=True), batched=True, remove_columns=["question", "possible_answers"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")
    
    def tokenize_function(examples):        
        inputs = [prompt_prefix + item for item in examples['question']]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True
        )
        labels = tokenizer(
            text_target=[pa[0] for pa in examples["possible_answers"]],
            max_length=max_target_length,
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    tokenized_train_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["question", "possible_answers"],
        desc="Running tokenizer on dataset",
    )
    print(tokenized_train_datasets)
    
    # === Print a sample of dataset
    input_text = tokenizer.decode(
        tokenized_train_datasets['train'][0]["input_ids"],
        skip_special_tokens=True
    )
    label_text = tokenizer.decode(
        tokenized_train_datasets['train'][0]["labels"],
        skip_special_tokens=True
    )
    print(input_text)
    print(label_text)
    
    return tokenized_train_datasets

def load_training_args(args):
    version = 1
    repo_name = f"{args.model_name_or_path.split('/')[1]}-peft-{version}"
    output_dir = os.path.join(args.output_model_dir, repo_name.split('/')[-1])
    
    training_arguments = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        num_train_epochs=args.epochs,
        learning_rate=2e-3, # 5e-5
        fp16=False, # Overflows with fp16
        # logging & evaluation strategies
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
        # push to hub parameters
        report_to="wandb",
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id=repo_name,
        hub_token=HfFolder.get_token(),
    )
    
    return training_arguments

def main(args):
    
    args.repo_name = "HeydarS/{}_{}_v{}".format(
        args.model_name_or_path.split('/')[-1],
        'peft' if with_peft else 'no_peft',
        args.version
    )
    
    set_seed(42)
    model, tokenizer, data_collator = load_model(args, with_peft=with_peft)
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    tokenized_train_datasets = load_dataset_qa(tokenizer, test_files)
    training_arguments = load_training_args(args)
    
    metric = evaluate.load("rouge")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return result
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_train_datasets["dev"],
        compute_metrics=compute_metrics
    )
    
    print("Fine-tuning ....")
    save_model_dir = os.path.join(args.output_model_dir, args.repo_name.split('/')[-1])
    trainer.train()
    model.save_pretrained(save_model_dir)
    model.push_to_hub(args.repo_name, token=True)
    print("Fine-tuning is done.")

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
