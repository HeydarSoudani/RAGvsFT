#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets
from huggingface_hub import HfFolder
import torch

import os, json, argparse
import numpy as np
import logging
import random

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)
os.environ["WANDB_MODE"] = "offline"

print("Available GPUs:", torch.cuda.device_count())
target_relation_ids = 'all'
subset_percentage = 1.0

# V1: small
prompt_prefix = "Answer the question:"
# V2: others
# prompt_prefix = "Question:"

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

def load_relations_data(args):
    subfolders = ['train', 'dev', 'test']  
      
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = f"{args.data_dir}/{args.generation_method}/{subfolder}"
        
        if os.path.exists(subfolder_path):    
            for file in os.listdir(subfolder_path):
                relation_id = file.split('.')[0]
                if relation_id not in relation_files:
                    relation_files[relation_id] = []
                relation_files[relation_id].append(os.path.join(subfolder_path, file))    

    # Select one relation =================
    if target_relation_ids == "all":
        if args.dataset_name == 'popQA':
            test_relation_ids = ['22', '91', '97', '106', '164', '182', '218', '257', '292', '422', '472', '484', '526', '533', '560', '639']
        elif args.dataset_name == 'witQA':
            test_relation_ids = ['17', '19', '22', '25', '27', '36', '50', '57', '58', '69', '86', '106', '123', '136', '140', '149', '162', '184', '344', '452', '462', '641', '674', '1038', '1050', '1376', '1431', '1433', '2012', '2936', '3301', '4647']
        elif args.dataset_name == 'EQ':
            test_relation_ids = ['17', '19', '20', '26', '30', '36', '40', '50', '69', '106', '112', '127', '131', '136', '159', '170', '175', '176', '264', '276', '407', '413', '495', '740', '800']
    else:
        test_relation_ids = target_relation_ids
    
    test_files = {subfolder: [] for subfolder in subfolders}
    for subfolder in subfolders:
        subfolder_path = f"{args.data_dir}/{args.generation_method}/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_id = file.split('.')[0]
                if file_id in test_relation_ids:
                    test_files[subfolder].append(os.path.join(subfolder_path, file))

    print("Selected Relation ID:", test_relation_ids)
    logging.info(f"Selected Relation ID: {test_relation_ids}")

    return test_relation_ids, test_files, relation_files     

def load_dataset_qa(tokenizer, test_files):
    
    # === Load all data ========================
    train_data = []
    for file in test_files['train']:
        train_data.extend(load_json_file(file))    
    
    dev_data = []
    for file in test_files['dev']:
        dev_data.extend(load_json_file(file)) 
    
    # === Select subset of data ================
    train_subset_size = int(subset_percentage * len(train_data))
    subset_train_data = random.sample(train_data, train_subset_size)
    dev_subset_size = int(subset_percentage * len(dev_data))
    subset_dev_data = random.sample(dev_data, dev_subset_size)

    train_questions = [item['question'] for item in subset_train_data]
    train_answers = [item['answers'] for item in subset_train_data]
    val_questions = [item['question'] for item in subset_dev_data]
    val_answers = [item['answers'] for item in subset_dev_data]

    # === Convert to dataset ===================
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
    
    # === Tokenizing dataset ====================
    
    # = Get max length =====
    tokenized_inputs = concatenate_datasets([raw_dataset['train'], raw_dataset['dev']]).map(
    lambda x: tokenizer(x["question"], truncation=True), batched=True, remove_columns=["question", "possible_answers"]
    )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([raw_dataset["train"], raw_dataset["dev"]]).map(
        lambda x: tokenizer(x["possible_answers"][0], truncation=True), batched=True, remove_columns=["question", "possible_answers"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")
    
    # = Tokenize function =====
    def tokenize_function(examples):
        inputs = [f"{prompt_prefix} {item}" for item in examples['question']]
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
    
    # = Print a sample of dataset
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

def load_model(args):
    if args.with_peft:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_use_double_quant=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            return_dict=True,
            trust_remote_code=True,
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
        
        lora_alpha = 16
        lora_dropout = 0.05
        lora_r = 32
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            r=lora_r,
            lora_dropout=lora_dropout,
            target_modules=["q", "v"],
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, peft_config)

        print_trainable_parameters(model)
        model.print_trainable_parameters()
    
    else:  
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            # device_map={"": 0}
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params") 
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    return model, tokenizer, data_collator

def load_training_args(args):
    training_arguments = Seq2SeqTrainingArguments(
        output_dir=args.save_model_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr, # 5e-5
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb",
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id=args.repo_name.split('/')[-1],
        hub_token=HfFolder.get_token()
        # fp16=False, # Overflows with fp16
        # logging & evaluation strategies
        # load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
        # push to hub parameters
    )
    
    return training_arguments

def main(args):
    # == Create data & output dir ===========================
    args.data_dir = f"component0_preprocessing/generated_data/{args.dataset_name}_costomized"
    
    if args.output_path:
        args.output_model_dir = args.output_path
    else:
        args.output_model_dir = f"component2_AnsGen/models/{args.dataset_name}"
    os.makedirs(args.output_model_dir, exist_ok=True)
    
    # == Define model output dir ============================
    training_approach = "peft" if args.with_peft else "full"
    args.repo_name = f"HeydarS/{args.llm_model_name}_{args.dataset_name}_{args.generation_method}_{training_approach}_v{args.version}"
    args.save_model_dir = os.path.join(args.output_model_dir, args.repo_name.split('/')[-1])
    
    logging.info(f"""
        Base Model: {args.model_name_or_path}
        Epoch: {args.epochs}
        Batch size: {args.batch_size}
        Lr: {args.lr}
        With PEFT: {args.with_peft}
        Output Model Dir: {args.save_model_dir}
        Version: {args.version}
    """)
    set_seed(42)
    
    # == Load data & model ==================================
    model, tokenizer, data_collator = load_model(args)
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    tokenized_train_datasets = load_dataset_qa(tokenizer, test_files)
    training_arguments = load_training_args(args)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_train_datasets["dev"],
        data_collator=data_collator,
    )
    
    print("Fine-tuning ....")
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    model.save_pretrained(args.save_model_dir)
    model.push_to_hub(args.repo_name, token=True)
    print("Fine-tuning is done.")


if __name__ == "__main__":
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--llm_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--generation_method", type=str, choices=['prompting', 'pipeline'], default='prompting')
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--with_peft", type=str2bool, default=False)
    parser.add_argument("--version", default=1, type=int)
    parser.add_argument("--output_path", type=str, default=None)
    
    args = parser.parse_args()
    main(args)
