#!/usr/bin/env python3


import random
import requests
from io import BytesIO
from zipfile import ZipFile
from datasets import Dataset, DatasetDict
import argparse, os, json
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset, DatasetDict
import evaluate

from huggingface_hub import HfFolder, HfApi

import numpy as np
import torch


os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda:0'
dataset_name = 'popQA' # [TQA, popQA, EQ]
completion_template_wo_ans = "Q: {} A:"
completion_template_with_ans = "Q: {} A: {}"

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

def load_model(args, with_peft=False):
    
    if not with_peft:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map={"": 0},
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map={"": 0},
            quantization_config=bnb_config
        )
        model.config.use_cache = False # From ft llama with qlora ...
        model.gradient_checkpointing_enable()
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

        # v1: From ft llama with qlora ...
        # alpha = 16
        # dropout = 0.1
        # r = 64
        
        # v2
        # lora_alpha = 32
        # lora_dropout = 0.05
        # r = 8 

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)

        print_trainable_parameters(model)
        model.print_trainable_parameters()
        
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    return model, tokenizer

def create_few_shot_examples(relation_files, selected_relations, num_samples, split_name):
    few_shot_examples = []
    
    for relation_id in selected_relations:
        files = relation_files[relation_id]
        split_file = next((file for file in files if split_name in file), None)
        data = load_json_file(split_file)
        
        sampled_examples = random.sample(data, min(num_samples, len(data)))
        for example in sampled_examples:
            
            if dataset_name in ['EQ', 'popQA']:
                question = example['question']
                answers = example['answers']
            elif dataset_name == 'TQA':
                question = example['Question']
                answers = example['Answer']['NormalizedAliases']
            
            completion = completion_template_with_ans.format(question, random.choice(answers))
            few_shot_examples.append(completion)
    return few_shot_examples

def load_relations_data(args):
    
    # If you need to download EQ or TQA dataset
    if not os.path.isdir(args.data_dir):
        if dataset_name == "TQA":
            url = "https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz"     # For TQA dataset
        
        url = "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip" # For EQ dataset
        response = requests.get(url)
        zip_file = ZipFile(BytesIO(response.content))
        zip_file.extractall("entity_questions_dataset")
        zip_file.close()
    
    if dataset_name == "TQA":
        num_relations = 1
        subfolders = ['train', 'dev']
    else:
        num_relations = 15
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
    selected_relation_id = random.choice(list(relation_files.keys()))
    # selected_relation_id = "106"
    selected_files = {}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(args.data_dir, subfolder)
        for file in os.listdir(subfolder_path):
            if file.startswith(selected_relation_id):
                selected_files[subfolder] = os.path.join(subfolder_path, file)

    print("Selected Relation ID:", selected_relation_id)
    print("Selected Files:", selected_files)
    
    # Other relations ===============
    if dataset_name in ['EQ', 'popQA']:
        relation_files.pop(selected_relation_id)
    
    selected_relations = random.sample(relation_files.keys(), num_relations)

    return relation_files, selected_relations, selected_files
    
def load_dataset(tokenizer, relation_files, selected_relations, selected_files, with_fs=True):
    num_samples_per_relation = 1
    subset_percentage = 0.8
    input_max_length = 64
    output_max_length = 4
    
    # Load data from selected files
    train_data = load_json_file(selected_files['train'])
    dev_data = load_json_file(selected_files['dev'])

    train_subset_size = int(subset_percentage * len(train_data))
    subset_train_data = random.sample(train_data, train_subset_size)
    dev_subset_size = int(subset_percentage * len(dev_data))
    subset_dev_data = random.sample(dev_data, dev_subset_size)

    # Extract questions and answers
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
    
    # Create the dataset in the desired format
    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "question": train_questions,
            "possible_answers": train_answers
        }),
        'dev': Dataset.from_dict({
            "question": val_questions,
            "possible_answers": val_answers
        }),
    #     'test': Dataset.from_dict({
    #         "question": test_questions,
    #         "possible_answers": test_answers
    #     })
    })
    print(raw_dataset)
    
    # tokenize train & dev datasets ===================
    def qa_tokenize_function(examples, split_name):
        input_prompts = []
        len_dataset = len(examples['question'])
        
        ### === version 1 
        # for idx in range(len_dataset):
        #     if with_fs:
        #         few_shot_examples = create_few_shot_examples(
        #             relation_files,
        #             selected_relations,
        #             num_samples_per_relation,
        #             split_name
        #         )
        #         np.random.shuffle(few_shot_examples)
        #         few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"
        #     else:
        #         few_shot_examples_text = "\n\n"
        #     prompt = few_shot_examples_text + completion_template_wo_ans.format(examples['question'][idx])
                
        #     input_prompts.append(prompt)

        # model_inputs = tokenizer(
        #     input_prompts,
        #     max_length=input_max_length,
        #     truncation=True,
        #     # padding="max_length",
        #     padding=True,
        # )

        # # with tokenizer.as_target_tokenizer():
        # labels = tokenizer(
        #     [random.choice(pa) for pa in examples['possible_answers']],
        #     max_length=output_max_length,
        #     truncation=True,
        #     # padding="max_length",
        #     padding=True,
        # )

        # model_inputs["labels"] = labels["input_ids"]
        
        ### === version 2 
        for idx in range(len_dataset):
            # print([completion_template_with_ans.format(examples['question'][idx], pa) for pa in examples['possible_answers'][idx]])
            input_prompts.extend(
                [completion_template_with_ans.format(examples['question'][idx], pa) for pa in examples['possible_answers'][idx]]
            )
            
        model_inputs = tokenizer(
            input_prompts,
            # max_length=input_max_length,
            # truncation=True,
            # # padding="max_length",
            # padding=True,
        )
        return model_inputs

    def tokenize_wrapper(split_name):
        def wrapper(examples):
            return qa_tokenize_function(examples, split_name)
        return wrapper

    tokenized_train_datasets = {}
    for split in raw_dataset.keys():
        tokenized_train_datasets[split] = raw_dataset[split].map(
            tokenize_wrapper(split),
            batched=True,
            remove_columns=raw_dataset[split].column_names,
            desc=f"Running tokenizer on {split} dataset"
        )
    # tokenized_datasets = raw_dataset.map(
    #     qa_tokenize_function,
    #     batched=True,
    #     remove_columns=["question", "possible_answers"],
    #     desc="Running tokenizer on dataset",
    # )
    print(tokenized_train_datasets)
    
    # = Print a sample of dataset
    input_text = tokenizer.decode(tokenized_train_datasets['train'][0]["input_ids"], skip_special_tokens=True)
    # label_text = tokenizer.decode(tokenized_train_datasets['train'][0]["labels"], skip_special_tokens=True)

    print(input_text)
    # print(label_text)
    
    # Load test_set
    if dataset_name in ['popQA', 'EQ']:
            
        test_data = load_json_file(selected_files['test'])
        test_subset_size = int(subset_percentage * len(test_data))
        subset_test_data = random.sample(test_data, test_subset_size)    
        test_questions = [item['question'] for item in subset_test_data]
        test_answers = [item['answers'] for item in subset_test_data]
    
        return tokenized_train_datasets, (test_questions, test_answers)
    
    return tokenized_train_datasets, (val_questions, val_answers)
    
def load_training_args(args):
    model_dir = "./models"
    repo_name = "HeydarS/{}_eq_qlora_v{}".format(args.model_name_or_path.split('/')[-1], args.version)
    output_dir = os.path.join(model_dir, repo_name.split('/')[-1])
        
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit", # "paged_adamw_8bit"
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0,
        lr_scheduler_type="linear",
        report_to=[],
        
        fp16=True,

        save_strategy="epoch",
        save_total_limit=2,
        # save_steps=1,
        # logging_steps=1,
        # fp16=True,
        # max_steps=200,
        # load_best_model_at_end=False,
    )
    
    return training_arguments

def inference_on_testset(
    model,
    tokenizer,
    test_questions,
    test_answers,
    relation_files,
    selected_relations,
    device,
    with_fs=True
): 
    num_samples_per_relation = 1
    
    model.eval()
    max_new_tokens=8
    accuracy = []
    for idx, query in enumerate(test_questions):
                
        if with_fs:
            few_shot_examples = create_few_shot_examples(
                relation_files,
                selected_relations,
                num_samples_per_relation,
                'test' if dataset_name in ['EQ', 'popQA'] else 'dev'
            )
            np.random.shuffle(few_shot_examples)
            few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"
        else:
            few_shot_examples_text = "\n\n"
        
        prompt = few_shot_examples_text + completion_template_wo_ans.format(query)
        
        inpts = tokenizer(prompt, return_tensors="pt").to(device)
        inpt_decoded = tokenizer.decode(inpts["input_ids"][0, :])
        
        with torch.no_grad():
            gen = model.generate(
                input_ids=inpts["input_ids"],
                attention_mask=inpts["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False
            )
        text = tokenizer.decode(gen[0])
        
        print(text)
        
        pred = text[2+len(prompt):]
        # pred = pred[5:]
        pred = pred.split("\n")[0]

        # if idx % 15 == 0:
        print('Pred: {}'.format(pred))
        print('Labels: {}'.format(test_answers[idx]))
            # print('\n\n')
        
        is_correct = False
        for pa in test_answers[idx]:
            if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                is_correct = True
        accuracy.append(is_correct)

    acc = sum(accuracy) / len(accuracy)
    print(f"Accuracy: {acc * 100:.2f}%")
    
def main(args):
    with_peft = False
    with_fs = False
    args.repo_name = "HeydarS/{}_{}_v{}".format(
        args.model_name_or_path.split('/')[-1],
        'peft' if with_peft else 'no_peft',
        args.version
    )
    output_dir = os.path.join(args.output_dir, args.repo_name.split('/')[-1])
    # model_name_or_path = "facebook/opt-350m"
    
    # set_seed(42)
    model, tokenizer = load_model(args, with_peft=with_peft)
    
    relation_files, selected_relations, selected_files = load_relations_data(args)
    
    tokenized_train_datasets, (test_questions, test_answers) = load_dataset(
        tokenizer,
        relation_files,
        selected_relations,
        selected_files,
        with_fs
    )
    
    training_arguments = load_training_args(args)
    
    # === Training process ==============
    # def preprocess_logits_for_metrics(logits, labels):
    #     if isinstance(logits, tuple):
    #         logits = logits[0]
    #     return logits.argmax(dim=-1)
    
    # metric = evaluate.load("accuracy")
    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds

    #     print(labels)
    #     print(preds)

    #     labels = labels[:, 1:].reshape(-1)
    #     preds = preds[:, :-1].reshape(-1)

    #     print(labels)
    #     print(preds)
    #     print('\n')
    #     return metric.compute(predictions=preds, references=labels)

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_train_datasets["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    
    print("Inference before fine-tuning ....")
    inference_on_testset(
        model,
        tokenizer,
        test_questions,
        test_answers,
        relation_files,
        selected_relations,
        device,
        with_fs
    )
    
    print('\n\n')
    print("Fine-tuning ....")
    trainer.train()
    # model.save_pretrained(output_dir)
    # model.push_to_hub(args.repo_name, token=True)
    
    print("Inference after fine-tuning ....")
    inference_on_testset(
        model,
        tokenizer,
        test_questions,
        test_answers,
        relation_files,
        selected_relations,
        device,
        with_fs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--version", default=1, type=int)
    
    args = parser.parse_args()
    main(args)
