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
from torch.utils.data import Dataset, DataLoader



os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    device = 'cuda:0'
    completion_template = "Q: {} A:"
    model_name_or_path = "facebook/opt-125m"
    
    ### === Defining model ====================
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map={"": 0},
        quantization_config=bnb_config,
        # trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    
    ### === Introducing PEFT to the model ====
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
        model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token


    ### === Loading dataset =======================
    def load_json_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    # Download and unzip the dataset
    url = "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip"
    response = requests.get(url)
    zip_file = ZipFile(BytesIO(response.content))
    zip_file.extractall("entity_questions_dataset")
    zip_file.close()

    # List available relation IDs in each subfolder
    data_dir = "entity_questions_dataset/dataset"
    subfolders = ['train', 'dev', 'test']
        
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = os.path.join(data_dir, subfolder)
        for file in os.listdir(subfolder_path):
            relation_id = file.split('.')[0]
            if relation_id not in relation_files:
                relation_files[relation_id] = []
            relation_files[relation_id].append(os.path.join(subfolder_path, file))    

    # Selected file =================
    selected_relation_id = random.choice(list(relation_files.keys()))

    selected_files = {}
    for subfolder in subfolders:
        subfolder_path = os.path.join(data_dir, subfolder)
        for file in os.listdir(subfolder_path):
            if file.startswith(selected_relation_id):
                selected_files[subfolder] = os.path.join(subfolder_path, file)

    print("Selected Relation ID:", selected_relation_id)
    # print("Selected Files:", selected_files)

    # Other relations ===============
    relation_files.pop(selected_relation_id)

    num_relations = 4
    num_samples_per_relation = 1

    selected_relations = random.sample(relation_files.keys(), num_relations)

    def create_few_shot_examples(relation_files, num_samples, split_name):
        few_shot_examples = []
        
        for relation_id in selected_relations:
            files = relation_files[relation_id]
            split_file = next((file for file in files if "train" in file), None)
            data = load_json_file(split_file)
            
            sampled_examples = random.sample(data, min(num_samples, len(data)))
            for example in sampled_examples:
                completion = "Q: {} A: {}".format(example['question'], random.choice(example['answers']))
                few_shot_examples.append(completion)
        return few_shot_examples


    def load_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    # Load data from selected files
    train_data = load_data(selected_files['train'])
    dev_data = load_data(selected_files['dev'])
    test_data = load_data(selected_files['test'])

    # Select a subset of the training data randomly (e.g., 10%)
    subset_percentage = 0.05

    train_subset_size = int(subset_percentage * len(train_data))
    subset_train_data = random.sample(train_data, train_subset_size)

    dev_subset_size = int(subset_percentage * len(dev_data))
    subset_dev_data = random.sample(dev_data, dev_subset_size)

    test_subset_size = int(subset_percentage * len(test_data))
    subset_test_data = random.sample(test_data, test_subset_size)


    # Extract questions and answers
    train_questions = [item['question'] for item in subset_train_data]
    train_answers = [item['answers'] for item in subset_train_data]
    val_questions = [item['question'] for item in subset_dev_data]
    val_answers = [item['answers'] for item in subset_dev_data]
    test_questions = [item['question'] for item in subset_test_data]
    test_answers = [item['answers'] for item in subset_test_data]

    # Create the dataset in the desired format
    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "question": train_questions,
            "possible_answers": train_answers
        }),
        'validation': Dataset.from_dict({
            "question": val_questions,
            "possible_answers": val_answers
        }),
    #     'test': Dataset.from_dict({
    #         "question": test_questions,
    #         "possible_answers": test_answers
    #     })
    })
    print(raw_dataset)
    
    input_max_length = 1024
    output_max_length = 4

    def qa_tokenize_function(examples, split_name):
        input_prompts = []
        len_dataset = len(examples['question'])
        for idx in range(len_dataset):
            few_shot_examples = create_few_shot_examples(
                relation_files,
                num_samples_per_relation,
                split_name
            )
            np.random.shuffle(few_shot_examples)
            few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"
            prompt = few_shot_examples_text + completion_template.format(examples['question'][idx])
                
            input_prompts.append(prompt)

        model_inputs = tokenizer(
            input_prompts,
            max_length=input_max_length,
            truncation=True,
            padding="max_length",
    #         padding=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                [random.choice(pa) for pa in examples['possible_answers']],
                max_length=output_max_length,
                truncation=True,
                padding="max_length",
    #             padding=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_wrapper(split_name):
        def wrapper(examples):
            return qa_tokenize_function(examples, split_name)
        return wrapper


    tokenized_datasets = {}
    for split in raw_dataset.keys():
        tokenized_datasets[split] = raw_dataset[split].map(
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

    print(tokenized_datasets)
    
    
    # = Print a sample of dataset
    input_text = tokenizer.decode(tokenized_datasets['train'][0]["input_ids"], skip_special_tokens=True)
    label_text = tokenizer.decode(tokenized_datasets['train'][0]["labels"], skip_special_tokens=True)

    print(input_text)
    print('\n')
    print(label_text)
    
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
    
    # === Training process ==============
    epochs = 5
    version = 1
    model_dir = "./models"
    repo_name = "HeydarS/{}_eq_qlora_v{}".format(model_name_or_path.split('/')[-1], version)
    output_dir = os.path.join(model_dir, repo_name.split('/')[-1])

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
        num_train_epochs=epochs,
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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        # eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    
    trainer.train()
    # model.save_pretrained(model_save_path)
    # model.push_to_hub(repo_name, token=True)
    
    
    ### === Inference on testset =============== 
    # def inference(model, tokenizer):
    print("Inference before fine-tuning ....")
    model.eval()
    max_new_tokens=15
    accuracy = []
    for idx, query in enumerate(test_questions):
        
        few_shot_examples = create_few_shot_examples(relation_files, num_samples_per_relation, 'test')
        np.random.shuffle(few_shot_examples)
        few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"
        prompt = few_shot_examples_text + completion_template.format(query)
        
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
        pred = text[len(prompt):]
        pred = pred[5:]
        pred = pred.split("\n")[0]

        print('Pred: {}'.format(pred))
        print('Labels: {}'.format(test_answers[idx]))
        
        is_correct = False
        for pa in test_answers[idx]:
            if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                is_correct = True
        accuracy.append(is_correct)
        print('\n\n')

    acc = sum(accuracy) / len(accuracy)
    print(f"Accuracy: {acc * 100:.2f}%")
    
    
    print("Fine-tuning ....")
    trainer.train()
    
    
    ### === Inference on testset =============== 
    # def inference(model, tokenizer):
    print("Inference after fine-tuning ....")
    model.eval()
    max_new_tokens=15
    accuracy = []
    for idx, query in enumerate(test_questions):
        
        few_shot_examples = create_few_shot_examples(relation_files, num_samples_per_relation, 'test')
        np.random.shuffle(few_shot_examples)
        few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"
        prompt = few_shot_examples_text + completion_template.format(query)
        
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
        pred = text[len(prompt):]
        pred = pred[5:]
        pred = pred.split("\n")[0]

        print('Pred: {}'.format(pred))
        print('Labels: {}'.format(test_answers[idx]))
        
        is_correct = False
        for pa in test_answers[idx]:
            if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                is_correct = True
        accuracy.append(is_correct)
        print('\n\n')

    acc = sum(accuracy) / len(accuracy)
    print(f"Accuracy: {acc * 100:.2f}%")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--version", default=1, type=int)
    
    args = parser.parse_args()
    main(args)
