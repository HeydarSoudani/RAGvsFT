#!/usr/bin/env python3

import random
import requests
from io import BytesIO
from zipfile import ZipFile
from datasets import Dataset, DatasetDict
import argparse, os, json
from itertools import chain
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset, DatasetDict
import evaluate

from huggingface_hub import HfFolder, HfApi

import numpy as np
import torch

print("Available GPUs:", torch.cuda.device_count())

os.environ["WANDB_MODE"] = "offline"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda:0'
dataset_name = 'popQA' # [TQA, popQA, EQ]
completion_template_wo_ans = "Q: {} A:"
completion_template_with_ans = "Q: {} A: {}"
dev_split = 0.1
with_peft = True
with_fs = False
# with_rag = False
training_style = 'qa' # ['clm', 'qa']
# target_relation_ids = ["91", "106", "22", "182"]
target_relation_ids = ["22", "218", "91", "257", "182", "164", "526", "97", "533", "639", "472", "106", "560", "484", "292", "422"]
# target_relation_ids = ["91"]

subset_percentage = 1.0
if dataset_name == "TQA":
    num_relations = 1
else:
    num_relations = 15


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

def load_json_files(relation_files, split_name):
    json_data = {}
    for relation_id, files in relation_files.items():
        split_file = next((file for file in files if split_name in file), None)
        with open(split_file, 'r') as file:
            json_data[relation_id] = json.load(file)
    return json_data

def load_model(args, with_peft=False):
    
    if not with_peft:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            # device_map={"": 0},
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
            # device_map={"": 0},
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
    # tokenizer.padding_side = 'left'
    
    return model, tokenizer

# def create_few_shot_examples(
#     relation_files,
#     query_relation,
#     num_samples,
#     split_name
# ):
    
#     # relation_files.pop(query_relation)
#     # fewshot_relations = random.sample(relation_files.keys(), num_relations)
#     keys_to_sample = [key for key in relation_files.keys() if key != query_relation]
#     fewshot_relations = random.sample(keys_to_sample, num_relations)
    
#     few_shot_examples = []
    
#     for relation_id in fewshot_relations:
#         files = relation_files[relation_id]
#         split_file = next((file for file in files if split_name in file), None)
#         data = load_json_file(split_file)
        
#         sampled_examples = random.sample(data, min(num_samples, len(data)))
#         for example in sampled_examples:
            
#             if dataset_name in ['EQ', 'popQA']:
#                 question = example['question']
#                 answers = example['answers']
#             elif dataset_name == 'TQA':
#                 question = example['Question']
#                 answers = example['Answer']['NormalizedAliases']
            
#             completion = completion_template_with_ans.format(question, random.choice(answers))
#             few_shot_examples.append(completion)
#     return few_shot_examples

def format_example(example, dataset_name):
    if dataset_name in ['EQ', 'popQA']:
        return example['question'], example['answers']
    elif dataset_name == 'TQA':
        return example['Question'], example['Answer']['NormalizedAliases']

def create_few_shot_examples(relation_id, data, num_samples):
    few_shot_examples = []
    sampled_examples = random.sample(data[relation_id], min(num_samples, len(data[relation_id])))
    for example in sampled_examples:
        # Format the example based on the dataset
        question, answers = format_example(example, dataset_name)
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
    # test_relation_id = random.choice(list(relation_files.keys()))
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
    
    # Other relations ===============
    # if dataset_name in ['EQ', 'popQA']:
    #     relation_files.pop(test_relation_id)
    # fewshot_relations = random.sample(relation_files.keys(), num_relations)

    return test_relation_ids, test_files, relation_files
    
def load_dataset_qa(tokenizer, test_files):
    
    ### === Train part ================================ 
    train_data = []
    # for file in test_files['train']:
    for file in test_files['test']:
        train_data.extend(load_json_file(file))    
    dev_data = []
    for file in test_files['dev']:
        dev_data.extend(load_json_file(file))    
    
    # train_data = load_json_file(test_files['train'])
    # dev_data = load_json_file(test_files['dev'])

    train_subset_size = int(subset_percentage * len(train_data))
    subset_train_data = random.sample(train_data, train_subset_size)
    dev_subset_size = int(subset_percentage * len(dev_data))
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
    
    def qa_tokenize_function(examples):
        input_prompts = []
        len_dataset = len(examples['question'])
    
        for idx in range(len_dataset):
            input_prompts.extend(
                [completion_template_with_ans.format(examples['question'][idx], pa) for pa in examples['possible_answers'][idx]]
            )    
        model_inputs = tokenizer(input_prompts)
        return model_inputs
    
    tokenized_train_datasets = raw_dataset.map(
        qa_tokenize_function,
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
    print(input_text)
    # label_text = tokenizer.decode(tokenized_train_datasets['train'][0]["labels"], skip_special_tokens=True)
    # print(label_text)
    
    ### === Test part ================================= 
    if dataset_name in ['popQA', 'EQ']:        
        
        test_data = []
        # test_data = load_json_file(test_files['test'])
        for file in test_files['test']:
            relation_id = file.split('/')[-1].split('.')[0]
            data = load_json_file(file)
            for item in data:
                item['relation_id'] = relation_id
            test_data.extend(data) 
        
        test_subset_size = int(subset_percentage * len(test_data))
        subset_test_data = random.sample(test_data, test_subset_size)    
        test_questions = [(item['query_id'], item['question'], item['pageviews'], item['relation_id']) for item in subset_test_data]
        test_answers = [item['answers'] for item in subset_test_data]
    
        return tokenized_train_datasets, (test_questions, test_answers)
    
    return tokenized_train_datasets, (val_questions, val_answers)

def load_dataset_corpus(tokenizer, test_relation_id, test_files):
    subset_percentage = 1.0
    max_pos_embeddings = 1024
    
    corpus_path = f"{args.data_dir}/corpus/{test_relation_id}.corpus.json"
    with open(corpus_path, 'r') as in_file:
        corpus_data = json.load(in_file)
    corpus_content_data = [item["content"] for item in corpus_data]
    
    random.shuffle(corpus_content_data)
    split_index = int(len(corpus_content_data) * dev_split)
    dev_content = corpus_content_data[:split_index]
    train_content = corpus_content_data[split_index:]
    
    train_subset_size = int(subset_percentage * len(train_content))
    subset_train_data = random.sample(train_content, train_subset_size)
    dev_subset_size = int(subset_percentage * len(dev_content))
    subset_dev_data = random.sample(dev_content, dev_subset_size)
    
    raw_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "text": subset_train_data,
        }),
        'dev': Dataset.from_dict({
            "text": subset_dev_data,
        })
    })
    print(raw_dataset)
    
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
        print(
            f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
        )
        if max_pos_embeddings > 0:
            block_size = min(1024, max_pos_embeddings)
        else:
            block_size = 1024
    
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
    
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_train_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    
    ### === Test part =================================         
    if dataset_name in ['popQA', 'EQ']:  
        test_data = load_json_file(test_files['test'])
        test_subset_size = int(subset_percentage * len(test_data))
        subset_test_data = random.sample(test_data, test_subset_size)    
        test_questions = [(item['query_id'], item['question'], item['pageviews']) for item in subset_test_data]
        test_answers = [item['answers'] for item in subset_test_data]
    
    return tokenized_train_datasets, (test_questions, test_answers)

def load_training_args(args):
    repo_name = "HeydarS/{}_ft_v{}".format(args.model_name_or_path.split('/')[-1], args.version)
    output_dir = os.path.join(args.output_model_dir, repo_name.split('/')[-1])
        
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
    test_relation_ids,
    relation_files,
    device,
    args,
    with_fs,
    prefix="bf",
    with_rag=False
):
    model.eval()
    
    # Create results dir
    out_results_dir = f"{args.output_result_dir}/results"
    os.makedirs(out_results_dir, exist_ok=True)
    model_name = args.model_name_or_path.split('/')[-1]
    str_rels = '_'.join(test_relation_ids)
    out_results_path = f"{out_results_dir}/{str_rels}.{model_name}.{prefix}_results.jsonl"
    
    if with_rag:
        # Get retrieval results
        ret_results = []
        ret_results_dir = f"{args.data_dir}/retrieved"
        
        for test_relation_id in test_relation_ids:
            ret_results_path = f"{ret_results_dir}/{test_relation_id}.ret_results.jsonl"
            with open (ret_results_path, 'r') as file:
                ret_results.extend([json.loads(line) for line in file])
   
   
    # Load JSON files once
    loaded_json_data = load_json_files(relation_files, 'test' if dataset_name in ['EQ', 'popQA'] else 'dev')
    
    num_samples_per_relation = 1
    max_new_tokens=15
    accuracy = []
    
    all_results = []
    for idx, (query_id, query, query_pv, query_relation) in enumerate(test_questions):
                
        # few_shot_examples_text = ""
        # if with_fs:
        #     few_shot_examples = create_few_shot_examples(
        #         relation_files,
        #         query_relation,
        #         num_samples_per_relation,
        #         'test' if dataset_name in ['EQ', 'popQA'] else 'dev'
        #     )
        #     np.random.shuffle(few_shot_examples)
        #     few_shot_examples_text = "\n\n".join(few_shot_examples)
        
        few_shot_examples_text = ""
        if with_fs:
            few_shot_examples = []
            keys_to_sample = [key for key in relation_files.keys() if key != query_relation]
            fewshot_relations = random.sample(keys_to_sample, num_relations)
            for relation_id in fewshot_relations:
                sampled_examples = random.sample(loaded_json_data[relation_id], min(num_samples_per_relation, len(loaded_json_data[relation_id])))
                for example in sampled_examples:
                    question, answers = format_example(example, dataset_name)
                    completion = completion_template_with_ans.format(question, random.choice(answers))
                    few_shot_examples.append(completion)
            random.shuffle(few_shot_examples)
            few_shot_examples_text = "\n\n".join(few_shot_examples)
        
        
        retrieved_text = ""
        if with_rag:
            for ret_result in ret_results:
                if ret_result['id'] == query_id:
                    retrieved_text = ret_result['ctxs'][0]['text']
                    break
            if retrieved_text == "":
                print("No retrieved text found for query: {}".format(query))
        
        prompt = few_shot_examples_text + "\n\n" + retrieved_text + "\n\n" + completion_template_wo_ans.format(query)
        
        inpts = tokenizer(prompt, return_tensors="pt").to(device)
        inpt_decoded = tokenizer.decode(inpts["input_ids"][0, :])
        
        generation_config = GenerationConfig(
            num_beams=4,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            
            do_sample=False,
            early_stopping=True,
            decoder_start_token_id=0,
            # eos_token_id=tokenizer.eos_token_id,
            # pad_token=model.config.pad_token_id,
            # num_beams=4,
        )
        
        
        with torch.no_grad():
            gen = model.generate(
                **inpts,
                generation_config=generation_config,
                # input_ids=inpts["input_ids"],
                # attention_mask=inpts["attention_mask"],
                # pad_token_id=tokenizer.eos_token_id,
                # max_new_tokens=max_new_tokens,
                # num_beams=1,
                # do_sample=False
            )
        text = tokenizer.decode(gen[0])
        
        # print(text)
        
        pred = text[2+len(prompt):]
        # pred = pred[5:]
        pred = pred.split("\n")[0]

        is_correct = False
        for pa in test_answers[idx]:
            if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                is_correct = True
        accuracy.append(is_correct)
        
        if idx % 100 == 0:
            print('Query: {}'.format(query))
            print('Pred: {}'.format(pred))
            print('Labels: {}'.format(test_answers[idx]))
            print('Final decision: {}'.format(is_correct))
            # print('====')
        
        # Write to file
        all_results.append({
            "query_id": query_id,
            "question": query,
            "possible_answers": test_answers[idx],
            "pred": pred,
            "is_correct": is_correct,
            "pageviews": query_pv
        })
        
    with open(out_results_path, 'w') as file:
        for item in all_results:
            file.write(json.dumps(item) + '\n')

    acc = sum(accuracy) / len(accuracy)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("===========================")
    print('\n')

def main(args):
    args.repo_name = "HeydarS/{}_{}_v{}".format(
        args.model_name_or_path.split('/')[-1],
        'peft' if with_peft else 'no_peft',
        args.version
    )
    repo_name = "HeydarS/{}_ft_v{}".format(args.model_name_or_path.split('/')[-1], args.version)
    output_dir = os.path.join(args.output_model_dir, repo_name.split('/')[-1])
    
    set_seed(42)
    model, tokenizer = load_model(args, with_peft=with_peft)
    
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    
    if training_style == 'qa':
        tokenized_train_datasets, (test_questions, test_answers) = load_dataset_qa(
            tokenizer,
            test_files
        )
    elif training_style == 'clm':
        tokenized_train_datasets, (test_questions, test_answers) = load_dataset_corpus(
            tokenizer,
            test_relation_ids,
            test_files
        )
    
    training_arguments = load_training_args(args)
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
    
    # print("Inference before fine-tuning & without RAG ....")
    # inference_on_testset(
    #     model,
    #     tokenizer,
    #     test_questions,
    #     test_answers,
    #     test_relation_ids,
    #     relation_files,
    #     device,
    #     args,
    #     with_fs,
    #     prefix="bf_norag",
    #     with_rag=False
    # )
    
    # print("Inference before fine-tuning & with RAG ....")
    # inference_on_testset(
    #     model,
    #     tokenizer,
    #     test_questions,
    #     test_answers,
    #     test_relation_ids,
    #     relation_files,
    #     device,
    #     args,
    #     with_fs,
    #     prefix="bf_rag",
    #     with_rag=True
    # )
    
    print('\n\n')
    print("Fine-tuning ....")
    trainer.train()
    model.save_pretrained(output_dir)
    model.push_to_hub(args.repo_name, token=True)
    
    # print("Inference after fine-tuning & without RAG ....")
    # inference_on_testset(
    #     model,
    #     tokenizer,
    #     test_questions,
    #     test_answers,
    #     test_relation_ids,
    #     relation_files,
    #     device,
    #     args,
    #     with_fs,
    #     prefix="af_norag",
    #     with_rag=False
    # )
    
    # print("Inference after fine-tuning & with RAG ....")
    # inference_on_testset(
    #     model,
    #     tokenizer,
    #     test_questions,
    #     test_answers,
    #     test_relation_ids,
    #     relation_files,
    #     device,
    #     args,
    #     with_fs,
    #     prefix="af_rag",
    #     with_rag=True
    # )

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
