import pandas as pd
import numpy as np
import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import default_data_collator, get_scheduler
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.optimization import Adafactor 
from accelerate import Accelerator
from datasets import load_dataset, Dataset, DatasetDict
from tqdm.auto import tqdm
import time
import warnings
import nltk
import string
import evaluate

from utils import load_json_file, read_tsv_column

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("Running on the GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
    print("Running on the mps")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


def EQ_dataset_preprocess(path):
    
    data = {}
    for item in ["train", "dev", "test"]:
        concatenated_data = []
        data[item] = {}
        try:
            folder_path = os.path.join(path, item)
            for filename in os.listdir(folder_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        json_data = json.load(file)
                    concatenated_data.extend(json_data)
        except Exception as e:
            print(f"Error concatenating JSON files: {e}")
        
        data[item]['question'] = [j['question'] for j in concatenated_data]
        data[item]['answers'] = [j['answers'][0] for j in concatenated_data]
    
    eq_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "question": data['train']['question'], 
            "answers": data['train']['answers']
        }),
        'validation': Dataset.from_dict({
            "question": data['dev']['question'], 
            "answers": data['dev']['answers']
        }),
        'test': Dataset.from_dict({
            "question": data['test']['question'], 
            "answers": data['test']['answers']
        })
    })
    return eq_dataset

def clean_text(text):
    sentences = nltk.sent_tokenize(text.strip())
    sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
    sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                    if len(sent) > 0 and
                                    sent[-1] in string.punctuation]
    text_cleaned = "\n".join(sentences_cleaned_no_titles)
    return text_cleaned

def compute_metrics(tokenizer, eval_pred):
    metric = evaluate.load("rouge")
    
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(
        [label[label != -100] for label in labels],
        skip_special_tokens=True
    )
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                    for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                    for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # # Extract ROUGE f1 scores
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # # Add mean generated length to metrics
    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
    #                 for pred in predictions]
    # result["gen_len"] = np.mean(prediction_lens)
    
    # return {k: round(v, 4) for k, v in result.items()}
    return result

def compute_accuracy(possible_answers, preds):
    accuracy = []
    # print(possible_answers)
    print(preds)
    
    for i in range (len(preds)):
        is_correct = False
        for pa in possible_answers[i]:
            if pa in preds[i] or pa.lower() in preds[i] or pa.capitalize() in preds[i]:
                is_correct = True
        
        accuracy.append(is_correct)
    
    return (sum(accuracy) / len(accuracy))*100

def dataset_preparation(tokenizer, batch_size):
    # Define variables
    prefix = "qa: "
    max_input_length = 64
    max_target_length = 16
    data_collator = DataCollator(tokenizer)
    subsample_train = 10000
    subsample_val = 2000
    
    # Define functions
    def preprocess_data(examples):
        inputs = [prefix + text for text in examples["question"]]
        model_inputs = tokenizer(
            inputs,
            padding="max_length",
            max_length=max_input_length,
            truncation=True
        )

        # with tokenizer.as_target_tokenizer():
        # If the dataset is SQUAD
        # first_answers = [obj['text'][0] for obj in examples["answers"]]
        
        labels = tokenizer(
            examples["answers"],
            padding="max_length",
            max_length=max_target_length, 
            truncation=True
        ).input_ids
        
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)
        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs
    
    # Load squad dataset
    # train_dataset = load_dataset('squad')
    # train_dataset = train_dataset.remove_columns(["id", "title", "context"])
    
    # Load EntityQuestion dataset
    train_dataset = EQ_dataset_preprocess('./data/dataset/EntityQuestions')
    print(train_dataset)
    
    train_dataset["train"] = train_dataset["train"].shuffle().select(range(subsample_train))
    train_dataset["validation"] = train_dataset["validation"].shuffle().select(range(subsample_val))
    
    tokenized_datasets = train_dataset.map(
        preprocess_data, batched=True, remove_columns=["question", "answers"]
    )
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    
    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=batch_size)
    
    return train_dataloader, valid_dataloader

def test_dataset_preparation(tokenizer, batch_size):
    # Define variables
    prefix = "qa: "
    max_input_length = 64
    max_target_length = 8
    data_collator = DataCollator(tokenizer)
    subsample_test = 40
    
    # Define functions
    def preprocess_data(examples):
        inputs = [prefix + text for text in examples["question"]]
        model_inputs = tokenizer(
            inputs,
            padding="max_length",
            max_length=max_input_length,
            truncation=True
        )

        # Setup the tokenizer for targets
        # with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["answers"],
            padding="max_length",
            max_length=max_target_length, 
            truncation=True
        ).input_ids
        
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)
        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs
    
    dataset_test_path = "./data/dataset/popQA.tsv"
    questions = read_tsv_column(dataset_test_path, 'question')
    possible_answers = read_tsv_column(dataset_test_path, 'possible_answers', dtype='list')
    # possible_answers = get_list_column_from_tsv(dataset_test_path, 'possible_answers')
    popqa_dataset = DatasetDict({
        'test': Dataset.from_dict({
            "question": questions, 
            "answers": [pa[0] for pa in possible_answers]
        })
    })
    popqa_dataset = popqa_dataset["test"].shuffle().select(range(subsample_test))
    tokenized_datasets = popqa_dataset.map(
        preprocess_data, batched=True, remove_columns=["question", "answers"]
    )
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    
    test_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=batch_size)
    return test_dataloader, possible_answers

def training_arguments(model, train_dataloader, eval_dataloader, num_train_epochs):
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    
    accelerator = Accelerator()
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    return model, optimizer, train_dataloader, eval_dataloader, accelerator, lr_scheduler, num_training_steps

def training_loop(
    model,
    tokenizer,
    optimizer,
    train_dataloader,
    eval_dataloader,
    accelerator,
    lr_scheduler,
    num_train_epochs,
    num_training_steps,
    batch_size):
    
    output_dir = "./data/saved-model"
    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(num_train_epochs):
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
        preds = []
        targets = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                generated_ids = model.generate(**batch)
            preds.extend(generated_ids)
            targets.extend(batch['labels'])
                
            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))

        print("ROUGE: {}".format(compute_metrics(tokenizer, (preds, targets))))
        losses = torch.cat(losses)
        losses = losses[: 32]
        
        # Save
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
  
def inference():
    print("test on PopQA ....")
    
    batch_size = 8
    model_path = "./data/saved-model"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    test_dataloader, possible_answers = test_dataset_preparation(tokenizer, batch_size)
    
    # Evaluation
    model.eval()
    losses = []
    preds = []
    targets = []
    progress_bar = tqdm(range(len(test_dataloader)))
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            generated_ids = model.generate(**batch)
            preds.extend(generated_ids)
            targets.extend(batch['labels'])
        progress_bar.update(1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print("Acc: {}".format(compute_accuracy(possible_answers, decoded_preds)))
    # print("ROUGE: {}".format(compute_metrics(tokenizer, (preds, targets))))
      
def main():
    # === Define variables ===========
    model_checkpoint = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
    model.to(device)
    
    num_train_epochs = 3
    batch_size = 16
    
    # === Prepare dataset ============
    # dataset_preparation(tokenizer, batch_size)
    # train_dataloader, eval_dataloader = dataset_preparation(tokenizer, batch_size)
    # # = test dataset
    # batch = next(iter(train_dataloader))
    # print(batch.keys())
    # print(tokenizer.decode(batch['input_ids'][0]))
    # labels = batch['labels'][0]
    # print(tokenizer.decode([label for label in labels if label != -100]))

    # # === Training arguments =========    
    # model, \
    # optimizer, \
    # train_dataloader, \
    # eval_dataloader, \
    # accelerator, \
    # lr_scheduler, \
    # num_training_steps = training_arguments(
    #     model, train_dataloader, eval_dataloader, num_train_epochs
    # )
    
    # # === Training loop ==============
    # training_loop(
    #     model,
    #     tokenizer,
    #     optimizer,
    #     train_dataloader,
    #     eval_dataloader,
    #     accelerator,
    #     lr_scheduler,
    #     num_train_epochs,
    #     num_training_steps,
    #     batch_size
    # )
    
    # === Inference ==================
    inference()

if __name__ == "__main__":
    main()
    