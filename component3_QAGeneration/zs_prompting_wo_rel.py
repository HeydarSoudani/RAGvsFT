#!/usr/bin/env python3

from transformers import AutoTokenizer
from transformers import pipeline
import re
import torch
import argparse, json, os
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import random

import nltk
nltk.download('punkt')

output_dir = "component0_preprocessing/generated_data/popQA_EQformat"
corpus_dir = f"{output_dir}/corpus_all"
# corpus_dir = f"{output_dir}/corpus_summary"

train_dir = f"{output_dir}/prompting/train2" 
dev_dir = f"{output_dir}/prompting/dev2" 
qrels_train_dir = f"{output_dir}/prompting/qrels-train2" 

os.makedirs(f"{output_dir}/prompting", exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(dev_dir, exist_ok=True)
os.makedirs(qrels_train_dir, exist_ok=True)

max_tokens = 512
dev_split = 0.1

def extract_json_objects(text):
    pattern = r'\{[^{}]*\}'
    json_strings = re.findall(pattern, text)
    
    json_objects = []
    for json_str in json_strings:
        try:
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    
    return json_objects

truncate_text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def truncate_text(text, max_tokens):
    
    tokens = truncate_text_tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    truncated_text = truncate_text_tokenizer.convert_tokens_to_string(tokens)
    return truncated_text

def split_text_to_sentences(text, max_tokens):
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Adding a period back to each sentence except the last one
        sentence = sentence.strip() if sentence != sentences[-1] else sentence.strip()
        sentence_length = len(sentence.split())

        # Check if adding the current sentence would exceed the maximum token count
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Add the current chunk to the chunks list and start a new chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def prompting_qa_generation(relation_id):
    
    # Load model
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    prompt_qa_generation = lambda context: f"""
    You are a question-answer generator. Your goal is to generate question-answer pairs given the Context.

    Example output: {{“question”: “”, “answer”: ""}}

    Context: {context}

    Step 1: Identifies spans that are likely to be answers to questions, identify as many as possible.
    Step 2: For each identified span, generate a question.
    Step 3: Respond to the question in only a few tokens concisely.
    Step 4: Output in JSON format following the example above (i.e., `{{...}}`).
    Ensure that you distinctly label and delineate Steps 1, 2, 3, and 4. Let's think step by step:
    """.replace('    ', '')
    

    # for corpus_file in os.listdir(corpus_dir):
    #     if corpus_file.endswith('.corpus.json'):
    #         relation_id = corpus_file.split('.')[0]
    # print(f"Processing corpus file: {corpus_file}")
    
    # print(f"Processing relation file: {relation_id}")
    query_id_counter = 0
    
    with open(f'{corpus_dir}/{relation_id}.corpus.json', 'r', encoding='utf-8') as cf:
        data = json.load(cf)
        
        all_qas = []
        qrels_train = []
        for item in tqdm(data, desc=f"Processing {relation_id} ..."):
        
        # for idx, item in enumerate(data):
            # if idx == 5:
            #     break
            
            context = item['content']
            doc_id = item['doc_id']
            
            max_tokens = 256
            chunks = split_text_to_sentences(context, max_tokens)
            for chunk in chunks:
            
                _prompt = [
                    { "role": "system", "content": "\n"},
                    { "role": "user", "content": prompt_qa_generation(chunk)}
                ]
            
                prompt = pipe.tokenizer.apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
                outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
                new_pt = outputs[0]["generated_text"]
                qas = extract_json_objects(new_pt)
            
                if qas is not None:
                    # print(qas)
                    for qa in qas:
                        if "question" in qa.keys() and "answer" in qa.keys():
                            # print("The question is: {}".format(qa["question"]))
                            # print("The answer is: {}".format(qa["answer"]))                     
                        
                            all_qas.append({
                                'query_id': f"qa_{relation_id}_{query_id_counter}",
                                'question': qa["question"],
                                'answers': [qa["answer"]]
                            })
                            qrels_train.append({
                                'query_id': f"qa_{relation_id}_{query_id_counter}",
                                'doc_id': doc_id,
                                'score': 1
                            })
                            query_id_counter += 1
                        else:
                            print("This QA object is missing either 'question' or 'answer' keys:", qa.keys())
    
    # Filtering step
    pattern = r'context\W'
    
    # filtered_qas = [qa for qa in all_qas if len(qa["question"].split()) >= 4]    
    filtered_qas = [
        qa for qa in all_qas 
        if isinstance(qa["question"], str) and isinstance(qa["answers"], list) and
            all(isinstance(answer, str) for answer in qa["answers"]) and
            len(qa["question"].split()) >= 4 and
            not re.search(pattern, qa["question"], re.IGNORECASE) and
            not any(re.search(pattern, answer, re.IGNORECASE) for answer in qa["answers"])
    ]
    
    random.shuffle(filtered_qas)
    split_index = int(len(filtered_qas) * dev_split)
    train_qas = filtered_qas[split_index:]
    dev_qas = filtered_qas[:split_index]

    with open(f'{train_dir}/{relation_id}.train.json', 'w', encoding='utf-8') as tf:
        json.dump(train_qas, tf, indent=4)
    
    with open(f'{dev_dir}/{relation_id}.dev.json', 'w', encoding='utf-8') as df:
        json.dump(dev_qas, df, indent=4)
    
    with open(f'{qrels_train_dir}/{relation_id}.qrels-train.json', 'w', encoding='utf-8') as qf:
        json.dump(qrels_train, qf, indent=4)

def post_filtering(relation_id):
    pattern = r'context\W'
    
    with open(f'{train_dir}/{relation_id}.train.json', 'r', encoding='utf-8') as cf:
        all_qas = json.load(cf)
        
    filtered_qas = [qa for qa in all_qas if len(qa["question"].split()) >= 4]
    filtered_qas = [
        qa for qa in filtered_qas 
        if isinstance(qa["question"], str) and isinstance(qa["answers"], list) and
            all(isinstance(answer, str) for answer in qa["answers"]) and
            not re.search(pattern, qa["question"], re.IGNORECASE) and
            not any(re.search(pattern, answer, re.IGNORECASE) for answer in qa["answers"])
    ]
    # if not re.search(pattern, qa["question"], re.IGNORECASE) and not re.search(pattern, qa["answers"][0], re.IGNORECASE)
    
    random.shuffle(filtered_qas)
    split_index = int(len(filtered_qas) * dev_split)
    train_qas = filtered_qas[split_index:]
    dev_qas = filtered_qas[:split_index]
    
    with open(f'{train_dir}/{relation_id}.filter.train.json', 'w', encoding='utf-8') as tf:
        json.dump(train_qas, tf, indent=4)
    
    with open(f'{dev_dir}/{relation_id}.filter.dev.json', 'w', encoding='utf-8') as df:
        json.dump(dev_qas, df, indent=4)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    
    # Done: 106, 22, 560, 218, 182, 97, 257, 164, 526, 639, 91, 533, 472, 292, 422
    # Doing: 484, 
    # To Do: 
    
    ### === Second round
    # Done: 
    # Doing: 
    # To Do: 182, 106, 22, 560, 218, 97, 257, 164, 526, 639, 91, 533, 472, 292, 422, 484
    relation_id = "472"
    prompting_qa_generation(relation_id=relation_id)
    
    # post_filtering(relation_id=relation_id)