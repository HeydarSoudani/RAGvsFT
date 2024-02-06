#!/usr/bin/env python3

from transformers import AutoTokenizer
from transformers import pipeline
import re
import torch
import argparse, json, os


output_dir = "component0_preprocessing/generated_data/popQA_EQformat"
corpus_dir = f"{output_dir}/corpus_all"
corpus_dir = f"{output_dir}/corpus_summary"
train_dir = f"{output_dir}/train" 
dev_dir = f"{output_dir}/dev" 
qrels_train_dir = f"{output_dir}/qrels-train" 
max_tokens = 512
dev_split = 0.1

def extract_json_objects(text):
    # Define a regular expression pattern for JSON objects
    # This is a simplistic pattern and might need adjustments for complex cases
    pattern = r'\{(?:[^{}]|(?R))*\}'
    
    # Find all substrings that match the JSON pattern
    json_strings = re.findall(pattern, text)
    
    # Parse the JSON strings into Python dictionaries
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

def main():
    
    # Load model
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    prompt_qa_generation = lambda context: f"""
    You are a question-answer generator. Your goal is to generate question-answer pairs given the Context.

    Example output:
    {{“question”: “What is George Rankin's occupation?”, “answer”: “politician”}}

    ===

    Context:
    {context}

    ===

    Step 1: Extract all entities that have potential for being an answer, identify as many as possible.
    Step 2: For each identified entity, generate a question.
    Step 3: Output in JSON format following the example above (i.e., `{{...}}`).
    Ensure that you distinctly label and delineate Steps 1, 2, and 3. Let's think step by step: 
    """.replace('    ', '')
    

    for corpus_file in os.listdir(corpus_dir):
        if corpus_file.endswith('.corpus.json'):
            
            relation_id = corpus_file.split('.')[0]
            print(f"Processing corpus file: {corpus_file}")
            print(f"Processing relation file: {relation_id}")
            query_id_counter = 0
            
            with open(f'{corpus_dir}/{corpus_file}', 'r', encoding='utf-8') as cf:
                data = json.load(cf)
                
                all_qas = []
                qrels_train = []
                for idx, item in enumerate(data):
                    
                    if idx == 5:
                        break
                    
                    context = item['content']
                    doc_id = item['doc_id']
                    _prompt = [
                        { "role": "system", "content": "\n"},
                        { "role": "user", "content": prompt_qa_generation(truncate_text(context, 256))}
                    ]
                    
                    prompt = pipe.tokenizer.apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
                    outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
                    new_pt = outputs[0]["generated_text"]
                    qas = extract_json_objects(new_pt)
                    
                    if qas is not None:
                        print(qas)
                        for question, answer in qas:                                
                            
                            all_qas.append({
                                'query_id': f"qa_{relation_id}_{query_id_counter}",
                                'question': question,
                                'answers': [answer]
                            })
                            qrels_train.append({
                                'query_id': f"qa_{relation_id}_{query_id_counter}",
                                'doc_id': doc_id,
                                'score': 1
                            })
                            query_id_counter += 1
                                        

            with open(f'{train_dir}/{relation_id}.train.json', 'w', encoding='utf-8') as tf:
                json.dump(all_qas, tf, indent=4)
            
            with open(f'{qrels_train_dir}/{relation_id}.qrels-train.json', 'w', encoding='utf-8') as qf:
                json.dump(qrels_train, qf, indent=4)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    main()