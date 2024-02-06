#!/usr/bin/env python3

import argparse, json, os
import torch
from transformers import pipeline

def extract_tokens_after_key_tag(input_text, key_tag):
    # Find the start position of key tag
    start_pos = input_text.find(key_tag)
    if start_pos == -1:
        return "Key tag not found in the input text."
    
    # Move the start position to the end of the key tag to capture the token
    start_pos += len(key_tag)
    
    # Find the end position of the token by looking for the first dot (.) after the key tag
    end_pos = input_text.find('.', start_pos)
    
    # If a dot is not found, return the text until the end of the string
    if end_pos == -1:
        return input_text[start_pos:]
    else:
        # Extract and return the token
        return input_text[start_pos:end_pos]


def main(args):
    
    # Load model
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    data_path = "component0_preprocessing/generated_data/popQA_EQformat"
    corpus_dir = f"{data_path}/corpus_all"
    
    for corpus_file in os.listdir(corpus_dir):
        if corpus_file.endswith('.corpus.json'):
            
            prop_id = corpus_file.split('.')[0]
            print(f"Processing relation file: {prop_id}")
            
            with open(f'{corpus_dir}/{corpus_file}', 'r', encoding='utf-8') as cf:
                data = json.load(cf)
                for rel in data:
                    pass
    
    

    
    # Load data
    context = "John Mayne was a Scottish printer, journalist and poet born in Dumfries. In 1780, his poem The Siller Gun appeared in its original form in Ruddiman's Magazine, published by Walter Ruddiman in Edinburgh. It is a humorous work on an ancient custom in Dumfries of shooting for the \\\"Siller Gun.\\\" He also wrote a poem on Hallowe'en in 1780 which influenced Robert Burns's 1785 poem Halloween. Mayne also wrote a version of the ballad Helen of Kirkconnel. His verses were admired by Walter Scott."
    relation_name = "occupation"
    
    first_prompt = [
        {
            "role": "system",
            "content": "\n",
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nBased on the provided context, answer the following questions with a single Yes or No:\n1. Do the context mention a specific {relation_name}?"
        },
    ]
    
    second_prompt = [
        {
            "role": "system",
            "content": "\n",
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nExtract the exact sentences that mentioned {relation_name}. Please remember that you cannot generate the answer on your own but should only copy a continuous span from the original text."
        },
    ]
    
    prompt = pipe.tokenizer.apply_chat_template(first_prompt, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    res = outputs[0]["generated_text"]
    print(res)
    
    filtered_res = extract_tokens_after_key_tag(res, "<|assistant|>")
    
    if "Yes" in res:
        prompt = pipe.tokenizer.apply_chat_template(second_prompt, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        res = outputs[0]["generated_text"]
        print(res)
    
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    
    args = parser.parse_args()
    main(args)


