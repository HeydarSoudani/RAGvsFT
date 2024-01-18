#!/usr/bin/env python3

import argparse, json, os
import re
from typing import List
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def remove_parentheses(input_file, output_file):
    with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
        for idx, line in enumerate(in_file):
            # if idx == 200:
            #     break
            
            text = json.loads(line)
            cleaned_text = re.sub(r'\([^)]*\)\s*', '', text['contents'])
            
            cleaned_obj = {"id": text["id"], "contents": cleaned_text }
            out_file.write(json.dumps(cleaned_obj) + "\n")

def count_tokens(sentence: str) -> int:
    return len(sentence.split())

def split_text_to_sentences(text: str, max_tokens: int) -> List[str]:
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

def splitting_corpus(input_file, output_file, token_num):
    
    org_qrels_file_path = 'component0_preprocessing/generated_data/popQA_religion/qrels.jsonl'
    new_qrels_file_path = 'component0_preprocessing/generated_data/popQA_religion/qrels_512token.jsonl'
    
    original_qrels = []
    with open(org_qrels_file_path, 'r') as file:
        for line in file:
            original_qrels.append(json.loads(line))
    
    with open(input_file, 'r') as wop_file, open(output_file, 'w') as sp_file, open(new_qrels_file_path, 'w') as new_qrel_file:
        for idx, line in enumerate(wop_file):
            
            text = json.loads(line)
            doc_id = text["id"]
               
            query_id = None
            score = None 
            for qrel in original_qrels:
                if qrel['doc_id'] == doc_id:
                    query_id, score = qrel['query_id'], qrel['score']
            
            query_id_score_pairs = []
            
            split_text = split_text_to_sentences(text["contents"], token_num)
            for i, split in enumerate(split_text):
                new_doc_id = doc_id+'-'+str(i)
                sp_obj = {"id": new_doc_id, "contents": split}
                sp_file.write(json.dumps(sp_obj) + "\n")
                
                qsp_obj = {"query_id": query_id, "doc_id": new_doc_id, "score": score}
                new_qrel_file.write(json.dumps(qsp_obj) + "\n")               

def remove_according_to_types(input_file, output_file):

    phrases_to_exclude = ["when ", "how old", "how many", "how much"]

    with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
        for line in in_file:
            data = json.loads(line)

            if not any(data['question'].lower().startswith(phrase) for phrase in phrases_to_exclude):
                
                json.dump(data, output_file)
                out_file.write('\n')


def main(args):
    # input_file = "component0_preprocessing/generated_data/popQA_costomized/corpus.jsonl"
    input_file = "component0_preprocessing/generated_data/popQA_sm/filtered_corpus.jsonl"
    
    wo_parentheses_file = "component3_QAGeneration/generated_data/corpus_wo_parentheses.jsonl"
    splitted_file = "component3_QAGeneration/generated_data/filtered_corpus_splitted_sm.jsonl"
    type_filtered_file = "component3_QAGeneration/generated_data/typed_filtered_corpus_splitted_sm.jsonl"
    token_num = 512
    
    
    input_file = "component0_preprocessing/generated_data/popQA_religion/corpus.jsonl"
    splitted_file = "component0_preprocessing/generated_data/popQA_religion/corpus_512token.jsonl"
    
    # Step 1: remove parentheses
    remove_parentheses(input_file, wo_parentheses_file)
          
    # Step 2: Remove unuseful texts
    # By hand
    
    # Step 3: Split in some chunks
    splitting_corpus(wo_parentheses_file, splitted_file, token_num)
    
    # Step 3: remove questions start with ...
    # remove_according_to_types(splitted_file, type_filtered_file)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", type=str, required=True)
    # parser.add_argument("--output_file", type=str, required=True)
    
    args = parser.parse_args()
    main(args)