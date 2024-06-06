#!/usr/bin/env python3

import os
import sys
import nltk
import json
import torch
import argparse

nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

RELATIONS = {
    '22': 'Occupation',
    '91': 'Genre',
    '97': 'Capital of',
    '106': 'Religion',
    '164': 'Producer',
    '182': 'Country',
    '218': 'Place of birth',
    '257': 'Father',
    '292': 'Mother',
    '422': 'Capital',
    '472': 'Color',
    '484': 'Author',
    '526': 'Director',
    '533': 'Screenwriter',
    '560': 'Sport',
    '639': 'Composer'
}

print(os.getcwd())
import sys
sys.path.append(os.getcwd())
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

def split_text_by_tokens(text, token_count):
    tokens = word_tokenize(text)
    chunks = [tokens[i:i + token_count] for i in range(0, len(tokens), token_count)]
    chunked_texts = [' '.join(chunk) for chunk in chunks]
    return chunked_texts

def main(args):
    # Define the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Running on the GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
        print("Running on the mps")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    
    base_dir = f"component0_preprocessing/generated_data/{args.dataset_name}_costomized"
    # retrieved_passage_dir = f"{base_dir}/retrieved_passage/{args.retrieval_method}_{args.num_retrieved_passages}"
    retrieved_passage_dir = f"{base_dir}/retrieved_passage/{args.retrieval_method}"
    
    if args.output_path:
        reranked_sentences_dir = args.output_path
        os.makedirs(reranked_sentences_dir, exist_ok=True)
    else:
        # reranked_sentences_dir = f"{base_dir}/reranked_sentences/{args.retrieval_method}_{args.num_retrieved_passages}"
        reranked_sentences_dir = f"{base_dir}/reranked_sentences/{args.retrieval_method}"
        os.makedirs(f'{base_dir}/reranked_sentences', exist_ok=True)
        os.makedirs(reranked_sentences_dir, exist_ok=True)
    
    model = DRES(models.SentenceBERT(args.dense_model), batch_size=1, device=device)
    retriever = EvaluateRetrieval(model, score_function="dot", k_values=[1, 3, 5])
    
    for idx, filename in enumerate(os.listdir(retrieved_passage_dir)):
        # if idx == 1:
        #     break  
        if filename.endswith('.jsonl'):
            relation_id = filename.split('.')[0]
            input_file = f"{retrieved_passage_dir}/{filename}"
            output_file = f"{reranked_sentences_dir}/{relation_id}.{args.retrieval_method}.set_reranked.jsonl"
            print(f"\n\nProcessing {relation_id}...")
            
            with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
                for line in f_in:
                    data = json.loads(line.strip())
                    query_id = data['id']
                    query = data['question']
                    
                    
                    # === Step 1: Convert retrieved passages to a format that can be used by the dense retriever ===
                    corpus = {}
                    chunk_id = 0
                    for context in data['ctxs']:
                        doc_id = context['id']
                        
                        if args.split_type == 'word':
                            chunks = split_text_by_tokens(context['text'], args.word_num)
                        elif args.split_type == 'sentence':
                            chunks = sent_tokenize(context['text'])
                        
                        # print(chunks)
                        for chunk in chunks:
                            corpus[f"{doc_id}_{chunk_id}"] = {
                                "text": chunk,
                                "title": f"{doc_id}_{chunk_id}_"
                            }
                            chunk_id += 1
                    
                    # queries = {query_id: query}
                    queries = {query_id: f"{query} The relation is {RELATIONS[relation_id]}"} # Add the relation to the query
                    retrieve_results = retriever.retrieve(corpus, queries)
                    retrieve_results_sorted = dict(sorted(retrieve_results[query_id].items(), key=lambda item: item[1], reverse=True))
                    # print(retrieve_results_sorted)
                    
                    # === Step 2: Save the retrieved sentences to a file ===
                    out_sentences = []
                    for sent_id, score in retrieve_results_sorted.items():
                        out_sentences.append({
                            'ref_doc_id': sent_id.rsplit('_', 1)[0],
                            "sentence": corpus[sent_id]['text'],
                            "score": score
                        })

                    item = {
                        "id": query_id,
                        "question": query,
                        "sentences": out_sentences
                    }
                    f_out.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_model", required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split_type", type=str, required=True, choices=['sentence', 'word'])
    parser.add_argument("--word_num", type=int, default=10)
    parser.add_argument("--retrieval_method", type=str, required=True)
    parser.add_argument("--num_retrieved_passages", type=int, default=1)
    parser.add_argument("--output_path", type=str, default=None, required=True)
    args = parser.parse_args()
    main(args)