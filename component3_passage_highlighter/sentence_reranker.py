#!/usr/bin/env python3

import os
import sys
import nltk
import json
import torch
import argparse
from itertools import chain

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


print(os.getcwd())
import sys
sys.path.append(os.getcwd())

from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES



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
    
    dataset_name = "popQA" # popQA, witQA, EQ
    retrieval_method = 'dpr' # ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']

    base_dir = f"component0_preprocessing/generated_data/{dataset_name}_costomized"
    retrieved_passage_dir = f"{base_dir}/retrieved_passage/{retrieval_method}_3"
    reranked_sentences_dir = f"{base_dir}/reranked_sentences/{retrieval_method}_3"
    os.makedirs(f'{base_dir}/reranked_sentences', exist_ok=True)
    os.makedirs(reranked_sentences_dir, exist_ok=True)
    
    #### Reranking top-100 docs using Dense Retriever model 
    dense_retriever_model = 'msmarco-distilbert-base-v2'
    model = DRES(models.SentenceBERT(dense_retriever_model), batch_size=1, device=device)
    retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[1,3, 5])
    
    
    for idx, filename in enumerate(os.listdir(retrieved_passage_dir)):
        # if idx == 1:
        #     break
            
        if filename.endswith('.jsonl'):
            relation_id = filename.split('.')[0]
            print(f"Processing {relation_id}...")
            
            output_file_path = os.path.join(reranked_sentences_dir, f'{relation_id}.{retrieval_method}.set_reranked.jsonl')
            with open(f"{retrieved_passage_dir}/{filename}", 'r') as f_in, open(output_file_path, 'w') as f_out:
                for line in f_in:
                    data = json.loads(line.strip())
                    query_id = data['id']
                    query = data['question']
                    
                    # === Step 1: Convert retrieved passages to a format that can be used by the dense retriever ===
                    corpus = {}
                    sentence_id = 0
                    for context in data['ctxs']:
                        doc_id = context['id']
                        sentences = sent_tokenize(context['text'])
                        for sentence in sentences:
                            corpus[f"{doc_id}_{sentence_id}"] = {
                                "text": sentence,
                                "title": f"{doc_id}_{sentence_id}_"
                            }
                            sentence_id += 1
                    
                    queries = {query_id: query}
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
    args = parser.parse_args()
    main(args)