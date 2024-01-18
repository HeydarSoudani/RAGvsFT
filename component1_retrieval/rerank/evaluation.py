#!/usr/bin/env python3

import os, sys
import torch
import argparse
import json
import logging

from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models

print(os.getcwd())
import sys
sys.path.append(os.getcwd())

from component1_retrieval.customized_datasets.data_loader import CostomizedGenericDataLoader
from component1_retrieval.utils import save_qrels_file, save_evaluation_files


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def main(args):
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Running on the GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
        print("Running on the mps")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    
    dataloader = CostomizedGenericDataLoader(data_folder=args.data_path)
    corpus, queries = dataloader.load_corpus_queries()
    
    # === First stage with Elastic-search ===========
    # hostname = "http://localhost:9200"
    # index_name = "popqa"
    # initialize = True
    # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    # retriever = EvaluateRetrieval(model)
    # results = retriever.retrieve(corpus, queries)
    
    # === First stage from Google Colab =============
    # bm25_results_path = 'component1_retrieval/rerank/data/bm25_results_rerank.json'
    bm25_results_path = 'component1_retrieval/results/religion/bm25_results_rerank.json'
    with open(bm25_results_path, 'r') as f:
        results = json.load(f)
    
    #### Reranking top-100 docs using Dense Retriever model 
    model = DRES(models.SentenceBERT(args.dense_model), batch_size=128, device=device)
    dense_retriever = EvaluateRetrieval(model, score_function="dot", k_values=[1,5,10,100])
    print('a')
    rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)
    print('b')
    
    save_qrels_file(rerank_results, args)
    save_evaluation_files(dense_retriever, rerank_results, args)
    # ndcg, _map, recall, precision, hole = dense_retriever.evaluate(qrels, rerank_results, retriever.k_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_model", required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_results_dir", type=str)
    parser.add_argument("--output_results_filename", type=str)
    parser.add_argument("--results_save_file", default=None, type=str)
    args = parser.parse_args()
    main(args)