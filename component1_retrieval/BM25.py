#!/usr/bin/env python3

# pip install pyserini -q
# pip install faiss-cpu -q
# pip install beir -q

import json, os
import logging
import argparse
import torch

from beir import LoggingHandler
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval

print(os.getcwd())
import sys
sys.path.append(os.getcwd())

from component1_retrieval.customized_datasets.data_loader import CostomizedGenericDataLoader
from component1_retrieval.utils import save_qrels_file, save_evaluation_files_v1, save_evaluation_files_v2

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
    
    dataloader = CostomizedGenericDataLoader(data_folder = args.data_path)
    corpus, queries = dataloader.load_corpus_queries()
    
    
    hostname = "localhost" 
    index_name = f"{args.dataset_name.lower()}_index" 
    initialize = True 
    
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)
    
    if not os.path.exists(args.output_results_dir):
        os.makedirs(args.output_results_dir)
    
    resutls_for_rerank_path = f"{args.output_results_dir}/{args.dataset_name.lower()}_{args.bm25_results_for_rerank_file}"
    with open(resutls_for_rerank_path, 'w') as f:
        json.dump(results, f, indent=4)

    save_qrels_file(results, args)
    # save_evaluation_files_v1(retriever, results, args)
    save_evaluation_files_v2(retriever, results, args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_results_dir", type=str)
    parser.add_argument("--output_results_filename", type=str)
    parser.add_argument("--results_save_file", default=None, type=str)
    parser.add_argument("--bm25_results_for_rerank_file", default=None, type=str)
    
    args = parser.parse_args()
    main(args)
    
    