#!/usr/bin/env python3

import os, sys
import torch
import argparse
import logging

from beir import LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

print(os.getcwd())
import sys
sys.path.append(os.getcwd())

from component1_retrieval.customized_datasets.data_loader import CostomizedGenericDataLoader
from component1_retrieval.utils import save_qrels_file, save_evaluation_files, save_evaluation_files_v2

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
    
    model = DRES(models.SentenceBERT(args.model, batch_size=128), device=device)
    # model = DRES(models.SentenceBERT((
    #     "facebook-dpr-question_encoder-multiset-base",
    #     "facebook-dpr-ctx_encoder-multiset-base",
    #     " [SEP] "), batch_size=128), device=device)
    retriever = EvaluateRetrieval(model, score_function="dot")
    
    dataloader = CostomizedGenericDataLoader(data_folder = args.data_path)
    corpus, queries = dataloader.load_corpus_queries()
    # print(corpus)
    # print(queries)
    results = retriever.retrieve(corpus, queries)
    
    if not os.path.exists(args.output_results_dir):
        os.makedirs(args.output_results_dir)
    
    save_qrels_file(results, args)
    # save_evaluation_files(retriever, results, args)
    save_evaluation_files_v2(retriever, results, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_results_dir", type=str)
    parser.add_argument("--output_results_filename", type=str)
    parser.add_argument("--results_save_file", default=None, type=str)
    args = parser.parse_args()
    main(args)
    
    