#!/usr/bin/env python3

import torch
import argparse
import logging
from beir import LoggingHandler

from src.contriever_m import load_retriever
from src.utils_eval import evaluate_model
from src.utils_dist import is_main
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
    
    # === Model =====================
    model, tokenizer, _ = load_retriever(args.model)
    model = model.to(device)
    model.eval()
    query_encoder = model
    doc_encoder = model

    dataset = 'popqa'
    per_gpu_batch_size = 128
    norm_query = True
    norm_doc = True
    score_function = 'dot'
    save_results_path = args.output_results_dir
    lower_case = True
    normalize_text = True
    
    retriever, results = evaluate_model(
        query_encoder=query_encoder,
        doc_encoder=doc_encoder,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=per_gpu_batch_size,
        norm_query=norm_query,
        norm_doc=norm_doc,
        is_main=is_main(),
        split="dev" if dataset == "msmarco" else "test",
        score_function=score_function,
        data_path=args.data_path,
        save_results_path=save_results_path,
        lower_case=lower_case,
        normalize_text=normalize_text,
        device=device
    )
    
    save_qrels_file(results, args)
    save_evaluation_files(retriever, results, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_results_dir", type=str)
    parser.add_argument("--output_results_filename", type=str)
    parser.add_argument("--results_save_file", default=None, type=str)
    args = parser.parse_args()
    main(args)
