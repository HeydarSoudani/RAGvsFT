#!/usr/bin/env python3

import torch
import argparse
import csv, os, json
import logging

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from customized_datasets.data_loader import CostomizedGenericDataLoader

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def preprocessing_qrels(qid_list):
    input_qrels = "component0_preprocessing/generated_data/popQA_costomized/qrels.jsonl"
    
    with open(input_qrels, 'r') as in_qrels: 
        qrels = {}
        for line in in_qrels:
            data = json.loads(line)
            query_id, corpus_id, score = data["query_id"], data["doc_id"], int(data["score"])
            
            if len(qid_list) == 0:
                if query_id not in qrels:
                    qrels[query_id] = {corpus_id: score}
                else:
                    qrels[query_id][corpus_id] = score
            else:
                if data["query_id"] in qid_list:
                    if query_id not in qrels:
                        qrels[query_id] = {corpus_id: score}
                    else:
                        qrels[query_id][corpus_id] = score
        return qrels

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
    
    # if not os.path.exists(args.output_results_dir):
    #     os.makedirs(args.output_results_dir)
    # resutls_path = os.path.join(args.output_results_dir, args.output_results_filename)
    
    model = DRES(models.SentenceBERT(args.model, batch_size=128), device=device)
    # model = DRES(models.SentenceBERT((
    #     "facebook-dpr-question_encoder-multiset-base",
    #     "facebook-dpr-ctx_encoder-multiset-base",
    #     " [SEP] "), batch_size=128), device=device)
    retriever = EvaluateRetrieval(model, score_function="dot")
    
    dataloader = CostomizedGenericDataLoader(data_folder=args.data_path)
    corpus, queries = dataloader.load_corpus_queries()
    
    results = retriever.retrieve(corpus, queries)
    
    # save qrels file
    # if args.results_save_file:
    #     qrels_resutls_path = os.path.join(args.output_results_dir, args.results_save_file)
    #     k = 1
    #     with open(qrels_resutls_path, 'w', newline='') as file:
    #         writer = csv.DictWriter(file, fieldnames=['query_id', 'corpus_id', 'score'], delimiter='\t')
    #         writer.writeheader()
    #         for query_id, value in results.items():
    #             sorted_doc_scores = dict(sorted(value.items(), key=lambda item: item[1], reverse=True))
    #             writer.writerow({
    #                 'query_id': query_id,
    #                 'corpus_id': list(sorted_doc_scores.keys())[0],
    #                 'score': list(sorted_doc_scores.values())[0]
    #             })
    
    # Without bucketting results
    qrels = preprocessing_qrels([])                    
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 5, 10, 100]) #retriever.k_values

    logging.info('without bucketting')
    logging.info(ndcg)
    logging.info(_map)
    logging.info(recall)
    logging.info(precision)
    print("\n")
        
    
    # # With bucketting results
    # with open(resutls_path, 'w', newline='') as file:
    #     tsv_writer = csv.writer(file, delimiter='\t')
    #     tsv_writer.writerow([
    #         "Title",
    #         "NDCG@1", "NDCG@5", "NDCG@10", "NDCG@100",
    #         "MAP@1", "MAP@5", "MAP@10", "MAP@100",
    #         "Recall@1", "Recall@5", "Recall@10", "Recall@100",
    #         "P@1", "P@5", "P@10", "P@100",
    #     ])
        
        
    #     queries_bk_path = "component0_preprocessing/generated_data/popQA_costomized/queries_bucketing.json"
    #     with open(queries_bk_path, 'r') as in_queries:
    #         query_data = json.load(in_queries)
        
    #     for relation_name, relation_data in query_data.items():
    #         for bk_name, bk_data in relation_data.items():
    #             # print('{}, {}'.format(relation_name, bk_name))
    #             logging.info('{}, {}'.format(relation_name, bk_name))
                
    #             if len(bk_data) == 0:
    #                 eval_res = [relation_name+'_'+bk_name] + [0]*16

    #             else:
    #                 qid_list = [q_sample["entity_id"] for q_sample in bk_data]
    #                 qrels = preprocessing_qrels(qid_list)
                    
    #                 ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 5, 10, 100]) #retriever.k_values

    #                 logging.info(ndcg)
    #                 logging.info(_map)
    #                 logging.info(recall)
    #                 logging.info(precision)
    #                 # print(ndcg)
    #                 # print(_map)
    #                 # print(recall)
    #                 # print(precision)
    #                 print("\n")
        
    #                 eval_res = [relation_name+'_'+bk_name]\
    #                     +list(ndcg.values())\
    #                     +list(_map.values())\
    #                     +list(recall.values())\
    #                     +list(precision.values())
    #             tsv_writer.writerow(eval_res)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_results_dir", type=str)
    parser.add_argument("--output_results_filename", type=str)
    parser.add_argument("--results_save_file", default=None, type=str)
    args = parser.parse_args()
    main(args)
    
    