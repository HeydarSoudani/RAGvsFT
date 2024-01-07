#!/usr/bin/env python3

# pip install pyserini -q
# pip install faiss-cpu -q
# pip install beir -q

from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from pyserini.search.lucene import LuceneSearcher
import json, os, csv
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def load_queries(file_path):
    queries = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            queries[data['entity_id']] = data['question']
    return queries

def preprocessing_qrels(qrels_file, qid_list):
    with open(qrels_file, 'r') as in_qrels:
        qrels = {}
        for line in in_qrels:
            data = json.loads(line)
            
            if data["query_id"] in qid_list:
                query_id, corpus_id, score = data["query_id"], data["doc_id"], int(data["score"])
                if query_id not in qrels:
                    qrels[query_id] = {corpus_id: score}
                else:
                    qrels[query_id][corpus_id] = score
        return qrels

def retrieve(corpus_pyserini_index_path, queries, k, results_save_file=None):
    searcher = LuceneSearcher(corpus_pyserini_index_path)
    results = {}
    
    for query_id, query_text in queries.items():
        hits = searcher.search(query_text, k)
        
        for hit in hits:
            if query_id not in results:
                results[query_id] = {hit.docid: hit.score}
            else:
                results[query_id][hit.docid] = hit.score

    if results_save_file:
        with open(results_save_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['query_id', 'corpus_id', 'score'], delimiter='\t')
            writer.writeheader()
            for query_id, value in results.items():
                writer.writerow({
                    'query_id': query_id,
                    'corpus_id': list(value.keys())[0],
                    'score': list(value.values())[0]
                })
    
    return results

def main():
    # Files for local
    # corpus_file = 'data/generated/popQA_costomized/corpus.jsonl'
    # queries_file = 'data/generated/popQA_costomized/queries.jsonl'
    # qrels_file = 'data/generated/popQA_costomized/qrels.jsonl'
    # queries_bk_path = "data/generated/popQA_costomized/queries_bucketing.json"
    # output_results_dir = "component1_retrieval/results"
    # output_results_filename = "bm25_eval.tsv"

    # Files for Google drive
    corpus_file = '/content/drive/MyDrive/RAGvsFT/data_bm25/corpus.jsonl'
    queries_file = '/content/drive/MyDrive/RAGvsFT/data_bm25/queries.jsonl'
    qrels_file = '/content/drive/MyDrive/RAGvsFT/data_bm25/qrels.jsonl'
    queries_bk_path = "/content/drive/MyDrive/RAGvsFT/data_bm25/queries_bucketing.json"
    corpus_pyserini_index_path = '/content/drive/MyDrive/RAGvsFT/data_bm25/index'
    
    output_results_dir = "/content/drive/MyDrive/RAGvsFT/data_bm25"
    output_results_filename = "bm25_eval.tsv"
    results_save_file = 'bm25-qrels.tsv'
    
    ### Indexing
    # python -m pyserini.index.lucene -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 9 \
    # -input /content/drive/MyDrive/RAGvsFT/data_bm25/corpus -index /content/drive/MyDrive/RAGvsFT/data_bm25/index -storePositions -storeDocvectors -storeRaw

    top_k = 100
    queries = load_queries(queries_file)
    
    # Without bucketting
    
    
    # With bucketting
    if not os.path.exists(output_results_dir):
        os.makedirs(output_results_dir)
    resutls_path = os.path.join(output_results_dir, output_results_filename)
    qrels_resutls_path = os.path.join(output_results_dir, results_save_file)
    
    ret_results = retrieve(corpus_pyserini_index_path, queries, top_k, qrels_resutls_path)
    ret_evaluator = EvaluateRetrieval()
        
    with open(queries_bk_path, 'r') as in_queries:
        query_data = json.load(in_queries)
    
    with open(resutls_path, 'w', newline='') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow([
            "Title",
            "NDCG@1", "NDCG@5", "NDCG@10", "NDCG@100",
            "MAP@1", "MAP@5", "MAP@10", "MAP@100",
            "Recall@1", "Recall@5", "Recall@10", "Recall@100",
            "P@1", "P@5", "P@10", "P@100",
        ])
        
        for relation_name, relation_data in query_data.items():
            for bk_name, bk_data in relation_data.items():
                print('{}, {}'.format(relation_name, bk_name))
                logging.info('{}, {}'.format(relation_name, bk_name))
                
                if len(bk_data) == 0:
                    eval_res = [relation_name+'_'+bk_name] + [0]*16

                else:
                    qid_list = [q_sample["entity_id"] for q_sample in bk_data]
                    qrels = preprocessing_qrels(qrels_file, qid_list)
                    
                    ndcg, _map, recall, precision = ret_evaluator.evaluate(qrels, ret_results, [1, 5, 10, 100]) #retriever.k_values

                    # logging.info(ndcg)
                    # logging.info(_map)
                    # logging.info(recall)
                    # logging.info(precision)
                    print(ndcg)
                    print(_map)
                    print(recall)
                    print(precision)
                    print("\n")
        
                    eval_res = [relation_name+'_'+bk_name]\
                        +list(ndcg.values())\
                        +list(_map.values())\
                        +list(recall.values())\
                        +list(precision.values())
                tsv_writer.writerow(eval_res)

if __name__ == "__main__":
    main()
    
    