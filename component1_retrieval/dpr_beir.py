from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random
import json


def data_preprocessing():
    
    if not os.path.exists("component1_retrieval/popqa_data"):
        os.makedirs("component1_retrieval/popqa_data")
    
    if not os.path.exists("component1_retrieval/popqa_data/qrels"):
        os.makedirs("component1_retrieval/popqa_data/qrels")
    
    input_corpus = "data/generated/popQA_costomized/corpus.jsonl"
    output_corpus = "component1_retrieval/popqa_data/corpus.jsonl"
    
    input_queries = "data/generated/popQA_costomized/queries.jsonl"
    output_queries = "component1_retrieval/popqa_data/queries.jsonl"
    
    input_qrels = "data/generated/popQA_costomized/qrels.jsonl"
    output_qrels = "component1_retrieval/popqa_data/qrels/test.tsv"
    
    with open(input_corpus, 'r') as in_corpus, open(output_corpus, 'w') as out_corpus:
        for idx, line in enumerate(in_corpus):
            data = json.loads(line.strip()) 

            new_json = {
                "_id": data["id"],
                "title": "",
                "text": data["contents"]
            }
       
            corpus_jsonl_line = json.dumps(new_json)
            out_corpus.write(corpus_jsonl_line + '\n')
    
    with open(input_queries, 'r') as in_queries, open(output_queries, 'w') as out_queries:
        for idx, line in enumerate(in_queries):
            data = json.loads(line.strip()) 

            new_json = {
                "_id": data["entity_id"],
                "text": data["question"]
            }
       
            query_jsonl_line = json.dumps(new_json)
            out_queries.write(query_jsonl_line + '\n')

    with open(input_qrels, 'r') as in_qrels, open(output_qrels, 'w') as out_qrels:
        out_qrels.write("query_id\tcorpus_id\tscore\n")   
        for line in in_qrels:
            data = json.loads(line)
            tsv_line = '{}\t{}\t{}\n'.format(data.get("query_id", ""), data.get("doc_id", ""), data.get("score", 0))
            out_qrels.write(tsv_line)


if __name__ == "__main__":
    
    # corpus -> jsonl {"_id", "title", "text"}
    # queries -> jsonl {"_id", "text"}
    # qrels -> tsv {"quesry_id", "corpus_id", "score"}
    preprocessed_data = data_preprocessing()
    
    data_path = "component1_retrieval/popqa_data"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    model = DRES(models.SentenceBERT((
        "facebook-dpr-question_encoder-multiset-base",
        "facebook-dpr-ctx_encoder-multiset-base",
        " [SEP] "), batch_size=128))
    
    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    
    # logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    print("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    print(ndcg)
    print(_map)
    print(recall)
    print(precision)
    
  
    # Print an example  
    top_k = 3
    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    print("Query : %s\n" % queries[query_id])

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        print("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
    