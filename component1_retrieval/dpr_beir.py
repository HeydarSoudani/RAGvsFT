from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import torch
import csv, os, json

from customized_datasets.data_loader import CostomizedGenericDataLoader

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("Running on the GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
    print("Running on the mps")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def data_preprocessing():
    
    if not os.path.exists("component1_retrieval/popqa_data"):
        os.makedirs("component1_retrieval/popqa_data")

    input_corpus = "data/generated/popQA_costomized/corpus.jsonl"
    output_corpus = "component1_retrieval/popqa_data/corpus.jsonl"
        
    input_qrels = "data/generated/popQA_costomized/qrels.jsonl"
    output_qrels = "component1_retrieval/popqa_data/qrels/test.tsv"
    
    input_queries = "data/generated/popQA_costomized/queries.jsonl"
    output_queries = "component1_retrieval/popqa_data/queries.jsonl"
    
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
    
    results_dir = 'component1_retrieval/results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_filename = 'dpr_beir.tsv'
    resutls_path = os.path.join(results_dir, results_filename)
    
    # corpus -> jsonl {"_id", "title", "text"}
    # queries -> jsonl {"_id", "text"}
    # qrels -> tsv {"quesry_id", "corpus_id", "score"}
    preprocessed_data = data_preprocessing()
    corpus_path = "component1_retrieval/popqa_data/corpus.jsonl"
    
    if not os.path.exists("component1_retrieval/popqa_data/qrels"):
        os.makedirs("component1_retrieval/popqa_data/qrels")
    
    input_qrels = "data/generated/popQA_costomized/qrels.jsonl"
    output_qrels = "component1_retrieval/popqa_data/qrels/test.tsv"
    
    queries_bk_path = "data/generated/popQA_costomized/queries_bucketing.json"
    output_queries = "component1_retrieval/popqa_data/queries.jsonl"
    
    with open(queries_bk_path, 'r') as in_queries:
        query_data = json.load(in_queries)
    
    model = DRES(models.SentenceBERT((
        "facebook-dpr-question_encoder-multiset-base",
        "facebook-dpr-ctx_encoder-multiset-base",
        " [SEP] "), batch_size=128), device=device)
    retriever = EvaluateRetrieval(model, score_function="dot")
    
    data_path = "component1_retrieval/popqa_data"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    dataloader = CostomizedGenericDataLoader(device=device)
    corpus = dataloader.load_corpus(data_folder=data_path)
    
    
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
                if len(bk_data) == 0:
                    eval_res = [relation_name+'_'+bk_name] + [0]*16
                
                else:
                    qid_list = []
                    
                    with open(output_queries, 'w') as out_queries:
                        for q_sample in bk_data:
                            new_json = {
                                "_id": q_sample["entity_id"],
                                "text": q_sample["question"]
                            }
                            qid_list.append(q_sample["entity_id"])
                
                            query_jsonl_line = json.dumps(new_json)
                            out_queries.write(query_jsonl_line + '\n')

                    with open(input_qrels, 'r') as in_qrels, open(output_qrels, 'w') as out_qrels:
                        out_qrels.write("query_id\tcorpus_id\tscore\n")   
                        for line in in_qrels:
                            data = json.loads(line)
                            if data["query_id"] in qid_list:
                                tsv_line = '{}\t{}\t{}\n'.format(data.get("query_id", ""), data.get("doc_id", ""), data.get("score", 0))
                                out_qrels.write(tsv_line)
                                     
                    queries, qrels = dataloader.load_queries_qrels(
                        data_folder=data_path
                    )

                    results = retriever.retrieve(corpus, queries)
                    print(results)
                    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 5, 10, 100]) #retriever.k_values

                    
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
    
    # Print an example  
    # top_k = 3
    # query_id, ranking_scores = random.choice(list(results.items()))
    # scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    # print("Query : %s\n" % queries[query_id])

    # for rank in range(top_k):
    #     doc_id = scores_sorted[rank][0]
    #     print("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
    