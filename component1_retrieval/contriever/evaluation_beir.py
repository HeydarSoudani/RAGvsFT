import torch
import os, json

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("Running on the GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
    print("Running on the mps")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def preprocessing_corpus_queries():
    
    if not os.path.exists("component1_retrieval/popqa_data"):
        os.makedirs("component1_retrieval/popqa_data")

    input_corpus = "data/generated/popQA_costomized/corpus.jsonl"
    output_corpus = "component1_retrieval/popqa_data/corpus.jsonl"
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
