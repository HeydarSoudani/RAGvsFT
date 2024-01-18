import csv, os, json

def preprocessing_corpus_queries_qrels(relation_name):
    
    # input_corpus = "component0_preprocessing/generated_data/popQA_costomized/corpus.jsonl"
    # input_queries = "component0_preprocessing/generated_data/popQA_costomized/queries.jsonl"
    # input_qrels = "component0_preprocessing/generated_data/popQA_costomized/qrels.jsonl"
    
    # For religion
    input_queries = "component0_preprocessing/generated_data/popQA_religion/queries_30ds.jsonl"
    input_corpus = "component0_preprocessing/generated_data/popQA_religion/corpus_30ds_512tk.jsonl"
    input_qrels = "component0_preprocessing/generated_data/popQA_religion/qrels_30ds_512tk.jsonl"
    input_queries = "component0_preprocessing/generated_data/popQA_religion/gen-queries.jsonl"
    input_qrels = "component0_preprocessing/generated_data/popQA_religion/gen_qrels.jsonl"
    
    output_corpus = "component1_retrieval/data/popqa_{}/corpus.jsonl".format(relation_name)
    output_queries = "component1_retrieval/data/popqa_{}/queries.jsonl".format(relation_name)
    output_qrels = "component1_retrieval/data/popqa_{}/qrels/test.tsv".format(relation_name)
    
    output_queries = "component1_retrieval/data/popqa_{}/gen-queries.jsonl".format(relation_name)
    output_qrels = "component1_retrieval/data/popqa_{}/gen-qrels/train.tsv".format(relation_name)
    
    
    directory = os.path.dirname(output_qrels)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # with open(input_corpus, 'r') as in_corpus, open(output_corpus, 'w') as out_corpus:
    #     for idx, line in enumerate(in_corpus):
    #         data = json.loads(line.strip()) 

    #         new_json = {
    #             "_id": data["id"],
    #             "title": "",
    #             "text": data["contents"]
    #         }
       
    #         corpus_jsonl_line = json.dumps(new_json)
    #         out_corpus.write(corpus_jsonl_line + '\n')
    
    with open(input_queries, 'r') as in_queries, open(output_queries, 'w') as out_queries:
        for idx, line in enumerate(in_queries):
            data = json.loads(line.strip()) 

            new_json = {
                "_id": data["query_id"],
                "text": data["question"]
            }
       
            query_jsonl_line = json.dumps(new_json)
            out_queries.write(query_jsonl_line + '\n')

    with open(input_qrels, 'r') as in_qrels, open(output_qrels, 'w') as out_qrels:
        out_qrels.write("query_id\tcorpus_id\tscore\n")   
        for line in in_qrels:
            data = json.loads(line)
            if data.get("score") == 1:
                tsv_line = '{}\t{}\t{}\n'.format(data.get("query_id", ""), data.get("doc_id", ""), data.get("score", 0))
                out_qrels.write(tsv_line)

def main():
    # corpus -> jsonl {"_id", "title", "text"}
    # queries -> jsonl {"_id", "text"}
    # qrels -> tsv {"quesry_id", "corpus_id", "score"}
    relation_name = "religion"
    os.makedirs("component1_retrieval/data", exist_ok=True)
    os.makedirs("component1_retrieval/data/popqa_{}".format(relation_name), exist_ok=True)
    
    preprocessing_corpus_queries_qrels(relation_name)
    
if __name__ == "__main__":
    main()