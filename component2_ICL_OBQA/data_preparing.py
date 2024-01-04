import os, json, csv

def get_corpus():
    corpus = {}
    cor_file = 'component0_preprocessing/generated_data/popQA_costomized/corpus.jsonl'
    with open(cor_file, 'r') as corpus_file:
        for line in corpus_file:
            obj = json.loads(line)
            corpus[obj['id']] = obj['contents']
    return corpus

def get_queries():
    queries = {}
    qur_file = 'component0_preprocessing/generated_data/popQA_costomized/queries.jsonl'
    with open(qur_file, 'r') as queries_file:
        for line in queries_file:
            obj = json.loads(line)
            queries[obj['entity_id']] = obj['question']
    return queries

def retrieval_resutls_ideal():
    corpus = get_corpus()
    queries = get_queries()

    qr_file = 'component0_preprocessing/generated_data/popQA_costomized/qrels.jsonl'
    out_file = 'component2_ICL_OBQA/data/popqa/gt_result.jsonl'
    with open(qr_file, 'r') as qrels_file, open(out_file, 'w') as output_file:
        for idx, line in enumerate(qrels_file):
            # if idx == 10:
            #     break
            
            obj = json.loads(line)
            
            doc_id = obj['doc_id']
            query_id = obj['query_id']
            context = corpus.get(doc_id, "No context found")
            question = queries.get(query_id, "No question found")
            
            combined_obj = {
                "id": query_id,
                "question": question,
                "ctxs": [{
                    "id": doc_id,
                    "text": context,
                    "hasanswer": True if obj['score'] == 1 else False
                }],
            }
            
            output_file.write(json.dumps(combined_obj) + "\n")

def retrieval_resutls_no_ideal():
    corpus = get_corpus()
    queries = get_queries()
    
    qr_file = 'component0_preprocessing/generated_data/popQA_costomized/qrels.jsonl'
    
    # For bm25
    # retrieved_qr_file = 'component1_retrieval/results/bm25-qrels.tsv'
    # out_file = 'component2_ICL_OBQA/data/popqa/bm25_results.jsonl'
    # For DPR, no FT
    # retrieved_qr_file = 'component1_retrieval/results/dpr_noft-qrels.tsv'
    # out_file = 'component2_ICL_OBQA/data/popqa/dpr_noft_results.jsonl'
    # For DPR, FT
    retrieved_qr_file = 'component1_retrieval/results/dpr_ft-qrels.tsv'
    out_file = 'component2_ICL_OBQA/data/popqa/dpr_ft_results.jsonl'
    
    with open(retrieved_qr_file, 'r') as ret_qrels_file, open(qr_file, 'r') as gt_qrels_file, open(out_file, 'w') as output_file:
        
        ret_qrels_reader = csv.DictReader(ret_qrels_file, delimiter='\t')
        
        for idx, ret_qrels_row in enumerate(ret_qrels_reader):
            
            ret_doc_id = ret_qrels_row["corpus_id"]
            ret_query_id = ret_qrels_row["query_id"]
            hasanswer = False
            
            for idx, line in enumerate(gt_qrels_file):
                gt_qrels_row = json.loads(line)

                if ret_query_id == gt_qrels_row.get("query_id", None):
                    if ret_doc_id == gt_qrels_row.get("doc_id", None):
                        if gt_qrels_row["score"] == 1:
                            hasanswer = True
                    break
            
            context = corpus.get(ret_doc_id, "No context found")
            question = queries.get(ret_query_id, "No question found")
            combined_obj = {
                "id": ret_query_id,
                "question": question,
                "ctxs": [{
                    "id": ret_doc_id,
                    "text": context,
                    "hasanswer": hasanswer
                }],
            }
            output_file.write(json.dumps(combined_obj) + "\n")

def main():
    os.makedirs("component2_ICL_OBQA/data", exist_ok=True)
    os.makedirs("component2_ICL_OBQA/data/popqa", exist_ok=True)
    
    # retrieval_resutls_ideal()
    retrieval_resutls_no_ideal()
    



if __name__ == "__main__":
    main()