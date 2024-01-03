import json

def main():
    
    corpus = {}
    cor_file = 'data/generated/popQA_costomized/corpus.jsonl'
    with open(cor_file, 'r') as corpus_file:
        for line in corpus_file:
            obj = json.loads(line)
            corpus[obj['id']] = obj['contents']
    
    queries = {}
    qur_file = 'data/generated/popQA_costomized/queries.jsonl'
    with open(qur_file, 'r') as queries_file:
        for line in queries_file:
            obj = json.loads(line)
            queries[obj['entity_id']] = obj['question']

    qr_file = 'data/generated/popQA_costomized/qrels.jsonl'
    out_file = 'data/generated/popQA_costomized/ret_result.jsonl'
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


if __name__ == "__main__":
    main()