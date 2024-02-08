import os, json, csv
import argparse


def main(args):
    retrieval_method = 'bm25' # ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']
    dataset_dir = "component1_retrieval/data/popqa"
    output_dir = f"component0_preprocessing/generated_data/popQA_EQformat/retrieved/{retrieval_method}"
    os.makedirs(output_dir, exist_ok=True)
    
    query_dir = f"{dataset_dir}/test"
    corpus_file = f"{dataset_dir}/corpus.jsonl"
    gt_qrels_dir = f"{dataset_dir}/qrels"
    ret_qrels_file = f"component1_retrieval/results/{retrieval_method}-qrels.tsv"
    
    
    def get_corpus():
        corpus = {}
        with open(corpus_file, 'r') as cf:
            for line in cf:
                obj = json.loads(line)
                corpus[obj['_id']] = obj['text']
        return corpus
    
    def get_queries(relation_id):
        with open(f"{query_dir}/{relation_id}.test.json", 'r') as qf:
            q_data = json.load(qf)
        return q_data
    
    def get_gt_qrels(relation_id):
        qrels = {}
        with open(f"{gt_qrels_dir}/{relation_id}.qrels.json", 'r') as qf:
            q_data = json.load(qf)
            
        for item in q_data:
            query_id, corpus_id, score = item["query_id"], item["doc_id"], int(item["score"])
            if query_id not in qrels:
                qrels[query_id] = {corpus_id: score}
            else:
                qrels[query_id][corpus_id] = score
        return qrels
    
    corpus = get_corpus()
    
    with open(ret_qrels_file, 'r') as ret_q_file:
        tsv_reader = csv.reader(ret_q_file, delimiter='\t')
        
        for idx, filename in enumerate(os.listdir(query_dir)):
            if filename.endswith('.json'):
                relation_id = filename.split('.')[0]
                queries = get_queries(relation_id)
                gt_qrels = get_gt_qrels(relation_id)
                
                output_file = f"{output_dir}/{relation_id}.{retrieval_method}.ret_results.jsonl"
                with open(output_file, 'w') as ofile:
                    
                    for item in queries:
                        query_id = item["query_id"]
                        question = item["question"]
                        
                        for row in tsv_reader:
                            ret_query_id, ret_doc_id = row[0], row[1]
                            if query_id == ret_query_id:
                                context = corpus.get(ret_doc_id, "No context found")
                                
                                # === For hasanswer ====
                                hasanswer = False
                                if query_id in gt_qrels:
                                    if ret_doc_id in gt_qrels[query_id]:
                                        if gt_qrels[query_id][ret_doc_id] == 1:
                                            hasanswer = True
                                # ======================
                                
                                combined_obj = {
                                    "id": ret_query_id,
                                    "question": question,
                                    "ctxs": [{
                                        "id": ret_doc_id,
                                        "text": context,
                                        "hasanswer": hasanswer
                                    }],
                                }
                                ofile.write(json.dumps(combined_obj) + "\n")
                                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
