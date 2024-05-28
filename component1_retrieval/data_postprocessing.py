import os, json, csv
import argparse
import ast


def main(args):
    dataset_name = "popQA" # popQA, witQA, EQ
    retrieval_method = 'contriever' # ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']
    number_of_passages = 3
    dataset_dir = f"component1_retrieval/data/{dataset_name}"
    
    if number_of_passages > 1:
        output_dir = f"component0_preprocessing/generated_data/{dataset_name}_costomized/retrieved_passage/{retrieval_method}_{number_of_passages}"
    else:
        output_dir = f"component0_preprocessing/generated_data/{dataset_name}_costomized/retrieved_passage/{retrieval_method}"
    os.makedirs(output_dir, exist_ok=True)
    
    query_dir = f"{dataset_dir}/test"
    corpus_file = f"{dataset_dir}/corpus.jsonl"
    gt_qrels_dir = f"{dataset_dir}/qrels"
    
    if retrieval_method == 'ideal':
        ret_qrels_file = f"component1_retrieval/data/{dataset_name}/qrels/test.tsv"
    else:
        ret_qrels_file = f"component1_retrieval/results/top5/{dataset_name}/{retrieval_method}-qrels.tsv"
    
    def get_corpus():
        corpus = {}
        with open(corpus_file, 'r') as cf:
            for line in cf:
                obj = json.loads(line)
                corpus[obj['_id']] = obj
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
    
    if retrieval_method == 'ideal':
        for idx, filename in enumerate(os.listdir(query_dir)):

            if filename.endswith('.json'):
                relation_id = filename.split('.')[0]
                print(f"Processing {relation_id}...")
                queries = get_queries(relation_id)
                gt_qrels = get_gt_qrels(relation_id)
                
                output_file = f"{output_dir}/{relation_id}.{retrieval_method}.ret_results.jsonl"
                with open(output_file, 'w') as ofile:
                    
                    for item in queries:
                        contexts = []
                        query_id = item["query_id"]
                        question = item["question"]
                        doc_id = list(gt_qrels[query_id].keys())[0]
                        # context = corpus.get(doc_id, "No context found")
                        first_context = corpus[doc_id]['text']
                        contexts.append({
                                "id": doc_id,
                                "text": first_context,
                                "hasanswer": True
                            })
                        
                        if number_of_passages > 1:
                            for i in range(1, number_of_passages):
                                title_prefix = corpus[doc_id]['title'].split('_')[0]
                                related_items = [
                                    item for item in corpus.values()
                                    if item["title"].startswith(title_prefix) and not item["title"].endswith("sum")
                                ][:number_of_passages-1]
                                
                                for related_item in related_items:
                                    doc_id = related_item['_id']
                                    context = related_item['text']
                                    contexts.append({
                                        "id": doc_id,
                                        "text": context,
                                        "hasanswer": False
                                    })
                        
                        # ======================            
                        combined_obj = {
                            "id": query_id,
                            "question": question,
                            "ctxs": contexts,
                        }
                        # combined_obj = {
                        #     "id": query_id,
                        #     "question": question,
                        #     "ctxs": [{
                        #         "id": doc_id,
                        #         "text": context,
                        #         "hasanswer": True
                        #     }],
                        # }
                        ofile.write(json.dumps(combined_obj) + "\n")
    
    else:
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
                                # TODO: ret_doc_id is a list
                                ret_query_id, ret_doc_ids = row[0], row[1]
                                if query_id == ret_query_id:
                                    doc_ids = ast.literal_eval(ret_doc_ids)
                                    contexts = []
                                    # print(doc_ids[0])
                                    for ret_doc_id in doc_ids[:number_of_passages]:
                                        context = corpus[ret_doc_id]['text']
                                        
                                        # === For hasanswer ====
                                        hasanswer = False
                                        if query_id in gt_qrels:
                                            if ret_doc_id in gt_qrels[query_id]:
                                                if gt_qrels[query_id][ret_doc_id] == 1:
                                                    hasanswer = True
                                        # ======================
                                        contexts.append({
                                            "id": ret_doc_id,
                                            "text": context,
                                            "hasanswer": hasanswer
                                        })
                                    
                                    combined_obj = {
                                        "id": ret_query_id,
                                        "question": question,
                                        "ctxs": contexts,
                                    }
                                    ofile.write(json.dumps(combined_obj) + "\n")
                                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
