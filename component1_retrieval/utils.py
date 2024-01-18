import os, csv, json
import logging

def preprocessing_qrels(qid_list):
    # input_qrels = "component0_preprocessing/generated_data/popQA_costomized/qrels.jsonl"
    input_qrels = "component0_preprocessing/generated_data/popQA_religion/qrels_30ds_512tk.jsonl"
    
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

def save_qrels_file(results, args):
    if args.results_save_file:
        qrels_resutls_path = os.path.join(args.output_results_dir, args.results_save_file)
        k = 1
        with open(qrels_resutls_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['query_id', 'corpus_id', 'score'], delimiter='\t')
            writer.writeheader()
            for query_id, value in results.items():
                sorted_doc_scores = dict(sorted(value.items(), key=lambda item: item[1], reverse=True))
                writer.writerow({
                    'query_id': query_id,
                    'corpus_id': list(sorted_doc_scores.keys())[0],
                    'score': list(sorted_doc_scores.values())[0]
                })

def save_evaluation_files(retriever, results, args):
    # queries_bk_path = "component0_preprocessing/generated_data/popQA_costomized/queries_bucketing.json"
    # with open(queries_bk_path, 'r') as in_queries:
    #     query_data = json.load(in_queries)
    
    queries_bk_path = "component0_preprocessing/generated_data/popQA_religion/queries_30ds_bk.json"
    with open(queries_bk_path, 'r') as in_queries:
        query_data = {"religion": json.load(in_queries)}
    
    if not os.path.exists(args.output_results_dir):
        os.makedirs(args.output_results_dir)
    resutls_wbk_path = os.path.join(args.output_results_dir, 'wbk_'+args.output_results_filename)
    resutls_wobk_path = os.path.join(args.output_results_dir, 'wobk_'+args.output_results_filename)
    
    
     
    # Without bucketting
    with open(resutls_wobk_path, 'w', newline='') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow([
            "Title",
            "NDCG@1", "NDCG@3", "NDCG@5",
            "MAP@1", "MAP@3", "MAP@5",
            "Recall@1", "Recall@3", "Recall@5",
            "P@1", "P@3", "P@5",
        ])
        
        for relation_name, relation_data in query_data.items():    
            rel_data = []
            for bk_name, bk_data in relation_data.items():
                rel_data.extend(bk_data)
            
            qid_list = [q_sample["query_id"] for q_sample in rel_data]
            qrels = preprocessing_qrels(qid_list)
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 3, 5]) #retriever.k_values
            logging.info(ndcg)
            logging.info(_map)
            logging.info(recall)
            logging.info(precision)
            print(ndcg)
            print(_map)
            print(recall)
            print(precision)
            print("\n")
        
            eval_res = [relation_name]\
                +list(ndcg.values())\
                +list(_map.values())\
                +list(recall.values())\
                +list(precision.values())
            tsv_writer.writerow(eval_res)
        
    # With bucketing
    with open(resutls_wbk_path, 'w', newline='') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow([
            "Title",
            "NDCG@1", "NDCG@3", "NDCG@5",
            "MAP@1", "MAP@3", "MAP@5",
            "Recall@1", "Recall@3", "Recall@5",
            "P@1", "P@3", "P@5",
        ])
        
        for relation_name, relation_data in query_data.items():
            for bk_name, bk_data in relation_data.items():
                # print('{}, {}'.format(relation_name, bk_name))
                logging.info('{}, {}'.format(relation_name, bk_name))
                
                if len(bk_data) == 0:
                    eval_res = [relation_name+'_'+bk_name] + [0]*16

                else:
                    qid_list = [q_sample["query_id"] for q_sample in bk_data]
                    qrels = preprocessing_qrels(qid_list)
                    
                    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 3, 5]) #retriever.k_values

                    logging.info(ndcg)
                    logging.info(_map)
                    logging.info(recall)
                    logging.info(precision)
                    # print(ndcg)
                    # print(_map)
                    # print(recall)
                    # print(precision)
                    print("\n")
        
                    eval_res = [relation_name+'_'+bk_name]\
                        +list(ndcg.values())\
                        +list(_map.values())\
                        +list(recall.values())\
                        +list(precision.values())
                tsv_writer.writerow(eval_res)