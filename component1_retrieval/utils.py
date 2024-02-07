import os, csv, json
import logging
import math

def split_to_buckets(objects, split_points):
    
    split_points = sorted(split_points)
    sp_len = len(split_points)
    bucket_data = {'bucket{}'.format(idx+1): list() for idx in range(sp_len+1)}
    
    for obj in objects:
        # rp = obj['relative_popularity']
        if obj['pageviews'] != 0:
            rp = math.log(int(obj['pageviews']), 10)
        else:
            rp = 0
        
        if rp < split_points[0]:
            if 'bucket1' in bucket_data.keys():
                bucket_data['bucket1'].append(obj)
            else:
                bucket_data['bucket1'] = [obj]
        
        if rp >= split_points[-1]:
            if 'bucket{}'.format(sp_len+1) in bucket_data.keys():
                bucket_data['bucket{}'.format(sp_len+1)].append(obj)
            else:
                bucket_data['bucket{}'.format(sp_len+1)] = [obj]

        for i in range(sp_len-1):
            if split_points[i] <= rp < split_points[i + 1]:
                if 'bucket{}'.format(i+2) in bucket_data.keys():
                    bucket_data['bucket{}'.format(i+2)].append(obj)
                else:
                    bucket_data['bucket{}'.format(i+2)] = [obj]
    
    return bucket_data

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

def save_evaluation_files_v1(retriever, results, args):
    queries_bk_path = "component0_preprocessing/generated_data/popQA_costomized/queries_bucketing.json"
    with open(queries_bk_path, 'r') as in_queries:
        query_data = json.load(in_queries)
    
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

def save_evaluation_files_v2(retriever, results, args):
    
    split_points = [2, 3, 4, 5] # Good for my pageviews
    k_values = [1, 3, 5]
    queries_filename_path = f"{args.data_path}/test"
    qrels_filename_dir = f"{args.data_path}/qrels"
    
    args.output_results_dir
    args.output_results_filename
    
    resutls_per_rel_path = f"{args.output_results_dir}/per_rel_{args.output_results_filename}"
    resutls_per_bk_path = f"{args.output_results_dir}/per_bk_{args.output_results_filename}"
        
    # Initiaite writing in files 
    with open(resutls_per_rel_path, 'w', newline='') as rel_res_file, open(resutls_per_bk_path, 'w', newline='') as bk_res_file:
        rel_tsv_writer = csv.writer(rel_res_file, delimiter='\t')
        rel_tsv_writer.writerow([
            "Title",
            "NDCG@1", "NDCG@3", "NDCG@5",
            "MAP@1", "MAP@3", "MAP@5",
            "Recall@1", "Recall@3", "Recall@5",
            "P@1", "P@3", "P@5",
        ])
        
        bk_tsv_writer = csv.writer(bk_res_file, delimiter='\t')
        bk_tsv_writer.writerow([
            "Title",
            "NDCG@1", "NDCG@3", "NDCG@5",
            "MAP@1", "MAP@3", "MAP@5",
            "Recall@1", "Recall@3", "Recall@5",
            "P@1", "P@3", "P@5",
        ])
        all_qrels = {}
        all_queries = []
    
        for filename in os.listdir(qrels_filename_dir):
            if filename.endswith('.json'):
                relation_id = filename.split('.')[0]
                qrels_file_path = os.path.join(qrels_filename_dir, filename)
                
                logging.info(f"Processing relation {relation_id} ...")
                print(f"Processing relation {relation_id} ...")
                
                with open(qrels_file_path, 'r') as qrels_infile:
                    qrels_rel_data = json.load(qrels_infile)
                
                # Get the evaluation results for each relation
                qrels = {}
                for item in qrels_rel_data:
                    query_id, corpus_id, score = item["query_id"], item["doc_id"], int(item["score"])
                    if query_id not in qrels:
                        qrels[query_id] = {corpus_id: score}
                    else:
                        qrels[query_id][corpus_id] = score
                all_qrels.update(qrels)
                
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values) #retriever.k_values
                # mrr = retriever.evaluate_custom(qrels, results, k_values, metric="mrr")
                # recall_cap = retriever.evaluate_custom(qrels, results, k_values, metric="recall_cap")
                # hole = retriever.evaluate_custom(qrels, results, k_values, metric="hole")
                # top_k_accuracy = retriever.evaluate_custom(qrels, results, k_values, metric="top_k_accuracy")
                
                eval_res = [f"{relation_id}"]\
                        +list(ndcg.values())\
                        +list(_map.values())\
                        +list(recall.values())\
                        +list(precision.values())
                rel_tsv_writer.writerow(eval_res)
                
                
                # =============================
                # === bucketing ===============
                query_file = f"{queries_filename_path}/{relation_id}.test.json"
                with open(query_file, 'r') as in_queries:
                    query_rel_data = json.load(in_queries)
                all_queries.extend(query_rel_data)
                
                bk_data = split_to_buckets(query_rel_data, split_points)
                
                for bk_name, bk_value in bk_data.items():
                    logging.info(f"Processing {bk_name} ...")
                    print(f"Processing {bk_name} ...")
                    
                    if len(bk_value) == 0:
                        eval_res = [f"{relation_id}_{bk_name}"] + [0]*12
                    
                    else:
                        qid_list = [q_sample["query_id"] for q_sample in bk_value]
                        
                        # Create qrels for each bucket
                        qrels = {}
                        for item in qrels_rel_data:
                            query_id, corpus_id, score = item["query_id"], item["doc_id"], int(item["score"])
                            if item["query_id"] in qid_list:
                                if query_id not in qrels:
                                    qrels[query_id] = {corpus_id: score}
                                else:
                                    qrels[query_id][corpus_id] = score
                    
                                    
                        if len(qrels) > 0:
                            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values) #retriever.k_values

                            eval_res = [f"{relation_id}_{bk_name}"]\
                                +list(ndcg.values())\
                                +list(_map.values())\
                                +list(recall.values())\
                                +list(precision.values())
                        else:
                            eval_res = [f"{relation_id}_{bk_name}"] + [0]*12

                    bk_tsv_writer.writerow(eval_res)
                    
            
        # Get the evaluation results for all data
        logging.info(f"Processing all data ...")
        print(f"Processing all data ...")
        ndcg, _map, recall, precision = retriever.evaluate(all_qrels, results, k_values) #retriever.k_values
        
        eval_res = ["all"]\
                +list(ndcg.values())\
                +list(_map.values())\
                +list(recall.values())\
                +list(precision.values())
        rel_tsv_writer.writerow(eval_res)   
        
        
        bk_data = split_to_buckets(all_queries, split_points)
                
        for bk_name, bk_value in bk_data.items():
            logging.info(f"Processing {bk_name} ...")
            print(f"Processing {bk_name} ...")
            
            if len(bk_value) == 0:
                eval_res = [f"{relation_id}_{bk_name}"] + [0]*12
            
            else:
                qid_list = [q_sample["query_id"] for q_sample in bk_value]
                
                # Create qrels for each bucket
                qrels = {}
                for item in qrels_rel_data:
                    query_id, corpus_id, score = item["query_id"], item["doc_id"], int(item["score"])
                    if item["query_id"] in qid_list:
                        if query_id not in qrels:
                            qrels[query_id] = {corpus_id: score}
                        else:
                            qrels[query_id][corpus_id] = score
            
                            
                if len(qrels) > 0:
                    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values) #retriever.k_values

                    eval_res = [f"all_{bk_name}"]\
                        +list(ndcg.values())\
                        +list(_map.values())\
                        +list(recall.values())\
                        +list(precision.values())
                else:
                    eval_res = [f"{relation_id}_{bk_name}"] + [0]*12

            bk_tsv_writer.writerow(eval_res)
        
    