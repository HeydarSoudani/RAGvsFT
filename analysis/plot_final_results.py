import pandas as pd
import matplotlib.pyplot as plt
import csv, os
import json
import math

relations = [
    'all',
    'occupation', 'mother', 'religion', 'place of birth',
    'genre', 'father', 'screenwriter', 'director',
    'producer', 'author', 'composer', 'country',
    'capital', 'capital of', 'color', 'sport'
]

RELATIONS = {
    "22": "Occupation",
    "218": "Place of birth",
    "91": "Genre",
    "257": "Father",
    "182": "Country",
    "164": "Producer",
    "526": "Director",
    "97": "Capital of",
    "533": "Screenwriter",
    "639": "Composer",
    "472": "Color",
    "106": "Religion",
    "560": "Sport",
    "484": "Author",
    "292": "Mother",
    "422": "Capital"
}

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

def retrieval_results_nobk():
    pass

def retrieval_results_bk():

    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 8))
    fig.delaxes(axes[0,1])  # Remove the first subplot (top-left)
    fig.delaxes(axes[0,2])  # Remove the third subplot (top-right)
    fig.delaxes(axes[0,3])  # Remove the third subplot (top-right)
    
    # file_path = 'results/final_results.tsv'
    
    # file_path = 'component1_retrieval/results/msmarco-distilbert-base-v3_dpr_beir.tsv'
    # file_path = 'component1_retrieval/results/wbk_bm25_eval.tsv'
    # file_path =  'component1_retrieval/results/wbk_noft_dpr_eval.tsv'
    # file_path =  'component1_retrieval/results/wbk_contriever_eval.tsv'
    file_path =  'component1_retrieval/results/wbk_rerank_eval.tsv'
    
    for idx, relation in enumerate(relations):
            
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            tsv_reader = csv.DictReader(file, delimiter='\t')
    
            if idx == 0:
                row = 0
                col = 0
            else:
                row = (idx+3) // 4
                col = (idx+3) % 4
            ax = axes[row, col]
        
            selected_rows = []
            for row in tsv_reader:
                if row['Title'].split('_')[0].lower() == relation:
                    selected_rows.append(row)
            
            # data = {obj['Title'].split('_')[-1]: obj['NDCG@10'] for obj in selected_rows}
            for i in [1, 5, 10, 100]:
                data = {obj['Title'].split('_')[-1]: obj[f'Recall@{i}'] for obj in selected_rows}
                sorted_keys = sorted(data.keys(), key=lambda x: int(x[6:]))  # Sort by the number part of the key
                sorted_data = {k: data[k] for k in sorted_keys}
                
                if idx ==0:                
                    ax.plot(
                        ['b1', 'b2', 'b3', 'b4', 'b5'],
                        [float(item) for item in sorted_data.values()],
                        label=f'Recall@{i}'
                    )
                else:
                    ax.plot(
                        ['b1', 'b2', 'b3', 'b4', 'b5'],
                        [float(item) for item in sorted_data.values()]
                    )
            
            ax.set_title(relation)
            ax.set_ylim(0, 1.05)
    
    # axes[0,2].legend(loc="upper right")
    fig.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def retrieval_results_bk_all_models():
    relation = 'religion'
    models = ["bm25", "contriever", "rerank", "dpr",]
    
    plt.figure(figsize=(10, 6))
    for idx, model in enumerate(models):
        
        file_path = 'component1_retrieval/results/religion/wbk_{}_eval.tsv'.format(model)
        df = pd.read_csv(file_path, sep='\t')
        recall_data = df['Recall@1']
        plt.plot(
            ['b1', 'b2', 'b3', 'b4', 'b5'],
            recall_data,
            label=model
        )
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def icl_obqa_results_bk():
    ret_model = 'bm25' # bm25, dpr_ft, dpr_noft, gt
    file_dir = 'component2_ICL_OBQA/results'
    
    
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 8))
    fig.delaxes(axes[0,1])  # Remove the first subplot (top-left)
    fig.delaxes(axes[0,2])  # Remove the third subplot (top-right)
    fig.delaxes(axes[0,3])  # Remove the third subplot (top-right)
    
    for idx, relation in enumerate(relations):

        if idx == 0:
            row = 0
            col = 0
        else:
            row = (idx+3) // 4
            col = (idx+3) % 4
        ax = axes[row, col]
        
        for ret_model in ['vanilla', 'bm25', 'contriever', 'rerank', 'dpr_noft', 'dpr_ft', 'gt']:
        # for ret_model in ['vanilla', 'clm_vanilla', 'clm_3e_vanilla']:
        
            filename = 'opt1-3_{}_bk.tsv'.format(ret_model)
            file_path = os.path.join(file_dir, filename)
            with open(file_path, 'r', newline='', encoding='utf-8') as file:
                tsv_reader = csv.DictReader(file, delimiter='\t')
            
                selected_rows = []
                for row in tsv_reader:
                    if row['Title'].split('_')[0].lower() == relation:
                        selected_rows.append(row)
        
                data = {obj['Title'].split('_')[-1]: obj['accuracy'] for obj in selected_rows}
                sorted_keys = sorted(data.keys(), key=lambda x: int(x[6:]))  # Sort by the number part of the key
                sorted_data = {k: data[k] for k in sorted_keys}
                
                if idx ==0:
                    ax.plot(
                        ['b1', 'b2', 'b3', 'b4', 'b5'],
                        [float(item) for item in sorted_data.values()],
                        label=ret_model
                    )
                else:
                    ax.plot(
                        ['b1', 'b2', 'b3', 'b4', 'b5'],
                        [float(item) for item in sorted_data.values()]
                    )
                
                ax.set_title(relation)
                ax.set_ylim(0, 1.1)
    
    fig.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def retrieval_results_per_relation():
    retrieval_methods = ['bm25', 'contriever', 'rerank', 'dpr']
    results_dir = "component1_retrieval/results"
    # filename = f"per_rel_{ret_method}_eval.tsv"

def calculated_accuracy(objects):
    correct_count = sum(obj['is_correct'] for obj in objects)
    total_count = len(objects)
    if total_count == 0:
        return 0
    accuracy = correct_count / total_count
    return accuracy
    

def icl_results():
    split_points = [2, 3, 4, 5]
    data_per_relation = {}
    data_per_bucket = {}
    accuracies = {}
    
    data_dir = "component0_preprocessing/generated_data/popQA_EQformat/results"
    filename = "all.flan-t5-base.bf_rag_contriever_nopeft_results.jsonl"
    file_path = os.path.join(data_dir, filename)
    
    data_per_relation['all'] = []
    
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            
            # Divide data per relation
            relation_id = item['query_id'].split('_')[0]
            if relation_id not in data_per_relation:
                data_per_relation[relation_id] = []
            
            data_per_relation['all'].append(item)
            data_per_relation[relation_id].append(item)
            
            # Accuracy per relation
            for relation_id, objects in data_per_relation.items():
                accuracy = calculated_accuracy(objects)
                accuracies[relation_id] = {"overall": accuracy}
    
        # Divide data per bucket
        for relation_id, objects in data_per_relation.items():
            bucket_data = split_to_buckets(objects, split_points)
            accuracies[relation_id]['per_bucket'] = {}
            for bucket_id, bucket_objects in bucket_data.items():
                accuracy = calculated_accuracy(bucket_objects)
                accuracies[relation_id]['per_bucket'][bucket_id] = accuracy
            
    print(accuracies)
    
    # Preparing data for plotting
    relation_name = [RELATIONS[item] if item != 'all' else 'all' for item in list(accuracies.keys())]
    # relation_name = [RELATIONS[item] if item in RELATIONS and item != 'all' else item for item in list(accuracies.keys())]
    overall_scores = [value['overall'] for value in accuracies.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(relation_name, overall_scores, color='skyblue')
    plt.xticks(rotation=90)
    plt.xlabel('Relation ID')
    plt.ylabel('Overall Score')
    plt.title('Overall Scores by Relation ID')
    plt.ylim(0, 1)  # Set y-axis limit to show scores from 0 to 1
    plt.show()
    
    
    
    
    
    


if __name__ == "__main__":
    # retrieval_results_nobk()
    # retrieval_results_bk()
    # retrieval_results_bk_all_models()
    # icl_obqa_results_bk()
    
    
    # retrieval_results_per_relation()
    icl_results()
    
    
    
    
                
            