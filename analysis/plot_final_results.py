import pandas as pd
import numpy as np
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
    bucket_data = {'b{}'.format(idx+1): list() for idx in range(sp_len+1)}
    
    for obj in objects:
        # rp = obj['relative_popularity']
        if obj['pageviews'] != 0:
            rp = math.log(int(obj['pageviews']), 10)
        else:
            rp = 0
        
        if rp < split_points[0]:
            if 'b1' in bucket_data.keys():
                bucket_data['b1'].append(obj)
            else:
                bucket_data['b1'] = [obj]
        
        if rp >= split_points[-1]:
            if 'b{}'.format(sp_len+1) in bucket_data.keys():
                bucket_data['b{}'.format(sp_len+1)].append(obj)
            else:
                bucket_data['b{}'.format(sp_len+1)] = [obj]

        for i in range(sp_len-1):
            if split_points[i] <= rp < split_points[i + 1]:
                if 'b{}'.format(i+2) in bucket_data.keys():
                    bucket_data['b{}'.format(i+2)].append(obj)
                else:
                    bucket_data['b{}'.format(i+2)] = [obj]
    
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
    accuracies = {}    
    data_dir = "component0_preprocessing/generated_data/popQA_EQformat/results"
    

    # =======================
    # === For FlanT5-small ==
    model_name = "FlanT5-small"
    bf_base_filename = "all.flan-t5-small.bf_{}_full_results.jsonl"
    filenames = [
        {"title": "NoFT_NoRAG", "filename": bf_base_filename.format("norag")},
        # {"title": "NoFT_bm25RAG", "filename": bf_base_filename.format("rag_bm25")},
        # {"title": "NoFT_ContrieverRAG", "filename": bf_base_filename.format("rag_contriever")},
        # {"title": "NoFT_RerankRAG", "filename": bf_base_filename.format("rag_rerank")},
        # {"title": "NoFT_DprRAG", "filename": bf_base_filename.format("rag_dpr")},
        {"title": "NoFT_IdealRAG", "filename": bf_base_filename.format("rag_ideal")},
        
        # {"title": "FT_NoRAG", "filename": "all.flan-t5-small_peft_v15.af_norag_peft_results.jsonl"},
        # {"title": "FT_IdealRAG", "filename": "all.flan-t5-small_peft_v15.af_rag_ideal_peft_results.jsonl"},
        
        {"title": "FT_NoRAG", "filename": "all.flan-t5-small_full_v16.af_norag_full_results.jsonl"},
        # {"title": "FT_DprRAG", "filename": "all.flan-t5-small_full_v16.af_rag_dpr_full_results.jsonl"},
        {"title": "FT_IdealRAG", "filename": "all.flan-t5-small_full_v16.af_rag_ideal_full_results.jsonl"},
        
        {"title": "FT_NoRAG_extra", "filename": "all.flan-t5-small_full_v19.af_extra_norag_full_results.jsonl"},
        {"title": "FT_IdealRAG_extra", "filename": "all.flan-t5-small_full_v19.af_extra_rag_ideal_full_results.jsonl"}
    ]
    
    # =======================
    # === For FlanT5-base ===
    # model_name = "FlanT5-base"
    # bf_base_filename = "all.flan-t5-base.bf_{}_full_results.jsonl"
    # filenames = [
    #     {"title": "NoFT_NoRAG", "filename": bf_base_filename.format("norag")}, 
    # #     # {"title": "NoFT_bm25RAG", "filename": bf_base_filename.format("rag_bm25")},
    # #     # {"title": "NoFT_ContrieverRAG", "filename": bf_base_filename.format("rag_contriever")},
    # #     # {"title": "NoFT_RerankRAG", "filename": bf_base_filename.format("rag_rerank")},
    # #     {"title": "NoFT_DprRAG", "filename": bf_base_filename.format("rag_dpr")},
    #     {"title": "NoFT_IdealRAG", "filename": bf_base_filename.format("rag_ideal")},

    #     {"title": "FT_NoRAG", "filename": "all.flan-t5-base_peft_v1.af_norag_peft_results.jsonl"},
    # #     {"title": "FT_DprRAG", "filename": "all.flan-t5-base_peft_v1.af_rag_dpr_peft_results.jsonl"},
    #     {"title": "FT_IdealRAG", "filename": "all.flan-t5-base_peft_v1.af_rag_ideal_peft_results.jsonl"}, 
        
    #     {"title": "FT_NoRAG_extra", "filename": "all.flan-t5-base_peft_v20.af_extra_norag_peft_results.jsonl"},
    #     {"title": "FT_IdealRAG_extra", "filename": "all.flan-t5-base_peft_v20.af_extra_rag_ideal_peft_results.jsonl"}
    # ]    
    
    # =======================
    # === For FlanT5-large ==
    model_name = "FlanT5-large"
    bf_base_filename = "all.flan-t5-large.bf_{}_full_results.jsonl"
    filenames = [
        {"title": "NoFT_NoRAG", "filename": bf_base_filename.format("norag")},
        
        # {"title": "NoFT_bm25RAG", "filename": bf_base_filename.format("rag_bm25")},
        # {"title": "NoFT_ContrieverRAG", "filename": bf_base_filename.format("rag_contriever")},
        # {"title": "NoFT_RerankRAG", "filename": bf_base_filename.format("rag_rerank")},
        {"title": "NoFT_DprRAG", "filename": bf_base_filename.format("rag_dpr")},
        {"title": "NoFT_IdealRAG", "filename": bf_base_filename.format("rag_ideal")},

        {"title": "FT_NoRAG", "filename": "all.flan-t5-large_peft_v12.af_norag_peft_results.jsonl"},
        {"title": "FT_DprRAG", "filename": "all.flan-t5-large_peft_v12.af_rag_dpr_peft_results.jsonl"},
        {"title": "FT_IdealRAG", "filename": "all.flan-t5-large_peft_v12.af_rag_ideal_peft_results.jsonl"}, 
        
        {"title": "FT_NoRAG_extra", "filename": "all.flan-t5-small_full_v19.af_extra_norag_full_results.jsonl"},
        {"title": "FT_IdealRAG_extra", "filename": "all.flan-t5-small_full_v19.af_extra_rag_ideal_full_results.jsonl"}
    ] 
    
    print(f"Model: {model_name}")
    for idx, _filename in enumerate(filenames):
        title = _filename['title']
        filename = _filename['filename']
        print(f'Processing {title}...')
        
        file_path = os.path.join(data_dir, filename)
        data_per_relation[title] = {}
        data_per_relation[title]['all'] = []
        accuracies[title] = {}
    
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            for line in file:
                item = json.loads(line)
                
                # Divide data per relation
                relation_id = item['query_id'].split('_')[0]
                if relation_id not in data_per_relation[title]:
                    data_per_relation[title][relation_id] = []
                
                data_per_relation[title]['all'].append(item)
                data_per_relation[title][relation_id].append(item)
                
                # Accuracy per relation
                for relation_id, obj in data_per_relation[title].items():
                    rel_accuracy = calculated_accuracy(obj)
                    accuracies[title][relation_id] = {"overall": rel_accuracy}
    
        # Divide data per bucket
        for relation_id, objects in data_per_relation[title].items():
            bucket_data = split_to_buckets(objects, split_points)
            accuracies[title][relation_id]['per_bucket'] = {}
            for bucket_id, bucket_objects in bucket_data.items():
                accuracy = calculated_accuracy(bucket_objects)
                accuracies[title][relation_id]['per_bucket'][bucket_id] = accuracy
    
    sorted_keys = sorted(accuracies[filenames[0]['title']].keys(), reverse=True)
    ordered_accuracies = {}
    for filename, accuracy in accuracies.items():
        ordered_dict = {k: accuracy[k] for k in sorted_keys}
        ordered_accuracies[filename] = ordered_dict
    
    for title, accuracy in ordered_accuracies.items():
        print(f"Title: {title}")
        # print(accuracy)
        for relation_id, value in accuracy.items():
            rel_name = RELATIONS[relation_id] if relation_id != 'all' else 'all'
            print(f"Relation ID: {relation_id}, {rel_name} -> {value}")
            # print(value)
        print('\n')
    
    
    # =============================
    # === Plotting per relation ===
    num_bars = len(filenames) # Number of bars per group
    ind = np.arange(len(ordered_accuracies[filenames[0]["title"]])) # Position of bars on x-axis
    width = 0.11 # Width of a bar
    fig, ax = plt.subplots() # Plotting the bars
    
    for i in range(num_bars):
        overall_scores = [value['overall'] for value in ordered_accuracies[filenames[i]["title"]].values()]
        ax.bar(ind + i * width, overall_scores, width, label=filenames[i]["title"])
    
    ax.set_xlabel('Relation ID')
    ax.set_ylabel('Accuracy')
    ax.set_title(f"Accuracy per Relation: {model_name}, +FT")

    relation_names = [RELATIONS[item] if item != 'all' else 'all' for item in list(ordered_accuracies[filenames[0]["title"]].keys())]
    ax.set_xticks(ind + width * (num_bars - 1) / 2)
    ax.set_xticklabels(relation_names)

    ax.legend()
    plt.xticks(rotation=25)
    plt.show()
    
    
    # =============================
    # === Plotting per bucket =====
    # num_plots = len(ordered_accuracies[title])
    # cols = 4
    # rows = (num_plots + cols - 1) // cols

    # fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2))
    # fig.delaxes(axes[0,1])  
    # fig.delaxes(axes[0,2])  
    # fig.delaxes(axes[0,3]) 
    
    # for _title, _accuracies in ordered_accuracies.items():
    #     for idx, (key, value) in enumerate(_accuracies.items()):
            
    #         if idx == 0:
    #             row = 0
    #             col = 0
    #         else:
    #             row = (idx+3) // 4
    #             col = (idx+3) % 4
    #         ax = axes[row, col]
            
    #         if 'per_bucket' in value:  # Check if 'per_bucket' exists to avoid errors
    #             buckets = list(value['per_bucket'].keys())
    #             scores = list(value['per_bucket'].values())
    #             if idx == 0:
    #                 ax.plot(buckets, scores,  label=_title)
    #             else:
    #                 ax.plot(buckets, scores, )
    #             rel_name = RELATIONS[key] if key != 'all' else 'all'
    #             ax.set_title(f"{rel_name}")
    #             ax.set_xlabel("")
    #             ax.set_ylabel("Accuracy")
    #             ax.set_ylim(0, 1)

    # fig.legend(loc='upper right')
    # plt.tight_layout()
    # plt.show()
    
    # =====================================
    # === Only plot "all", per bucket =====
    # plt.style.use('seaborn-darkgrid')
    font = {
        'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
    }
    palette = plt.get_cmap('Set2')
    
    for i, (method, _accuracies) in enumerate(ordered_accuracies.items()):
        key = 'all'
        value = _accuracies[key]
        
        if 'per_bucket' in value:  # Check if 'per_bucket' exists to avoid errors
            buckets = list(value['per_bucket'].keys())
            buckets = [f'$10^{i}$' for i in range(2, 7)]
            scores = list(value['per_bucket'].values())
            plt.plot(buckets, scores, label=method, marker='', color=palette(i), linewidth=2.5)
    
    # plt.title(f"Accuracy per bucket, {model_name}", fontdict=font)
    plt.xlabel("Popularity (pageviews)", fontdict=font)
    plt.ylabel("Accuracy", fontdict=font)
    plt.ylim(0, 1.0)
    # plt.legend()
    plt.legend(loc=2, ncol=3)
    plt.tight_layout()
    plt.savefig(f"main_{model_name}_extra.png", dpi=1000)
    plt.show()
    

if __name__ == "__main__":
    # retrieval_results_nobk()
    # retrieval_results_bk()
    # retrieval_results_bk_all_models()
    # icl_obqa_results_bk()
    
    
    # retrieval_results_per_relation()
    icl_results()
    
    
    
    
                
            