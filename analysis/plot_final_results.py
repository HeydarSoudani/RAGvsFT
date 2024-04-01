import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv, os
import json
import math
from collections import OrderedDict

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

# rel_pop_score = {
#     '257': 0.6138864810588756,
#     '22': 0.4410058888134515,
#     '91': 0.16768592403443858,
#     '218': 0.1,
#     '106': 0.5067754880971169,
#     '97': 0.6671048731510615,
#     '292': 0.6299331415869965,
#     '164': 0.21263254664084807,
#     '560': 0.11166066064556371,
#     '484': 0.12386409344801165,
#     '533': 0.1784324413416965,
#     '639': 0.17103813440718924,
#     '182': 0.11236930748067743,
#     '422': 1.0,
#     '526': 0.1453979686228685,
#     '472': 0.8016766267589054
# }
rel_pop_score = {'257': 0.6138864810588756, '22': 0.4410058888134515, '91': 0.16768592403443858, '218': 0.1, '106': 0.5067754880971169, '97': 0.6671048731510615, '292': 0.6299331415869965, '164': 0.21263254664084807, '560': 0.11166066064556371, '484': 0.12386409344801165, '533': 0.1784324413416965, '639': 0.17103813440718924, '182': 0.11236930748067743, '422': 1.0, '526': 0.1453979686228685, '472': 0.8016766267589054}


def split_to_buckets(objects, split_points):
    split_points = sorted(split_points)
    sp_len = len(split_points)
    bucket_data = {'b{}'.format(idx+1): list() for idx in range(sp_len+1)}
    
    for obj in objects:
        # rp = obj['relative_popularity']
        if obj['pageviews'] != 0:
            rel_id = obj['query_id'].split('_')[0]
            # rp = math.log(int(obj['pageviews']), 10) * rel_pop_score[rel_id]
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

def plot_bucket_num(split_points):
    dataset_name = 'popqa'
    # split_points = [1, 2, 3, 4]
    data_dir = "component0_preprocessing/generated_data/popQA_EQformat/test"

    if dataset_name == 'popqa':
        fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 8))
        fig.delaxes(axes[0,1])  # Remove the first subplot (top-left)
        fig.delaxes(axes[0,2])  # Remove the third subplot (top-right)
        fig.delaxes(axes[0,3])  # Remove the third subplot (top-right)
        # fig.delaxes(axes[0,4])  # Remove the third subplot (top-right)
    
    if dataset_name == 'eq':
        fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(12, 8))
        fig.delaxes(axes[0,1])  # Remove the first subplot (top-left)
        fig.delaxes(axes[0,2])  # Remove the third subplot (top-right)
        fig.delaxes(axes[0,3])  # Remove the third subplot (top-right)
        fig.delaxes(axes[0,4])  # Remove the third subplot (top-right)
        # fig.delaxes(axes[0,5])  # Remove the third subplot (top-right)

    all_dataset = []
    dataset_per_rel_per_bk = {}
    for filename in os.listdir(data_dir):
        rel_key = filename.split('.')[0]
        file_path = os.path.join(data_dir, filename)
    
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                all_dataset.extend(data)
                data_per_bk = split_to_buckets(data, split_points)
                dataset_per_rel_per_bk[rel_key] = data_per_bk

    data_per_bk = split_to_buckets(all_dataset, split_points)
    dataset_per_rel_per_bk["all"] = data_per_bk
    

    # Plot data for each key
    for idx, key in enumerate(dataset_per_rel_per_bk.keys()):
        if idx == 0:
            row = 0
            col = 0
        else:
            if dataset_name == 'popqa':
                row = (idx+3) // 4
                col = (idx+3) % 4
            if dataset_name == 'eq':
                row = (idx+4) // 5
                col = (idx+4) % 5
        ax = axes[row, col]

        # Count the number of elements in each bucket
        counts = [len(dataset_per_rel_per_bk[key][bucket]) for bucket in dataset_per_rel_per_bk['all'].keys()]

        ax.bar([i+1 for i in range(len(dataset_per_rel_per_bk['all'].keys()))], counts)
        # ax.set_xlabel('log pop')
        ax.set_title(key)

    # plt.title("EQ test-set")
    plt.tight_layout()
    plt.show()

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

def retrieval_results():

    # models = ["bm25"]
    models = ["bm25", "contriever", "rerank", "dpr"]
    data_dir = "component1_retrieval/results"
    per_relation_filename = "per_rel_{}_eval.tsv"
    per_bucket_filename = "per_bk_{}_eval.tsv"
    selected_metric = "Recall@1"
    
    results_per_relation = {}
    for idx, model in enumerate(models):    
        file_path = f"{data_dir}/{per_relation_filename.format(model)}"
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            headers = next(reader)
            recall_index = headers.index(selected_metric)
            # next(reader)
            for row in reader:
                relation_id = row[0]
                if relation_id not in results_per_relation:
                    results_per_relation[relation_id] = {}
                results_per_relation[relation_id][model] = row[recall_index]
                
                
    print(results_per_relation)
    print(len(results_per_relation))
    
    # =============================
    # === Plotting per relation ===
    # num_bars = len(models) # Number of bars per group
    # ind = np.arange(len(results_per_relation)) # Position of bars on x-axis
    # width = 0.11 # Width of a bar
    # width = 0.8 / num_bars 
    
    # fig, ax = plt.subplots() # Plotting the bars
    
    # for i in range(num_bars):
    #     scores = [list(value.values())[i] for value in results_per_relation.values()]
    #     print(scores)
    #     ax.bar(ind + i * width, scores, width, label=models[i])
    
    # ax.set_xlabel('Relation ID')
    # ax.set_ylabel('Accuracy')
    # # ax.set_title(f"Accuracy per Relation: {model_name}, +FT")

    # # relation_names = [RELATIONS[item] if item != 'all' else 'all' for item in list(results_per_relation.keys())]
    # # ax.set_xticks(ind + width * (num_bars - 1) / 2)
    # # ax.set_xticklabels(relation_names)

    # ax.legend()
    # plt.xticks(rotation=25)
    # plt.show()
    # results = OrderedDict(sorted(results_per_relation.items(), key=lambda x: (x[0])))
    
    # sorted_keys = sorted(results_per_relation.keys(), reverse=True)
    # ordered_accuracies = {}
    # for relationId, accuracy in results_per_relation.items():
    #     ordered_accuracies[relationId] = accuracy
    
    # Extract keys and models from the dictionary
    # keys = list(ordered_accuracies.keys())
    # models = list(ordered_accuracies[keys[0]].keys())

    # # Convert results to numpy array for plotting
    # results_array = np.array([[ordered_accuracies[key][model] for model in models] for key in keys])

    # # Plotting multi-bar plot for each key
    # width = 0.2  # Width of each bar
    # x = np.arange(len(keys))  # Index for each key

    # fig, ax = plt.subplots(figsize=(12, 6))

    # for i, model in enumerate(models):
    #     ax.bar(x + i * width, results_array[:, i], width, label=model)

    # ax.set_xlabel('Relation ID')
    # ax.set_ylabel('Accuracy')
    # ax.set_xticks(x + width * (len(models) - 1) / 2)
    # ax.set_xticklabels(keys)
    # ax.legend()
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()
    
def retrieval_results_all_models():

    models = ["bm25", "contriever", "rerank", "dpr",]
    data_dir = "component1_retrieval/results"
    
    per_relation_filename = "per_rel_{}_eval.tsv"
    per_bucket_filename = "per_bk_{}_eval.tsv"
    
    plt.figure(figsize=(10, 6))
    for idx, model in enumerate(models):
        
        # file_path = 'component1_retrieval/results/religion/wbk_{}_eval.tsv'.format(model)
        file_path = f"{data_dir}/{per_bucket_filename.format(model)}"
        df = pd.read_csv(file_path, sep='\t')
        recall_data = df['Recall@1'][-5:]
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

def calculated_accuracy(objects):
    correct_count = sum(obj['is_correct'] for obj in objects)
    total_count = len(objects)
    if total_count == 0:
        return 0
    accuracy = correct_count / total_count
    return accuracy
    
def icl_results(split_points):
    data_per_relation = {}
    accuracies = {}    
    data_dir = "component0_preprocessing/generated_data/popQA_EQformat/results"
    
    # =======================
    # === For FlanT5-small ==
    # model_name = "FlanT5-small"
    # bf_base_filename = "archive/all.flan-t5-small.bf_{}_full_results.jsonl"
    # filenames = [
    #     {"title": "NoFT/NoRAG", "filename": bf_base_filename.format("norag")},
    #     # {"title": "NoFT_bm25RAG", "filename": bf_base_filename.format("rag_bm25")},
    #     # {"title": "NoFT_ContrieverRAG", "filename": bf_base_filename.format("rag_contriever")},
    #     # {"title": "NoFT_RerankRAG", "filename": bf_base_filename.format("rag_rerank")},
    #     # {"title": "NoFT/DprRAG", "filename": bf_base_filename.format("rag_dpr")},
    #     {"title": "NoFT/IdealRAG", "filename": bf_base_filename.format("rag_ideal")},
        
    #     # {"title": "FT_NoRAG_3e", "filename": "archive/all.flan-t5-small_peft_v22.af_e3_norag_peft_results.jsonl"},    
    #     # {"title": "FT_IdealRAG_3e", "filename": "archive/all.flan-t5-small_peft_v22.af_e3_rag_ideal_peft_results.jsonl"},
    #     # {"title": "FT_NoRAG_5e", "filename": "archive/all.flan-t5-small_peft_v22.af_e5_norag_peft_results.jsonl"},    
    #     # {"title": "FT_IdealRAG_5e", "filename": "archive/all.flan-t5-small_peft_v22.af_e5_rag_ideal_peft_results.jsonl"},  
         
        
    #     {"title": "FT/NoRAG", "filename": "archive/all.flan-t5-small_full_v16.af_norag_full_results.jsonl"},
    #     # {"title": "FT/DprRAG", "filename": "archive/all.flan-t5-small_full_v16.af_rag_dpr_full_results.jsonl"},
    #     {"title": "FT/IdealRAG_10e", "filename": "archive/all.flan-t5-small_full_v16.af_rag_ideal_full_results.jsonl"},
        
    #     # # {"title": "FT_NoRAG_extra", "filename": "all.flan-t5-small_full_v19.af_extra_norag_full_results.jsonl"},
    #     # # {"title": "FT_IdealRAG_extra", "filename": "all.flan-t5-small_full_v19.af_extra_rag_ideal_full_results.jsonl"}
    # ]
    
    # =======================
    # === For FlanT5-base ===
    # model_name = "FlanT5-base"
    # bf_base_filename = "archive/all.flan-t5-base.bf_{}_full_results.jsonl"
    # filenames = [
        
    #     {"title": "FT_NoRAG_3e", "filename": "archive/all.flan-t5-base_full_v25.af_e3_norag_full_results.jsonl"},    
    #     {"title": "FT_IdealRAG_3e", "filename": "archive/all.flan-t5-base_full_v25.af_e3_rag_ideal_full_results.jsonl"},
    #     {"title": "FT_NoRAG_5e", "filename": "archive/all.flan-t5-base_full_v25.af_e5_norag_full_results.jsonl"},    
    #     {"title": "FT_IdealRAG_5e", "filename": "archive/all.flan-t5-base_full_v25.af_e5_rag_ideal_full_results.jsonl"},  
        
        
    #     # {"title": "NoFT_NoRAG", "filename": bf_base_filename.format("norag")}, 
    #     # {"title": "NoFT_bm25RAG", "filename": bf_base_filename.format("rag_bm25")},
    #     # {"title": "NoFT_ContrieverRAG", "filename": bf_base_filename.format("rag_contriever")},
    #     # {"title": "NoFT_RerankRAG", "filename": bf_base_filename.format("rag_rerank")},
    #     # {"title": "NoFT_DprRAG", "filename": bf_base_filename.format("rag_dpr")},
    #     # {"title": "NoFT_IdealRAG", "filename": bf_base_filename.format("rag_ideal")},

    #     {"title": "FT_NoRAG_10e", "filename": "archive/all.flan-t5-base_peft_v1.af_norag_peft_results.jsonl"},
    #     # {"title": "FT_DprRAG", "filename": "archive/all.flan-t5-base_peft_v1.af_rag_dpr_peft_results.jsonl"},
    #     {"title": "FT_IdealRAG_10e", "filename": "archive/all.flan-t5-base_peft_v1.af_rag_ideal_peft_results.jsonl"}, 
        
    #     # {"title": "FT_NoRAG_extra", "filename": "all.flan-t5-base_peft_v20.af_extra_norag_peft_results.jsonl"},
    #     # {"title": "FT_IdealRAG_extra", "filename": "all.flan-t5-base_peft_v20.af_extra_rag_ideal_peft_results.jsonl"}
    # ]    
    
    # =======================
    # === For FlanT5-large ==
    # model_name = "FlanT5-large"
    # bf_base_filename = "archive/all.flan-t5-large.bf_{}_full_results.jsonl"
    # filenames = [
        
    #     {"title": "FT_NoRAG_3e", "filename": "archive/all.flan-t5-large_peft_v26.af_e3_norag_peft_results.jsonl"},    
    #     {"title": "FT_IdealRAG_3e", "filename": "archive/all.flan-t5-large_peft_v26.af_e3_rag_ideal_peft_results.jsonl"},
    #     {"title": "FT_NoRAG_5e", "filename": "archive/all.flan-t5-large_peft_v26.af_e5_norag_peft_results.jsonl"},    
    #     {"title": "FT_IdealRAG_5e", "filename": "archive/all.flan-t5-large_peft_v26.af_e5_rag_ideal_peft_results.jsonl"},  
        
    #     # {"title": "NoFT_NoRAG", "filename": bf_base_filename.format("norag")},
    #     # # {"title": "NoFT_bm25RAG", "filename": bf_base_filename.format("rag_bm25")},
    #     # # {"title": "NoFT_ContrieverRAG", "filename": bf_base_filename.format("rag_contriever")},
    #     # # {"title": "NoFT_RerankRAG", "filename": bf_base_filename.format("rag_rerank")},
    #     # {"title": "NoFT_DprRAG", "filename": bf_base_filename.format("rag_dpr")},
    #     # {"title": "NoFT_IdealRAG", "filename": bf_base_filename.format("rag_ideal")},

    #     {"title": "FT_NoRAG_10e", "filename": "archive/all.flan-t5-large_peft_v12.af_norag_peft_results.jsonl"},
    #     # {"title": "FT_DprRAG", "filename": "archive/all.flan-t5-large_peft_v12.af_rag_dpr_peft_results.jsonl"},
    #     {"title": "FT_IdealRAG_10e", "filename": "archive/all.flan-t5-large_peft_v12.af_rag_ideal_peft_results.jsonl"}, 
        
    #     # {"title": "FT_NoRAG_extra", "filename": "all.flan-t5-large_peft_v21.af_extra_norag_peft_results.jsonl"},
    #     # {"title": "FT_IdealRAG_extra", "filename": "all.flan-t5-large_peft_v21.af_extra_rag_ideal_peft_results.jsonl"}
    # ] 
    
        # =======================
    # === For FlanT5-xl ==
    model_name = "FlanT5-xl"
    filenames = [
        {"title": "NoFT_NoRAG", "filename": "all.google.bf_norag_full_results.jsonl"},    
        {"title": "NoFT_IdealRAG", "filename": "all.google.bf_rag_ideal_full_results.jsonl"},
        {"title": "FT_NoRAG", "filename": "all.flan-t5-xl_peft_v7.af_norag_peft_results.jsonl"},    
        {"title": "FT_IdealRAG", "filename": "all.flan-t5-xl_peft_v7.af_rag_ideal_peft_results.jsonl"},  
    ]
    
    # # =======================
    # # === For Llama2 ========
    # model_name = "Llama2"
    # filenames = [
        
    #     {"title": "NoFT_NoRAG", "filename": "all.meta-llama.bf_norag_full_results.jsonl"},    
    #     {"title": "NoFT_IdealRAG", "filename": "all.meta-llama.bf_rag_ideal_full_results.jsonl"},
    #     {"title": "FT_NoRAG", "filename": "all.Llama-2-7b-chat-hf_peft_v6.af_norag_peft_results.jsonl"},    
    #     # {"title": "FT_IdealRAG", "filename": "all.Llama-2-7b-chat-hf_peft_v1.af_rag_ideal_peft_results.jsonl"},  
    # ] 
    
    # =======================
    # === For Mistral =======
    # model_name = "Mistral"
    # filenames = [
        
    #     {"title": "NoFT_NoRAG", "filename": "all.mistralai.bf_norag_full_results.jsonl"},    
    #     {"title": "NoFT_IdealRAG", "filename": "all.mistralai.bf_rag_ideal_full_results.jsonl"},
    #     {"title": "FT_NoRAG", "filename": "all.Mistral-7B-Instruct-v0.1_peft_v4.af_norag_peft_results.jsonl"},    
    #     # {"title": "FT_IdealRAG", "filename": "all.Llama-2-7b-chat-hf_peft_v1.af_rag_ideal_peft_results.jsonl"},  
    # ] 
    
    # =======================
    # === For Mistral =======
    # model_name = "Zephyr"
    # filenames = [
        
    #     {"title": "NoFT_NoRAG", "filename": "all.HuggingFaceH4.bf_norag_full_results.jsonl"},    
    #     {"title": "NoFT_IdealRAG", "filename": "all.HuggingFaceH4.bf_rag_ideal_full_results.jsonl"},
    #     {"title": "FT_NoRAG", "filename": "all.zephyr-7b-beta_peft_v5.af_norag_peft_results.jsonl"},    
    #     # {"title": "FT_IdealRAG", "filename": "all.Llama-2-7b-chat-hf_peft_v1.af_rag_ideal_peft_results.jsonl"},  
    # ] 
    
    
    print(f"Model: {model_name}")
    for idx, _filename in enumerate(filenames):
        title = _filename['title']
        filename = _filename['filename']
        print(f'Processing {title}...')
        
        file_path = f"{data_dir}/{filename}"
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
    # title_font = {
    #     # 'family': 'serif',
    #     'color':  'black',
    #     'weight': 'bold',
    #     'size': 13,
    # }
    
    # num_bars = len(filenames) # Number of bars per group
    # ind = np.arange(len(ordered_accuracies[filenames[0]["title"]])) # Position of bars on x-axis
    # width = 0.11 # Width of a bar
    # fig, ax = plt.subplots(figsize=(19, 5))
    # # plt.figure(figsize=(18, 5)) 
    
    # for i in range(num_bars):
    #     overall_scores = [value['overall'] for value in ordered_accuracies[filenames[i]["title"]].values()]
    #     ax.bar(ind + i * width, overall_scores, width, label=filenames[i]["title"])
    
    # ax.set_xlabel('Relation ID')
    # ax.set_ylabel('Accuracy')
    # ax.set_title(f"C) {model_name}", fontdict=title_font)

    # relation_names = [RELATIONS[item] if item != 'all' else 'all' for item in list(ordered_accuracies[filenames[0]["title"]].keys())]
    # ax.set_xticks(ind + width * (num_bars - 1) / 2)
    # ax.set_xticklabels(relation_names)

    # ax.legend(ncol=2)
    # plt.ylim(0, 1.0)
    # plt.xticks(rotation=25)
    # plt.savefig(f"per_rel_{model_name}.pdf", format='pdf', dpi=1000)
    # plt.show()
    
    
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
        # 'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
    }
    title_font = {
        # 'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 13,
    }
    palette = plt.get_cmap('Set2')
    plt.figure(figsize=(8, 5)) 
    
    # data = [
    #     {'b1': 0.18513223731236597, 'b2': 0.06127027959958578, 'b3': 0.03477443609022556, 'b4': 0.052348993288590606, 'b5': 0.137221269296741},
    #     {'b1': 0.19513223731236597, 'b2': 0.07127027959958578, 'b3': 0.04477443609022556, 'b4': 0.062348993288590606, 'b5': 0.187221269296741},
    #     {'b1': 0.20513223731236597, 'b2': 0.07827027959958578, 'b3': 0.05077443609022556, 'b4': 0.072348993288590606, 'b5': 0.237221269296741},
    #     {'b1': 0.22301644031451037, 'b2': 0.08629616845012081, 'b3': 0.05474624060150376, 'b4': 0.08411633109619687, 'b5': 0.31217838765008576}
    # ]
    for i, (method, _accuracies) in enumerate(ordered_accuracies.items()):
        key = 'all'
        value = _accuracies[key]
        print(method)
        print(value)
        
        if 'per_bucket' in value:
            # buckets = list(value['per_bucket'].keys())
            buckets = [f'$10^{i}$' for i in range(2, 7)]
            scores = list(value['per_bucket'].values())
            
            plt.plot(buckets, scores, label=method, marker='', color=palette(i), linewidth=2.5)
    
    # plt.title(f"A) {model_name}", fontdict=title_font)
    plt.xlabel("Popularity (pageviews)", fontdict=font)
    plt.ylabel("Accuracy", fontdict=font)
    plt.ylim(0, 1.0)
    # plt.legend()
    plt.legend(loc=2, ncol=2, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"main_{model_name}.pdf", format='pdf', dpi=1600)
    plt.savefig(f"main_{model_name}.png", dpi=1600)
    plt.show()
    

if __name__ == "__main__":
    # retrieval_results_nobk()
    # retrieval_results_bk()
    # retrieval_results_bk_all_models()
    # icl_obqa_results_bk()
    # retrieval_results_per_relation()
    # retrieval_results_all_models()
    # retrieval_results()
    
    # split_points = [0.5, 1, 2, 3]
    # split_points = [0.25, 0.4, 0.6, 2]
    split_points = [2, 3, 4, 5]
    
    # plot_bucket_num(split_points)
    
    icl_results(split_points)
    
    
    
    