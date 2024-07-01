
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import json
import os

# === Datasets variables ========================
dataset_name = 'EQ' # [popQA, witQA, EQ]
model_idx = -3

retrieval_models = ["bm25", "contriever", "rerank", "dpr"]
gen_models = [
    "flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl",
    "stable_lm2", "tiny_llama", "MiniCPM",
    "llama2", "mistral", "zephyr"
]
dataset_dir = 'component0_preprocessing/generated_data/{}_costomized'.format(dataset_name)
test_dir = f"{dataset_dir}/test"
answer_generator_img_save_path = f'analysis/images_results/3_answer_generator/llms/{gen_models[model_idx]}_{dataset_name}.png'

# PopQA
if dataset_name == 'popQA':
    tsv_file_path = "data/dataset/popQA/popQA.tsv"
    split_points = [2, 3, 4, 5]
    num_relations = 16
    relation_ids = ['22', '91', '97', '106', '164', '182', '218', '257', '292', '422', '472', '484', '526', '533', '560', '639']
    RELATIONS = {
        '22': 'Occupation',
        '91': 'Genre',
        '97': 'Capital of',
        '106': 'Religion',
        '164': 'Producer',
        '182': 'Country',
        '218': 'Place of birth',
        '257': 'Father',
        '292': 'Mother',
        '422': 'Capital',
        '472': 'Color',
        '484': 'Author',
        '526': 'Director',
        '533': 'Screenwriter',
        '560': 'Sport',
        '639': 'Composer'
    }

# WitQA
elif dataset_name == 'witQA':
    tsv_file_path = "data/dataset/WitQA/witqa.tsv"
    num_relations = 32
    relation_ids = ['17', '19', '22', '25', '27', '36', '50', '57', '58', '69', '86', '106', '123', '136', '140', '149', '162', '184', '344', '452', '462', '641', '674', '1038', '1050', '1376', '1431', '1433', '2012', '2936', '3301', '4647']
    split_points = [2, 3, 4, 5]
    RELATIONS = {
        '17': 'country',
        '19': 'place of birth',
        '22': 'father',
        '25': 'mother',
        '27': 'country of ci.', # country of citizenship
        '36': 'capital',
        '50': 'author',
        '57': 'director',
        '58': 'screenwriter',
        '69': 'educated at',
        '86': 'composer',
        '106': 'occupation',
        '123': 'publisher',
        '136': 'genre',
        '140': 'religion',
        '149': 'architectural st.', # architectural style
        '162': 'producer',
        '184': 'doctoral adv.', # doctoral advisor
        '344': 'director of ph.', # director of photography
        '452': 'industry',
        '462': 'color',
        '641': 'sport',
        '674': 'characters',
        '1038': 'relative',
        '1050': 'medical cond.', # medical condition
        '1376': 'capital of',
        '1431': 'executive prod.', # executive producer
        '1433': 'published in',
        '2012': 'cuisine',
        '2936': 'language used',
        '3301': 'broadcast by',
        '4647': 'loc of perf.' # location of first performance
    }

# EQ
elif dataset_name == 'EQ':
    sub_type = 'test'
    data_evidence_dir = "data/dataset/entity_questions_dataset/data_evidence"
    num_relations = 25
    relation_ids = ['17', '19', '20', '26', '30', '36', '40', '50', '69', '106', '112', '127', '131', '136', '159', '170', '175', '176', '264', '276', '407', '413', '495', '740', '800']
    split_points = [1, 3, 5, 7]
    RELATIONS = {
        '17': 'country loc. in', # country located in
        '19': 'birth place',
        '20': 'death place',
        '26': 'spouse',
        '30': 'continent',
        '36': 'capital',
        '40': 'child',
        '50': 'author',
        '69': 'edu place', # education place
        '106': 'occupation',
        '112': 'founder',
        '127': 'owner',
        '131': 'location',
        '136': 'music genre',
        '159': 'headquarters',
        '170': 'creator',
        '175': 'performer',
        '176': 'prod. company', # producer company
        '264': 'music label',
        '276': 'location',
        '407': 'lang. wr. in', # language written in
        '413': 'fame reason',
        '495': 'cr. country', # creation country
        '740': 'founding place',
        '800': 'pos. played' # position played
    }

# === Functions =================================
def split_to_buckets(objects, split_points):
    
    split_points = sorted(split_points)
    sp_len = len(split_points)
    bucket_data = {'bucket{}'.format(idx+1): list() for idx in range(sp_len+1)}
    
    for obj in objects:
        # rp = obj['relative_popularity']
        if int(obj['pageviews']) > 0:
            if dataset_name in ['popQA', 'witQA']:
                rp = math.log(int(obj['pageviews']), 10)
            elif dataset_name == 'EQ':
                rp = math.log(int(obj['pageviews']), 2)
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

def plot_buckets_distribution(only_all=False):
     
    if not only_all:
        if dataset_name == 'popQA':
            ncols = 4
            nrows = 5
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        elif dataset_name == 'EQ':
            ncols = 5
            nrows = 6
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        elif dataset_name == 'witQA':
            ncols = 5
            nrows = 8
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
        
        fig.delaxes(axes[0,1])
        fig.delaxes(axes[0,2])
        fig.delaxes(axes[0,3])
        if dataset_name == 'witQA':
            fig.delaxes(axes[0,4])
            fig.delaxes(axes[7,2])
            fig.delaxes(axes[7,3])
            fig.delaxes(axes[7,4])
        
        if dataset_name == 'EQ':
            fig.delaxes(axes[0,4])
        
        all_queries = []
        for idx, filename in enumerate(os.listdir(test_dir)):
            if filename.endswith('.json'):
                relation_id = filename.split('.')[0]
                print(f"Processing relation {relation_id}, {RELATIONS[relation_id]} ...")
                
                row = (idx // ncols) + 1
                col = idx % ncols
                ax = axes[row, col]
                            
                query_file_path = os.path.join(test_dir, filename)
                with open(query_file_path, 'r') as qf:
                    q_rel_data = json.load(qf) 
                all_queries.extend(q_rel_data)
                
                bk_data = split_to_buckets(q_rel_data, split_points)
                counts = [len(bk) for bk in bk_data.values()]
                ax.bar(["b1", "b2", "b3", "b4", "b5"], counts)
                ax.set_title(RELATIONS[relation_id])
        
        # q_file = 'queries_with_rerank_overlap_2.jsonl'
        # all_queries_obj = {}
        # all_queries = []
        # with open (q_file, 'r') as file:
        #     for line in file:
        #         data = json.loads(line.strip())
        #         relation_id = data['query_id'].split('_')[0]
        #         if relation_id not in all_queries_obj:
        #             all_queries_obj[relation_id] = [data]
        #         else:
        #             all_queries_obj[relation_id].append(data)
        #         all_queries.append(data)
        
        # for idx, (relation_id, q_rel_data) in enumerate(all_queries_obj.items()):
        #     print(f"Processing relation {relation_id}, {RELATIONS[relation_id]} ...")
        #     row = (idx // ncols) + 1
        #     col = idx % ncols
        #     ax = axes[row, col]
            
        #     bk_data = split_to_buckets(q_rel_data, split_points)
        #     counts = [len(bk) for bk in bk_data.values()]
        #     ax.bar(["b1", "b2", "b3", "b4", "b5"], counts)
        #     ax.set_title(RELATIONS[relation_id])
        
        row = 0
        col = 0
        ax = axes[row, col]
        bk_data = split_to_buckets(all_queries, split_points)
        counts = [len(bk) for bk in bk_data.values()]
        ax.bar(["b1", "b2", "b3", "b4", "b5"], counts)
        ax.set_title('all')   
        
        plt.tight_layout()
        plt.show()
    
    if only_all:
        all_queries = []
        for idx, filename in enumerate(os.listdir(test_dir)):
            if filename.endswith('.json'):
                query_file_path = os.path.join(test_dir, filename)
                with open(query_file_path, 'r') as qf:
                    q_rel_data = json.load(qf) 
                all_queries.extend(q_rel_data)
        
        # q_file = 'queries_with_rerank_overlap_2.jsonl'
        # all_queries = []
        # with open (q_file, 'r') as file:
        #     for line in file:
        #         data = json.loads(line.strip())
        #         all_queries.append (data)
        
        plt.figure(figsize=(8, 5)) 
        font = {
            # 'family': 'serif',
            'color':  'black',
            # 'weight': 'bold',
            'size': 16,
        }
    
        bk_data = split_to_buckets(all_queries, split_points)
        counts = [len(bk) for bk in bk_data.values()]
        buckets = ["b1", "b2", "b3", "b4", "b5"]
        # buckets = [f'$10^{i}$' for i in range(2, 7)]
    
        color_left = [0.69, 0.769, 0.871]  # lightsteelblue in RGB
        color_right = [0.255, 0.412, 0.882]  # royalblue in RGB
        interpolation_values = np.linspace(0, 1, len(buckets))
        # colors = [(color_left, color_right, value) for value in interpolation_values]
        
        for i, (bucket, value) in enumerate(zip(buckets, counts)):
            color = (1 - interpolation_values[i]) * np.array(color_left) + \
                interpolation_values[i] * np.array(color_right)    
            plt.bar(bucket, value, color=color)
        
        plt.title(f'{dataset_name}', fontdict={'size': 18, 'weight': 'bold'})
        plt.xlabel("Popularity (pageviews)", fontdict=font)
        plt.xticks(fontsize=14)
        plt.ylabel("# Samples", fontdict=font)
        plt.yticks(fontsize=14)
        plt.yticks(rotation=45)
        
        plt.tight_layout()   
        plt.savefig(f'analysis/images_results/1_popularity_per_bucket/bk_dist_{dataset_name}.png', dpi=1000)        
        plt.show()

def plot_retriever_results_per_relation():
    
    file_data = {}
    for ret_model in retrieval_models:
        file_path = f'component1_retrieval/results/{dataset_name}/per_rel_{ret_model}_eval.tsv'
        df = pd.read_csv(file_path, sep='\t', usecols=['Title', 'Recall@1'])
        file_data[ret_model] = df
    
    all_data = pd.DataFrame()
    for ret_model, data in file_data.items():
        pivoted = data.pivot_table(index='Title', values='Recall@1', aggfunc='first').rename(columns={'Recall@1': ret_model})
        all_data = pd.concat([all_data, pivoted], axis=1)
    
    all_data.fillna(0, inplace=True)
    all_data.index = all_data.index.map(lambda x: RELATIONS.get(x, x))
    
    if dataset_name == 'popQA':
        figsize = (15, 8)
    elif dataset_name == 'witQA':
        figsize = (20, 8)
    elif dataset_name == 'EQ':
        figsize = (18, 8)
    
    all_data.plot(kind='bar', figsize=figsize, width=0.7)
    plt.title('Recall@1 per Relation')
    plt.xlabel('Relation')
    plt.ylabel('Recall@1')
    plt.xticks(np.arange(len(all_data.index)), all_data.index, rotation=45)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    vline_position = len(all_data.index) - 1.5
    plt.axvline(x=vline_position, color='grey', linestyle='--')
    plt.show()

def plot_retriever_results_per_buckets(only_all=False):
    
    data_frames = []
    for ret_model in retrieval_models:
        file_path = f'component1_retrieval/results/top1/{dataset_name}/per_bk_{ret_model}_eval.tsv'
        df = pd.read_csv(file_path, sep='\t')
        df['Model'] = ret_model
        df[['RelationID', 'Bucket']] = df['Title'].str.extract('(\d+|all)_bucket(\d+)')
        df['Bucket'] = pd.to_numeric(df['Bucket'])
        data_frames.append(df)
        
    combined_df = pd.concat(data_frames)
    combined_df['RelationID'] = combined_df['RelationID'].astype(str)
    relations = sorted(set(combined_df['RelationID']), key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))
    models = combined_df['Model'].unique()
    
    if not only_all:
        if dataset_name == 'popQA':
            ncols = 4
            coref = 1.9
        elif dataset_name == 'witQA':
            ncols = 6
            coref = 1.4
        elif dataset_name == 'EQ':
            ncols = 5
            coref = 1.5  
        nrows = math.ceil(num_relations / ncols) + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * coref))
            
        fig.delaxes(axes[0,1])
        fig.delaxes(axes[0,2])
        if dataset_name == 'witQA':
            fig.delaxes(axes[0,3])
            fig.delaxes(axes[0,4])
            fig.delaxes(axes[6,2])
            fig.delaxes(axes[6,3])
            fig.delaxes(axes[6,4])
            fig.delaxes(axes[6,5])
        if dataset_name == 'EQ':
            fig.delaxes(axes[0,3])
        
        custom_xticks = ['b1', 'b2', 'b3', 'b4', 'b5']
        
        for idx, relation in enumerate(relations):
            if idx == 0:
                ax = axes[0, 0]
            else:
                row = ((idx-1) // ncols) +1
                col = (idx-1) % ncols
                ax = axes[row, col]
            
            for model in models:
                
                if model == 'bm25':
                    model_lb = 'BM25'
                elif model == 'contriever':
                    model_lb = 'Contriever'
                elif model == 'rerank':
                    model_lb = 'BM25+DPR'
                elif model == 'dpr':
                    model_lb = 'DPR'
                
                model_df = combined_df[(combined_df['Model'] == model) & (combined_df['RelationID'] == relation)]
                model_df = model_df.sort_values('Bucket')  # Sort by bucket for proper line plotting
                ax.plot(model_df['Bucket'], model_df['Recall@1'], '-o', label=model_lb)
            ax.set_ylim(0, 1.1)
            
            if idx == 0:
                ax.set_title(f'{relation}')
                # ax.set_xlabel('Bucket')
                ax.set_ylabel('Recall@1')
                ax.set_xticks(range(1, len(custom_xticks) + 1))
                ax.set_xticklabels(custom_xticks)
            else:
                ax.set_title(f'{RELATIONS[relation]}')
                ax.set_xlabel('')  # Remove x-label for others
                ax.set_ylabel('')
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x-ticks for subplots other than the first

        axes_flat = axes.flatten()
        legend_ax = axes_flat[ncols-1]  # Adjust based on desired legend position
        handles, labels = axes_flat[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, title='Model', loc='center', fontsize='12')
        legend_ax.axis('off')  # Turn off axis lines and labels
        
        plt.subplots_adjust(hspace=0.5)
        plt.show()
    
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        custom_xticks = ['b1', 'b2', 'b3', 'b4', 'b5']
        # ax = axes[0]
        for model in models:
            
            if model == 'bm25':
                model_lb = 'BM25'
            elif model == 'contriever':
                model_lb = 'Contriever'
            elif model == 'rerank':
                model_lb = 'BM25+DPR'
            elif model == 'dpr':
                model_lb = 'DPR'
            
            model_df = combined_df[(combined_df['Model'] == model) & (combined_df['RelationID'] == relations[0])]
            model_df = model_df.sort_values('Bucket') 
            axes.plot(model_df['Bucket'], model_df['Recall@1'], '-o', label=model_lb, linewidth=3)
        
        axes.set_title(f'{dataset_name}', fontdict={'size': 16, 'weight': 'bold'})
        axes.set_ylim(0, 1.1)
        axes.set_ylabel('Recall@1', fontsize=12)
        axes.set_xticks(range(1, len(custom_xticks) + 1))
        axes.set_xticklabels(custom_xticks, fontsize=12)
        
        plt.legend(title='Model', loc="upper right", ncol=1, fontsize=12,
                   title_fontsize=14
        )
        plt.tight_layout()
        plt.savefig(f'analysis/images_results/2_ret_buckets/ret_recall_{dataset_name}.png', dpi=1400)
        # plt.savefig(f'analysis/images_results/2_ret_buckets/ret_recall_{dataset_name}.pdf', format='pdf', dpi=1000)
        plt.show()
   
def calculated_accuracy(objects):
    correct_count = sum(obj['is_correct'] for obj in objects)
    total_count = len(objects)
    if total_count == 0:
        return 0
    accuracy = correct_count / total_count
    return accuracy

def plot_answer_generator_results(per_relation=False, per_bucket=False, only_all=False):
    
    ### ==== Define Variables =============
    model_name = gen_models[model_idx]
    
    if model_name in ["flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl"]:
        model_type = 'flant5'
    elif model_name in ["stable_lm2", "tiny_llama", "MiniCPM"]:
        model_type = 'slms'
    elif model_name in ["llama2", "mistral", "zephyr"]:
        model_type = 'llms'
    
    base_path  = "component0_preprocessing/generated_data"
    retrieval_model = 'ideal'
    result_files = [
        
        # FlanT5-small: 
        {"title": f"-FT/-RAG", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_bf_norag_full_results.jsonl"},
        {"title": f"-FT/idealRAG", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_bf_rag_ideal_full_results.jsonl"},
        {"title": f"+FT/-RAG", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_af_norag_peft_results.jsonl"},
        {"title": f"+FT/idealRAG", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_af_rag_ideal_peft_results.jsonl"},
        
        # FlanT5 base / No FT / popQA
        # {"title": f"-FT/-RAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/flant5/popQA_flant5_bs_bf_norag_full_results.jsonl"},
        # {"title": f"-FT/bm25RAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_flant5_base_1p_bf_rag_bm25_full_1_results.jsonl"},
        # {"title": f"-FT/contriverRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_flant5_base_1p_bf_rag_contriever_full_1_results.jsonl"},
        # {"title": f"-FT/rerankRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_flant5_base_1p_bf_rag_rerank_full_1_results.jsonl"},
        # {"title": f"-FT/dprRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/flant5/popQA_flant5_bs_bf_rag_dpr_full_results.jsonl"},
        # {"title": f"-FT/idealRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/flant5/popQA_flant5_bs_bf_rag_ideal_full_results.jsonl"},
        
        # FlanT5 base / with FT / popQA
        # {"title": f"+FT/-RAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/flant5/popQA_flant5_bs_af_norag_peft_results.jsonl"},
        # {"title": f"+FT/bm25RAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_flant5_base_1p_af_rag_bm25_peft_1g_results.jsonl"},
        # {"title": f"+FT/contriverRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_flant5_base_1p_af_rag_contriever_peft_1g_results.jsonl"},
        # {"title": f"+FT/rerankRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_flant5_base_1p_af_rag_rerank_peft_1g_results.jsonl"},
        # {"title": f"+FT/dprRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_flant5_base_1p_af_rag_dpr_peft_results.jsonl"},
        # {"title": f"+FT/idealRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/flant5/popQA_flant5_bs_af_rag_ideal_peft_results.jsonl"},

        # StableLM2 / No FT / popQA
        # {"title": f"-FT/-RAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/slms/popQA_stable_lm2_bf_norag_full_results.jsonl"},
        # {"title": f"-FT/bm25RAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/slms/popQA_stable_lm2_bf_rag_bm25_full_results.jsonl"},
        # {"title": f"-FT/contriverRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_stable_lm2_1p_bf_rag_contriever_full_results.jsonl"},
        # {"title": f"-FT/rerankRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_stable_lm2_1p_bf_rag_rerank_full_1_results.jsonl"},
        # {"title": f"-FT/dprRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/slms/popQA_stable_lm2_bf_rag_dpr_full_results.jsonl"},
        # {"title": f"-FT/idealRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/slms/popQA_stable_lm2_bf_rag_ideal_full_results.jsonl"},
        
        # StableLM2 / with FT / popQA
        # {"title": f"+FT/-RAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/slms/popQA_stable_lm2_af_norag_peft_results.jsonl"},
        # {"title": f"+FT/bm25RAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_stable_lm2_1p_af_rag_bm25_peft_1g_results.jsonl"},
        # {"title": f"+FT/contriverRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_stable_lm2_1p_af_rag_contriever_peft_1g_results.jsonl"},
        # {"title": f"+FT/rerankRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/highlighting_retrievers/popQA_stable_lm2_1p_af_rag_rerank_peft_1g_results.jsonl"},
        # {"title": f"+FT/dprRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/slms/popQA_stable_lm2_af_rag_dpr_peft_results.jsonl"},
        # {"title": f"+FT/idealRAG", "filename": f"component0_preprocessing/generated_data/popQA_costomized/results/slms/popQA_stable_lm2_af_rag_ideal_peft_results.jsonl"},
        
        # {"title": "NoFT/idealRAG", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_bf_rag_ideal_full_results.jsonl"},
        # {"title": "FT/idealRAG", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_af_rag_ideal_peft_results.jsonl"},
        
        # {'title': '-FT+2RS+3P', 'filename': 'component0_preprocessing/generated_data/popQA_costomized/results/for_reranking/popQA_flant5_base_1rp_3p_bf_rag_dpr_full_results.jsonl'},
        # {'title': '1r_2p', 'filename': 'analysis/on_false_results/popQA_MiniCPM_1rag_2p_bf_rag_ideal_full_results_two_sided.jsonl'},
        # {'title': '2r_2p', 'filename': 'analysis/on_false_results/popQA_MiniCPM_2rags_2p_bf_rag_ideal_full_results_two_sided.jsonl'},
        # {"title": f"voting", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_voting_results.jsonl"},
        # {"title": f"voting_2", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_voting_2_results.jsonl"},
        # {"title": f"NoFT/bm25RAG", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_bf_rag_bm25_full_results.jsonl"},
        # {"title": f"NoFT/dprRAG", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_bf_rag_dpr_full_results.jsonl"},
        # {"title": "Ens/wo_cot", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_moe_llm_results_(no_cot).jsonl"},
        # {"title": "Ens/w_cot", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_moe_llm_results_(cot).jsonl"},
        # {"title": "Ens/ideal", "filename": f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_moe_ideal_results.jsonl"},
    ]
    
    ### ==== Prepare data for plotting ====
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print('\n')
    
    data_per_relation = {}
    accuracies = {} 
    for idx, result_file in enumerate(result_files):
        title = result_file['title']
        file_path = result_file['filename']
        print(f'Processing {title}...')
        
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
    
    sorted_keys = sorted(accuracies[result_files[0]['title']].keys(), reverse=True)
    ordered_accuracies = {}
    for filename, accuracy in accuracies.items():
        ordered_dict = {k: accuracy[k] for k in sorted_keys}
        ordered_accuracies[filename] = ordered_dict
    
    for title, accuracy in ordered_accuracies.items():
        print(f"Title: {title}")
        for relation_id, value in accuracy.items():
            rel_name = RELATIONS[relation_id] if relation_id != 'all' else 'all'
            print(f"Relation ID: {relation_id}, {rel_name} -> {value}")
        print('\n')

    ### ==== Plotting configurations ====
    palette = plt.get_cmap('Set2')
    title_font = {
        # 'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 13,
    }
    font = {
        'color':  'black',
        'weight': 'normal',
        'size': 14,
    }
    
    ### === Plotting per relation ===
    if per_relation:
        if dataset_name == 'popQA':
            figsize = (15, 8)
        elif dataset_name == 'witQA':
            figsize = (20, 8)
        elif dataset_name == 'EQ':
            figsize = (18, 8)
        
        num_bars = len(result_files) # Number of bars per group
        ind = np.arange(len(ordered_accuracies[result_files[0]["title"]])) # Position of bars on x-axis
        width = 0.11 # Width of a bar
        fig, ax = plt.subplots(figsize=figsize)
        
        for i in range(num_bars):
            overall_scores = [value['overall'] for value in ordered_accuracies[result_files[i]["title"]].values()]
            ax.bar(ind + i * width, overall_scores, width, label=result_files[i]["title"], color=palette(i))
        
        ax.set_title(f"{model_name}", fontdict=title_font)
        ax.set_xlabel('Relation')
        ax.set_ylabel('Accuracy')
        
        relation_names = [RELATIONS[item] if item != 'all' else 'all' for item in list(ordered_accuracies[result_files[0]["title"]].keys())]
        ax.set_xticks(ind + width * (num_bars - 1) / 2)
        ax.set_xticklabels(relation_names)
        ax.legend(ncol=2)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=25)
        
        vline_position = 0.5
        plt.axvline(x=vline_position, color='grey', linestyle='--')
        
        # plt.savefig(f"per_rel_{model_name}.pdf", format='pdf', dpi=1000)
        plt.show()
    
    ### === Plotting per bucket =====
    if per_bucket:
        
        if dataset_name == 'popQA':
            ncols = 4
            coref = 1.9
        elif dataset_name == 'witQA':
            ncols = 6
            coref = 1.4
        elif dataset_name == 'EQ':
            ncols = 5
            coref = 1.5
        nrows = math.ceil(num_relations / ncols) + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * coref))
        
        fig.delaxes(axes[0,1])
        fig.delaxes(axes[0,2])
        if dataset_name == 'witQA':
            fig.delaxes(axes[0,3])
            fig.delaxes(axes[0,4])
            fig.delaxes(axes[6,2])
            fig.delaxes(axes[6,3])
            fig.delaxes(axes[6,4])
            fig.delaxes(axes[6,5])
        if dataset_name == 'EQ':
            fig.delaxes(axes[0,3])
        
        custom_xticks = ['b1', 'b2', 'b3', 'b4', 'b5']
        
        for _title, _accuracies in ordered_accuracies.items():
            for idx, (key, value) in enumerate(_accuracies.items()):
                
                if idx == 0:
                    ax = axes[0, 0]
                else:
                    row = (idx+ncols-1) // ncols
                    col = (idx+ncols-1) % ncols
                    ax = axes[row, col]
                
                if 'per_bucket' in value:  # Check if 'per_bucket' exists to avoid errors
                    buckets = list(value['per_bucket'].keys())
                    scores = list(value['per_bucket'].values())
                    ax.plot(buckets, scores, label=_title, marker='') # color=palette(idx), linewidth=2.5
                    
                    rel_name = RELATIONS[key] if key != 'all' else 'all'
                    ax.set_title(f"{rel_name}")
                    ax.set_ylim(0, 1.1)
                    if idx == 0:
                        ax.set_ylabel("Accuracy") 
                        ax.set_xticks(range(0, len(custom_xticks)))
                        ax.set_xticklabels(custom_xticks) 
                    else:
                        ax.set_xlabel("")
                        ax.set_ylabel('')
                        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x-ticks for subplots other than the first

        # fig.legend(loc='upper right')
        axes_flat = axes.flatten()
        legend_ax = axes_flat[ncols-1]  # Adjust based on desired legend position
        handles, labels = axes_flat[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, title='Model', loc='center')
        legend_ax.axis('off')  # Turn off axis lines and labels
        
        plt.tight_layout()
        plt.show()
        
    ### === Only plot "all", per bucket =====
    if only_all:
        plt.figure(figsize=(8, 4)) 
        custom_xticks = ['b1', 'b2', 'b3', 'b4', 'b5']
        
        for i, (method, _accuracies) in enumerate(ordered_accuracies.items()):
            key = 'all'
            value = _accuracies[key]
            value_for_print = value['overall']*100
            print(method)
            print(value)
            
            if 'per_bucket' in value:
                buckets = list(value['per_bucket'].keys())
                # buckets = [f'$10^{i}$' for i in range(2, 7)]
                scores = list(value['per_bucket'].values())
                # plt.plot(buckets, scores, label=f"{method}: {value_for_print:.2f}", marker='', color=palette(i), linewidth=3)
                plt.plot(buckets, scores, label=f"{method}", marker='o', color=palette(i), linewidth=3)
                plt.xticks(range(0, len(custom_xticks)), custom_xticks, fontsize=18)

        
        # plt.title(f"A) {model_name}", fontdict=title_font)
        plt.yticks([0.3, 0.6, 0.9], fontsize=20)
        # plt.xlabel("Popularity (pageviews)", fontdict=font)
        # plt.ylabel("Accuracy", fontdict=font)
        plt.ylim(0, 1.0)
        
        # plt.legend(ncol=3, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.2))
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # plt.legend(ncol=4, loc='upper center', fontsize=11)
        plt.tight_layout()
        
        # plt.savefig(f"main_{model_name}.pdf", format='pdf', dpi=1600)
        plt.savefig(f"main_{model_name}.png", dpi=1600)
        # plt.savefig(answer_generator_img_save_path, dpi=1600)
        # plt.savefig(answer_generator_img_save_path, format='pdf', dpi=1000)
        plt.show()

def sig_test(file1, file2):
      
    def load_jsonl(file_path):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
      
    model_a_results = load_jsonl(file1)
    model_b_results = load_jsonl(file2)
    assert len(model_a_results) == len(model_b_results)
    
    # Extract is_correct values
    model_a_correct = [entry['is_correct'] for entry in model_a_results]
    model_b_correct = [entry['is_correct'] for entry in model_b_results]

    # Create a contingency table
    a = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a == 0 and b == 0)
    b = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a == 0 and b == 1)
    c = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a == 1 and b == 0)
    d = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a == 1 and b == 1)

    contingency_table = [[a, b], [c, d]]

    # Perform McNemar's test
    result = mcnemar(contingency_table, exact=True)

    print("Statistic:", result.statistic)
    print("p-value:", result.pvalue)

    # Interpretation
    if result.pvalue < 0.05:
        print("The difference in performance is statistically significant.")
    else:
        print("The difference in performance is not statistically significant.")

def wilcoxon_sig_test(file1, file2):
    def load_accuracy_data(file_path):
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        return {item["query_id"]: int(item["is_correct"]) for item in data}

    def calculate_accuracy(labels):
        accuracy = (sum(labels) / len(labels))*100
        return accuracy
        
    accuracy_data_llm1 = load_accuracy_data(file1)
    accuracy_data_llm2 = load_accuracy_data(file2)

    # Ensure the order of results is the same based on query_id
    query_ids = sorted(accuracy_data_llm1.keys())
    accuracy_llm1 = [accuracy_data_llm1[query_id] for query_id in query_ids]
    accuracy_llm2 = [accuracy_data_llm2[query_id] for query_id in query_ids]

    llm1_acc = calculate_accuracy(accuracy_llm1)
    print(f'LLM1 Accuracy: {llm1_acc:.2f}')
    llm2_acc = calculate_accuracy(accuracy_llm2)
    print(f'LLM2 Accuracy: {llm2_acc:.2f}')

    # Perform the Wilcoxon signed-rank test
    stat, p_value = wilcoxon(accuracy_llm1, accuracy_llm2)

    # Output the test statistic and p-value
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")
    
    # Interpretation
    if p_value < 0.01:
        print("The difference in performance is statistically significant.")
    else:
        print("The difference in performance is not statistically significant.")
    
    

def main():
    # == 1) Plot buckets distribution: Number of data per bucket
    # plot_buckets_distribution(only_all=True)
    
    # == 2) Plot Retrival models output: Pre-relation & Pre-buckets
    # plot_retriever_results_per_relation()
    # plot_retriever_results_per_buckets(only_all=True)
    
    # == 3) Plot QA models output
    # plot_answer_generator_results(per_relation=True, per_bucket=False, only_all=False)
    # plot_answer_generator_results(per_relation=False, per_bucket=True, only_all=False)
    plot_answer_generator_results(per_relation=False, per_bucket=False, only_all=True)
    
    # == 4) Significance test
    # file1 = "component0_preprocessing/generated_data/EQ_costomized/results/highlighting_retrievers/EQ_stable_lm2_3p_af_rag_ideal_peft__3g_results.jsonl"
    # file2 = "component0_preprocessing/generated_data/EQ_costomized/results/highlighting_retrievers/EQ_stable_lm2_1rp_3p_bf_rag_ideal_full_3g_results.jsonl"
    # # sig_test(file1, file2)
    # wilcoxon_sig_test(file1, file2)
    

if __name__ == "__main__":
    main()