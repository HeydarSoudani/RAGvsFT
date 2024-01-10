import matplotlib.pyplot as plt
import csv, os

relations = [
    'all',
    'occupation', 'mother', 'religion', 'place of birth',
    'genre', 'father', 'screenwriter', 'director',
    'producer', 'author', 'composer', 'country',
    'capital', 'capital of', 'color', 'sport'
]

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

if __name__ == "__main__":
    # retrieval_results_nobk()
    # retrieval_results_bk()
    
    icl_obqa_results_bk()
    
    
    
    
                
            