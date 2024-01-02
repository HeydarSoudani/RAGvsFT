import matplotlib.pyplot as plt
import csv

relations = [
    'all',
    'occupation', 'mother', 'religion', 'place of birth',
    'genre', 'father', 'screenwriter', 'director',
    'producer', 'author', 'composer', 'country',
    'capital', 'capital of', 'color', 'sport'
]

if __name__ == "__main__":
    
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 8))
    fig.delaxes(axes[0,1])  # Remove the first subplot (top-left)
    fig.delaxes(axes[0,2])  # Remove the third subplot (top-right)
    fig.delaxes(axes[0,3])  # Remove the third subplot (top-right)
    
    # file_path = 'results/final_results.tsv'
    # file_path =  'component1_retrieval/results/dpr_beir.tsv'
    # file_path = 'component1_retrieval/results/msmarco-distilbert-base-v3_dpr_beir.tsv'
    file_path = 'component1_retrieval/results/ft_msmarco-distilbert-base-v3_dpr_beir.tsv'
        
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
                        ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
                        [float(item) for item in sorted_data.values()],
                        label=f'Recall@{i}'
                    )
                else:
                    ax.plot(
                        ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
                        [float(item) for item in sorted_data.values()]
                    )
            
            ax.set_title(relation)
            ax.set_ylim(0, 1.05)
    
    # axes[0,2].legend(loc="upper right")
    fig.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
            
            