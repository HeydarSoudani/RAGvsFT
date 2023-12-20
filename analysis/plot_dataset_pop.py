import json
import matplotlib.pyplot as plt
import seaborn as sns
import math

def get_logs(input_list, base=10):
    # Calculate logarithm for each element in the list
    logs = [math.log(x, base) for x in input_list]
    return logs

if __name__ == "__main__":
    
    jsonl_files = {
        "PopQA": "data/generated/popQA_costomized/queries.jsonl",
        "EQ_train": "data/generated/EntityQuestions_costomized/train/queries.jsonl",
        "EQ_dev": "data/generated/EntityQuestions_costomized/dev/queries.jsonl",
        "EQ_test": "data/generated/EntityQuestions_costomized/test/queries.jsonl"
    }
    edge_color = ['green', 'blue', 'orange']
        
    plt.figure(figsize=(10, 6))

    # Read and plot data from each file
    for idx, (dataset_name, file) in enumerate(jsonl_files.items()):
        pageviews = []

        # Read each file and extract pageviews
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)
                pageview = data.get('pageviews', 0)
                log_pageview = math.log(pageview, 10) if pageview != 0 else 0
                
                pageviews.append(log_pageview)

        # Plot distribution for each file
        # plt.hist(pageviews, bins=100, edgecolor=edge_color[idx], alpha=0.4, label=f'{dataset_name}')
        sns.kdeplot(pageviews, label=f'{dataset_name}', fill=False)
        # plt.hist(pageviews, bins=50, alpha=alpha, color=colors[i % len(colors)], label=f'{file}')

    plt.xlabel('Pageviews')
    plt.ylabel('Frequency')
    plt.title('Combined Pageview Distributions')
    plt.legend()
    plt.show()