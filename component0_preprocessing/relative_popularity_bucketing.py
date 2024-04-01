import requests, json, ast, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wikipediaapi
import math

rel_pop_score = {
    '257': 0.6138864810588756,
    '22': 0.4410058888134515,
    '91': 0.16768592403443858,
    '218': 0.1,
    '106': 0.5067754880971169,
    '97': 0.6671048731510615,
    '292': 0.6299331415869965,
    '164': 0.21263254664084807,
    '560': 0.11166066064556371,
    '484': 0.12386409344801165,
    '533': 0.1784324413416965,
    '639': 0.17103813440718924,
    '182': 0.11236930748067743,
    '422': 1.0,
    '526': 0.1453979686228685,
    '472': 0.8016766267589054
}


def split_by_relation(input_file_path, output_file_path):
    # Initialize a dictionary to hold the grouped data
    grouped_data = {}

    # Read the JSONL file and group data by relation_type
    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            data = json.loads(line)
            relation_type = data.get('relation_type')
            if relation_type:
                if relation_type not in grouped_data:
                    grouped_data[relation_type] = []
                grouped_data[relation_type].append(data)

    # Write the grouped data to a JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(grouped_data, output_file, indent=4)
        
    return grouped_data

def calculate_relative_popularity(objects):
    # Extract popularity scores
    popularity_scores = [obj['pageviews'] for obj in objects]
    # popularity_scores = [obj['popqa_pageviews'] for obj in objects]
    # Calculate mean and standard deviation
    mean = np.mean(popularity_scores)
    std_dev = np.std(popularity_scores)
    # Calculate and add relative popularity
    for obj in objects:
        obj['relative_popularity'] = (obj['pageviews'] - mean) / std_dev

    return objects

def split_to_buckets(objects, split_points):
    
    split_points = sorted(split_points)
    sp_len = len(split_points)
    bucket_data = {'bucket{}'.format(idx+1): list() for idx in range(sp_len+1)}
    
    for obj in objects:
        # rp = obj['relative_popularity']
        rel_key = filename.split('.')[0]
        if obj['pageviews'] != 0:
            rp = math.log(obj['pageviews'], 10)
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

def plot_density_rel(json_data):
    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(12, 8))
    fig.delaxes(axes[0,1])  # Remove the first subplot (top-left)
    fig.delaxes(axes[0,2])  # Remove the third subplot (top-right)
    fig.delaxes(axes[0,3])  # Remove the third subplot (top-right)
    fig.delaxes(axes[0,4])  # Remove the third subplot (top-right)
    fig.delaxes(axes[5,4])  # Remove the third subplot (top-right)

    # Plot data for each key
    for idx, key in enumerate(json_data.keys()):
        if idx == 0:
            row = 0
            col = 0
        else:
            row = (idx+3) // 5
            col = (idx+3) % 5
        ax = axes[row, col]

        relative_popularity_scores = [obj['relative_popularity'] for obj in json_data[key]]
        sns.kdeplot(relative_popularity_scores, ax=ax)
        ax.set_title(key)
        ax.set_xlim(-2.5, 5)
        ax.set_ylim(0, 2.5)
        ax.set_ylabel('')
        
        # Calculate and mark the standard deviation
        mean_val = np.mean(relative_popularity_scores)
        std_val = np.std(relative_popularity_scores)
        ax.axvline(mean_val, color='k', linestyle='--')
        ax.axvline(mean_val + std_val, color='r', linestyle='--')
        ax.axvline(mean_val - std_val, color='r', linestyle='--')
        ax.text(mean_val + std_val, 0.1, "{:.1f}".format(std_val), rotation=0, color='r')
        ax.text(mean_val - std_val, 0.1, "{:.1f}".format(-std_val), rotation=0, color='r')


    plt.tight_layout()
    plt.show()

def plot_bucket_num(json_data, dataset_name):

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

    # Plot data for each key
    for idx, key in enumerate(json_data.keys()):
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
        counts = [len(json_data[key][bucket]) for bucket in json_data['all'].keys()]

        ax.bar([i+1 for i in range(len(json_data['all'].keys()))], counts)
        # ax.set_xlabel('log pop')
        ax.set_title(key)

    # plt.title("EQ test-set")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # For popQA
    dataset_name = 'popqa'
    queries_file = "component0_preprocessing/generated_data/popQA_costomized/test/queries.jsonl"
    q_relation_file_path = "component0_preprocessing/generated_data/popQA_costomized/queries_by_relation.json"
    q_relative_pop_file_path = "component0_preprocessing/generated_data/popQA_costomized/queries_relative_pop.json"
    q_buckets_path = "component0_preprocessing/generated_data/popQA_costomized/queries_bucketing.json"
    
    # if not os.path.exists("data/generated/popQA_costomized/query_bucketing"):
    #     os.makedirs("data/generated/popQA_costomized/query_bucketing")
    
    # q_buckets_path = "data/generated/popQA_costomized/query_bucketing"
    
    # For EQ testset
    # dataset_name = 'eq'
    # # queries_file = "component0_preprocessing/generated_data/EntityQuestions_costomized/test/queries.jsonl"
    # # q_relation_file_path = "component0_preprocessing/generated_data/EntityQuestions_costomized/test/queries_by_relation.json"
    # # q_relative_pop_file_path = "component0_preprocessing/generated_data/EntityQuestions_costomized/test/queries_relative_pop.json"
    # # q_buckets_path = "component0_preprocessing/generated_data/EntityQuestions_costomized/test/queries_buckets.json"
    
    # For EQ devset
    # queries_file = "component0_preprocessing/generated_data/EntityQuestions_costomized/dev/queries.jsonl"
    # q_relation_file_path = "component0_preprocessing/generated_data/EntityQuestions_costomized/dev/queries_by_relation.json"
    # q_relative_pop_file_path = "component0_preprocessing/generated_data/EntityQuestions_costomized/dev/queries_relative_pop.json"
    # q_buckets_path = "component0_preprocessing/generated_data/EntityQuestions_costomized/dev/queries_buckets.json"
    
    # Convert all queries to list of objs
    with open(queries_file, 'r') as file:
        q_all = [json.loads(line) for line in file]
    q_by_relation = split_by_relation(queries_file, q_relation_file_path)
    
    # Add relative popularity to objs
    q_relative = {}
    all_new_objs = calculate_relative_popularity(q_all)
    q_relative['all'] = all_new_objs
    for relation, objects in q_by_relation.items():
        new_objs = calculate_relative_popularity(objects)
        q_relative[relation] = new_objs
    
    with open(q_relative_pop_file_path, 'w') as relative_output_file:   
        json.dump(q_relative, relative_output_file, indent=4)
    
    # with open(q_relative_pop_file_path, 'r') as file:
    #     q_relatives = json.load(file)
    #     plot_density_rel(q_relatives)
    
    
    ## Split each list to three buckets
    split_points = [-0.5, -0.3, -0.15, -0.05, 0, 0.75]
    split_points = [-0.30, -0.15, -0.05, 0.05]
    split_points = [2, 3, 4, 5] # Good for popqa_pageviews
    split_points = [3, 4, 5, 6] # Good for my pageviews
    
    with open(q_relative_pop_file_path, 'r') as file:
        q_by_relative = json.load(file)
    
    q_buckets = {}
    for relation, objects in q_by_relative.items():
        new_objs = split_to_buckets(objects, split_points)
        q_buckets[relation] = new_objs
    
    with open(q_buckets_path, 'w') as bk_output_file:
        json.dump(q_buckets, bk_output_file, indent=4)

    with open(q_buckets_path, 'r') as file:
        q_buckets = json.load(file)
        plot_bucket_num(q_buckets, dataset_name)

    ### get corpus for unpopular ones     
    # corpus_path = "data/generated/popQA_costomized/corpus_unpop"
    # qrels_path = "data/generated/popQA_costomized/qrels_unpop"
    
    # create_corpus_qrels_files_bucket(q_buckets_path, corpus_path, qrels_path)
        

    
    
    