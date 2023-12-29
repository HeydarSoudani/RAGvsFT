import requests, json, ast, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wikipediaapi

# Function to categorize an object into a bucket based on relative popularity
def categorize_by_popularity(obj):
    rp = obj['relative_popularity']
    if rp < -1:
        return 1  # Bucket 1
    elif -1 <= rp <= 1:
        return 2  # Bucket 2
    else:
        return 3  # Bucket 3

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

def split_to_buckets(objects, sp1, sp2, sp3):
    bucket_data = {}
    bucket_data['bucket1'] = []
    bucket_data['bucket2'] = []
    bucket_data['bucket3'] = []
    bucket_data['bucket4'] = []
    
    for obj in objects:
        rp = obj['relative_popularity']
        if rp < sp1:
            bucket_data['bucket1'].append(obj)
        elif sp1 <= rp < sp2:
            bucket_data['bucket2'].append(obj)
        elif sp2 <= rp < sp3:
            bucket_data['bucket3'].append(obj)
        else:
            bucket_data['bucket4'].append(obj)
    
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

def plot_bucket_num(json_data):

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

        # Count the number of elements in each bucket
        counts = [len(json_data[key][bucket]) for bucket in ['bucket1', 'bucket2', 'bucket3', 'bucket4']]

        ax.bar(['b1', 'b2', 'b3', 'b4'], counts)
        ax.set_title(key)

    plt.tight_layout()
    plt.show()

def get_wikipedia_title_from_wikidata(wikidata_id):
    wikidata_url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={wikidata_id}&format=json&props=sitelinks"
    response = requests.get(wikidata_url)
    data = response.json()

    entities = data.get('entities', {})
    if wikidata_id in entities:
        wikipedia_title = entities[wikidata_id].get('sitelinks', {}).get('enwiki', {}).get('title')
        return wikipedia_title
    return None

def get_wikipedia_summary_and_paragraphs(title):
    inappropriate_sections = [
        'References', 'Resources', 'Sources', 'Notes', 'External links', 'See also', 'Further reading',
        'Gallery', 'Ranks', 'Awards', 'Awards and nominations', 'Television', 'Covers and tributes',
        'Filmography', 'Discography', 'Selected discography', 'Bibliography', 'Songs', 'cast', 'Accolades',
        'Selected compositions',
        'Historic population', 'Family tree', 'Table'
    ]
    wiki_wiki = wikipediaapi.Wikipedia('my_project', 'en')
    page = wiki_wiki.page(title)

    if not page.exists():
        return "Page does not exist", []

    summary = page.summary
    paragraphs = [section.text for section in page.sections if section.title not in inappropriate_sections]
    paragraphs = [item for item in paragraphs if item != ""]

    return summary, paragraphs

def create_corpus_qrels_files_bucket(q_buckets_path, corpus_path, qrels_path):
    with open(q_buckets_path, 'r') as file:
        q_buckets = json.load(file)
        
    corpus_id_counter = 1
    add = 0
    for key, value in q_buckets.items():
        selected_queries = value['bucket2']
        
        print(len(selected_queries))
        add += len(selected_queries)
    print(add)
        # corpus_file = os.path.join(corpus_path, f"{key}.jsonl")
        # qrels_file = os.path.join(qrels_path, f"{key}.jsonl")
        # with open(corpus_file, 'w') as corpus, open(qrels_file, 'w') as qrels:
        #     for idx, data in enumerate(selected_queries):
        #         pass
                # if idx == 10:
                #     break
                
                # # data = json.loads(line.strip())
                # wikidata_id = data['entity_id']
                # wikipedia_title = data['wiki_title']
                # summary, paragraphs = get_wikipedia_summary_and_paragraphs(wikipedia_title)
                # corpus_data1 = {
                #     'id': str(corpus_id_counter),
                #     'contents': summary
                # }
                # qrels_data1 = {
                #     'query_id': wikidata_id,
                #     'doc_id': str(corpus_id_counter),
                #     'score': 1
                # }
                # corpus_id_counter += 1
                
                # corpus_jsonl_line = json.dumps(corpus_data1)
                # corpus.write(corpus_jsonl_line + '\n')
                # qrels_jsonl_line = json.dumps(qrels_data1)
                # qrels.write(qrels_jsonl_line + '\n')
                
                # # Write paragraphs in files
                # for paragraph in paragraphs:
                #     corpus_data2 = {
                #         'id': str(corpus_id_counter),
                #         'contents': paragraph
                #     }
                #     qrels_data2 = {
                #         'query_id': wikidata_id,
                #         'doc_id': str(corpus_id_counter),
                #         'score': 0
                #     }
                #     corpus_id_counter += 1
                    
                #     corpus_jsonl_line = json.dumps(corpus_data2)
                #     corpus.write(corpus_jsonl_line + '\n')
                #     qrels_jsonl_line = json.dumps(qrels_data2)
                #     qrels.write(qrels_jsonl_line + '\n')


if __name__ == "__main__":
    # For popQA
    # queries_file = "data/generated/popQA_costomized/queries.jsonl"
    # q_relation_file_path = "data/generated/popQA_costomized/queries_by_relation.json"
    # q_relative_pop_file_path = "data/generated/popQA_costomized/queries_relative_pop.json"
    # q_buckets_path = "data/generated/popQA_costomized/queries_buckets.json"
    
    # For EQ testset
    # queries_file = "data/generated/EntityQuestions_costomized/test/queries.jsonl"
    # q_relation_file_path = "data/generated/EntityQuestions_costomized/test/queries_by_relation.json"
    # q_relative_pop_file_path = "data/generated/EntityQuestions_costomized/test/queries_relative_pop.json"
    # q_buckets_path = "data/generated/EntityQuestions_costomized/test/queries_buckets.json"
    
    # For EQ devset
    queries_file = "data/generated/EntityQuestions_costomized/dev/queries.jsonl"
    q_relation_file_path = "data/generated/EntityQuestions_costomized/dev/queries_by_relation.json"
    q_relative_pop_file_path = "data/generated/EntityQuestions_costomized/dev/queries_relative_pop.json"
    q_buckets_path = "data/generated/EntityQuestions_costomized/dev/queries_buckets.json"
    
    # Convert all queries to list of objs
    # with open(queries_file, 'r') as file:
    #     q_all = [json.loads(line) for line in file]
    # q_by_relation = split_by_relation(queries_file, q_relation_file_path)
    
    # # Add relative popularity to objs
    # q_relative = {}
    # all_new_objs = calculate_relative_popularity(q_all)
    # q_relative['all'] = all_new_objs
    # for relation, objects in q_by_relation.items():
    #     new_objs = calculate_relative_popularity(objects)
    #     q_relative[relation] = new_objs
    
    # with open(q_relative_pop_file_path, 'w') as relative_output_file:   
    #     json.dump(q_relative, relative_output_file, indent=4)
    
    # with open(q_relative_pop_file_path, 'r') as file:
    #     q_relatives = json.load(file)
    #     plot_density_rel(q_relatives)
    
    
    
    ## Split each list to three buckets
    split_point1 = -1.0
    split_point2 = 0.0
    split_point3 = 1.0
    
    with open(q_relative_pop_file_path, 'r') as file:
        q_by_relative = json.load(file)
    
    q_buckets = {}
    for relation, objects in q_by_relative.items():
        new_objs = split_to_buckets(objects, split_point1, split_point2, split_point3)
        q_buckets[relation] = new_objs
    
    with open(q_buckets_path, 'w') as bk_output_file:
        json.dump(q_buckets, bk_output_file, indent=4)

    with open(q_buckets_path, 'r') as file:
        q_buckets = json.load(file)
        plot_bucket_num(q_buckets)


    
    ### get corpus for unpopular ones     
    # corpus_path = "data/generated/popQA_costomized/corpus_unpop"
    # qrels_path = "data/generated/popQA_costomized/qrels_unpop"
    
    # create_corpus_qrels_files_bucket(q_buckets_path, corpus_path, qrels_path)
        

    
    
    