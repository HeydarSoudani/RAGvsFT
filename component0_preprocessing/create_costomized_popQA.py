from typing import List

import os, random
import math
import requests, json, ast
import urllib.request as urllib2
from urllib.parse import quote
import wikipediaapi
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.tokenize import sent_tokenize

def convert_to_url_format(text):
    # Use urllib.parse.quote to convert text to URL format
    url_formatted_text = quote(text)
    return url_formatted_text

def get_pageviews(wiki_title):
    TOP_API_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.{project}/all-access/all-agents/{topic}/monthly/{date_from}/{date_to}"
    lang = 'en'
    project = 'wikipedia'
    date_from = "2023010100"
    date_to = "2023121400"
    all_views = 0
    
    edited_title = convert_to_url_format(wiki_title.replace(" ", "_"))
    url = TOP_API_URL.format(lang=lang,
                            project = project,
                            topic = edited_title,
                            date_from = date_from,
                            date_to = date_to)
    try:
        resp = urllib2.urlopen(url)
        resp_bytes = resp.read()
        data = json.loads(resp_bytes)
        all_views = sum([item['views'] for item in data['items']]) 
        # print("Target: {:<15}, Views: {}".format(edited_title, all_views))
    except urllib2.HTTPError as e:
        # print(e.code)
        print("Target: {:<20}, does not have a wikipedia page".format(edited_title))
    except urllib2.URLError as e:
        print(e.args)
    
    return all_views

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
        'Selected compositions', 'Select bibliography', 'Historic population', 'Family tree', 'Table',
        'Selected works', 'Quotes', 'Literary awards', 'Select critical works',
        'Interview', 'Publications', 'Works', 'Books', 'Films', 'Broadway', 'Off-Broadway and Regional',
        'Family', 'Tours', 'Videography', 'Authored books', 'Awards and recognitions', 'Published works',
        'Collaborations and guest appearances', 'Sources and external links', 'List of works'
    ]
    wiki_wiki = wikipediaapi.Wikipedia('my_project', 'en')
    page = wiki_wiki.page(title)

    if not page.exists():
        return "Page does not exist", []

    summary = page.summary
    paragraphs = [section.text for section in page.sections if section.title not in inappropriate_sections]
    paragraphs = [item for item in paragraphs if item != ""]

    return summary, paragraphs

def create_queries_file(tsv_file, jsonl_file, relation="all"):
    
    df = pd.read_csv(tsv_file, sep='\t')
    if relation != "all":
        filtered_df = df[df['prop'] == relation]
    df = filtered_df
    
    possible_answers = [ast.literal_eval(item) for item in df['possible_answers']]
    question = list(df['question'])
    relation_type = list(df['prop'])
    entity_id = [item.split('/')[-1] for item in df['s_uri']]
    
    directory = os.path.dirname(jsonl_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(jsonl_file, 'w') as jsonl:
        query_id_counter = 1
        for idx, item in enumerate(entity_id):
            wiki_title = get_wikipedia_title_from_wikidata(item)
            
            if wiki_title != None:
                print('Id: {}, wikiID: {}, title: {}'.format(idx, item, wiki_title))
                pageviews = get_pageviews(wiki_title)
                temp = {
                    "query_id": "Q_"+str(query_id_counter),
                    "question": question[idx],
                    "possible_answers": possible_answers[idx],
                    "pageviews": pageviews,
                    "entity_id": item,
                    'relation_type': relation_type[idx], 
                }
                query_id_counter += 1
                jsonl.write(json.dumps(temp) + '\n')
    
    print("All queries are processed ...")

def create_corpus_qrels_files(queries_file, corpus_file, qrels_file):
    print("Start processing corpus ...")
    
    directory = os.path.dirname(corpus_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(queries_file, 'r') as queries, open(corpus_file, 'w') as corpus, open(qrels_file, 'w') as qrels:
        corpus_id_counter = 1
        for idx, line in enumerate(queries):
            if (idx+1)%100 == 0:
                print("# processed queries:", idx+1)
            # if idx == 400:
            #     break
            
            data = json.loads(line.strip())
            query_id = data['query_id']
            wikidata_id = data['entity_id']
            wikipedia_title = get_wikipedia_title_from_wikidata(wikidata_id)
            
            if wikipedia_title:
                summary, paragraphs = get_wikipedia_summary_and_paragraphs(wikipedia_title)
                
                # Write summary in files
                # corpus_id = str(uuid.uuid4())
                corpus_data1 = {
                    'id': str(corpus_id_counter),
                    'contents': summary
                    # 'corpus_id': corpus_id,
                    # 'title': wikipedia_title,
                    # 'entityID': wikidata_id,
                    # 'text': summary,
                    # 'has_answer': True
                }
                qrels_data1 = {
                    'query_id': query_id,
                    'doc_id': str(corpus_id_counter),
                    'score': 1
                }
                corpus_id_counter += 1
                
                corpus_jsonl_line = json.dumps(corpus_data1)
                corpus.write(corpus_jsonl_line + '\n')
                qrels_jsonl_line = json.dumps(qrels_data1)
                qrels.write(qrels_jsonl_line + '\n')
                
                
                # Write paragraphs in files
                for paragraph in paragraphs:
                    # corpus_id = str(uuid.uuid4())
                    corpus_data2 = {
                        'id': str(corpus_id_counter),
                        'contents': paragraph
                        # 'corpus_id': corpus_id,
                        # 'title': wikipedia_title,
                        # 'entityID': wikidata_id,
                        # 'text': paragraph,
                        # 'has_answer': False
                    }
                    qrels_data2 = {
                        'query_id': query_id,
                        'doc_id': str(corpus_id_counter),
                        'score': 0
                    }
                    corpus_id_counter += 1
                    
                    corpus_jsonl_line = json.dumps(corpus_data2)
                    corpus.write(corpus_jsonl_line + '\n')
                    qrels_jsonl_line = json.dumps(qrels_data2)
                    qrels.write(qrels_jsonl_line + '\n')
    
    print("All corpuses are processed!!!")

def queries_downsampling(input_file, output_file, percentage_to_keep):
    
    with open(input_file, 'r') as file:
        queries = [json.loads(line) for line in file]
        
    num_to_keep = int(len(queries) * (percentage_to_keep / 100))
    
    downsampled_file_path = 'component0_preprocessing/generated_data/popQA_religion/queries_{}ds.jsonl'.format(percentage_to_keep)
    downsampled_queries = random.sample(queries, num_to_keep)
    
    with open(output_file, 'w') as file:
        for query in downsampled_queries:
            file.write(json.dumps(query) + '\n')
    print(f"Downsampled queries saved to: {output_file}")

def queries_bucketing(input_file, output_file=None, split_points=[1,2,3,4]):
    
    with open(input_file, 'r') as file:
        objects = [json.loads(line) for line in file]
    
    split_points = sorted(split_points)
    sp_len = len(split_points)
    
    bucket_data = {'bucket{}'.format(idx+1): list() for idx in range(sp_len+1)}
    for obj in objects:
        # rp = obj['relative_popularity']
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
        
        if output_file != None:
            with open(output_file, 'w') as bk_output_file:
                json.dump(bucket_data, bk_output_file, indent=4)

def plot_bucket_num(input_file, relation_name):

    with open(input_file, 'r') as file:
        objects = json.load(file)
    counts = [len(objects[bucket]) for bucket in objects.keys()]

    plt.bar([i+1 for i in range(len(objects.keys()))], counts)

    plt.title(relation_name)
    plt.tight_layout()
    plt.show()
  
def count_tokens(sentence: str) -> int:
    return len(sentence.split())

def split_text_to_sentences(text: str, max_tokens: int) -> List[str]:
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Adding a period back to each sentence except the last one
        sentence = sentence.strip() if sentence != sentences[-1] else sentence.strip()
        sentence_length = len(sentence.split())

        # Check if adding the current sentence would exceed the maximum token count
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Add the current chunk to the chunks list and start a new chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def corpus_cleaning(corpus_in_file, corpus_out_file, qrels_in_file, qrels_out_file, token_num):
    
    ### === Step 1: remove parentheses
    corpus_wo_parentheses = []
    with open(corpus_in_file, 'r') as in_file:
        for idx, line in enumerate(in_file):
            text = json.loads(line)
            cleaned_text = re.sub(r'\([^)]*\)\s*', '', text['contents'])
            
            cleaned_obj = {"id": text["id"], "contents": cleaned_text }
            corpus_wo_parentheses.append(cleaned_obj)

    ### === Step 2: Split in some chunks
    original_qrels = []
    with open(qrels_in_file, 'r') as file:
        for line in file:
            original_qrels.append(json.loads(line))
    
    with open(corpus_out_file, 'w') as c_out_file, open(qrels_out_file, 'w') as new_qrel_file:
        for idx, line in enumerate(corpus_wo_parentheses): 
            doc_id = line["id"]
            query_id = None
            score = None 
            for qrel in original_qrels:
                if qrel['doc_id'] == doc_id:
                    query_id, score = qrel['query_id'], qrel['score']
            
            split_text = split_text_to_sentences(line["contents"], token_num)
            for i, split in enumerate(split_text):
                new_doc_id = doc_id+'-'+str(i)
                sp_obj = {"id": new_doc_id, "contents": split}
                c_out_file.write(json.dumps(sp_obj) + "\n")
                
                qsp_obj = {"query_id": query_id, "doc_id": new_doc_id, "score": score}
                new_qrel_file.write(json.dumps(qsp_obj) + "\n") 
            

if __name__ == "__main__":
    
    popQA_input_file = "data/dataset/popQA/popQA.tsv"
    
    # For whole popQA dataset
    # queries_file = "data/generated/popQA_costomized/queries.jsonl"
    # corpus_file = "data/generated/popQA_costomized/corpus.jsonl"
    # qrels_file = "data/generated/popQA_costomized/qrels.jsonl"
    
    # Only for occupation relation
    # queries_file = 'component0_preprocessing/generated_data/popQA_occupation/queries.jsonl'
    # corpus_file = 'component0_preprocessing/generated_data/popQA_occupation/corpus.jsonl'
    # qrels_file = 'component0_preprocessing/generated_data/popQA_occupation/qrels.jsonl'
    
    
    ### === Only for religion relation ===========
    dataset_name = 'popqa'
    selected_relation = 'religion'
    
    queries_file = 'component0_preprocessing/generated_data/popQA_religion/queries.jsonl'
    # create_queries_file(popQA_input_file, queries_file, relation=selected_relation)
    
    percentage_to_keep = 30
    queries_ds_file = 'component0_preprocessing/generated_data/popQA_religion/queries_{}ds.jsonl'.format(str(percentage_to_keep))
    # queries_downsampling(queries_file, queries_ds_file, percentage_to_keep)
    
    corpus_file = 'component0_preprocessing/generated_data/popQA_religion/corpus_{}ds.jsonl'.format(str(percentage_to_keep))
    qrels_file = 'component0_preprocessing/generated_data/popQA_religion/qrels_{}ds.jsonl'.format(str(percentage_to_keep))
    # create_corpus_qrels_files(queries_ds_file, corpus_file, qrels_file)

    split_points = [3, 4, 5, 6]
    queries_ds_bk_file = 'component0_preprocessing/generated_data/popQA_religion/queries_{}ds_bk.jsonl'.format(str(percentage_to_keep))
    
    # queries_bucketing(queries_ds_file, queries_ds_bk_file, split_points)
    # plot_bucket_num(queries_ds_bk_file, selected_relation)
    
    queries_bk_file = 'component0_preprocessing/generated_data/popQA_religion/queries_bk.jsonl'
    # queries_bucketing(queries_file, queries_bk_file, split_points)
    # plot_bucket_num(queries_bk_file, selected_relation)
    # if os.path.exists(queries_bk_file):
    #     os.remove(queries_bk_file)

    token_num = 512
    corpus_tk_file = 'component0_preprocessing/generated_data/popQA_religion/corpus_{}ds_{}tk.jsonl'.format(str(percentage_to_keep), str(token_num))
    qrels_tk_file = 'component0_preprocessing/generated_data/popQA_religion/qrels_{}ds_{}tk.jsonl'.format(str(percentage_to_keep), str(token_num))
    corpus_cleaning(corpus_file, corpus_tk_file, qrels_file, qrels_tk_file, token_num)
    
    
    


