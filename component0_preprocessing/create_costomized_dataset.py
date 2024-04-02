#!/usr/bin/env python3

import csv
import json
import os
import ast
import re
import torch
import math
import logging
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import urllib.request as urllib2
from urllib.parse import quote
from lmqg import TransformersQG
from lmqg.exceptions import AnswerNotFoundError, ExceedMaxLengthError
from nltk.tokenize import sent_tokenize
# from datasets import load_dataset
import numpy as np
import requests
import wikipediaapi
from transformers import pipeline

import nltk
nltk.download('punkt')

### === Constants =====================  
# PopQA
dataset_name = 'popqa'
tsv_file_path = "data/dataset/popQA/popQA.tsv"
output_dir = 'component0_preprocessing/generated_data/popQA_costomized'
relation_ids = ["22", "218", "91", "257", "182", "164", "526", "97", "533", "639", "472", "106", "560", "484", "292", "422"]
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

# WitQA
dataset_name = 'witqa'
tsv_file_path = "data/dataset/WitQA/witqa.tsv"
output_dir = 'component0_preprocessing/generated_data/witQA_costomized'
relation_ids = ['58', '2936', '17', '1376', '674', '50', '22', '1431', '123', '1038', '3301', '140', '36', '136', '86', '69', '1433', '4647', '25', '452', '2012', '106', '27', '184', '1050', '162', '149', '641', '57', '462', '344', '19']
RELATIONS = {
    "58": "screenwriter",
    "2936": "language used",
    "17": "country",
    "1376": "capital of",
    "674": "characters",
    "50": "author",
    "22": "father",
    "1431": "executive producer",
    "123": "publisher",
    "1038": "relative",
    "3301": "broadcast by",
    "140": "religion",
    "36": "capital",
    "136": "genre",
    "86": "composer",
    "69": "educated at",
    "1433": "published in",
    "4647": "location of first performance",
    "25": "mother",
    "452": "industry",
    "2012": "cuisine",
    "106": "occupation",
    "27": "country of citizenship",
    "184": "doctoral advisor",
    "1050": "medical condition",
    "162": "producer",
    "149": "architectural style",
    "641": "sport",
    "57": "director",
    "462": "color",
    "344": "director of photography",
    "19": "place of birth"
}

# EQ
dataset_name = 'eq'
sub_type = 'test'
data_evidence_dir = "data/dataset/entity_questions_dataset/data_evidence"
output_dir = 'component0_preprocessing/generated_data/EQ_costomized'
relation_ids = ['17', '19', '20', '26', '36', '40', '50', '69', '106', '112', '127', '131', '136', '159', '170', '175', '176', '264', '276', '407', '413', '495', '740', '800']
RELATIONS = {
  "36": "capital",
  "407": "language written in",
  "26": "spouse",
  "159": "headquarters",
  "276": "location",
  "40": "child",
  "176": "producer company",
  "20": "death place",
  "112": "founder",
  "127": "owner",
  "19": "birth place",
  "740": "founding place",
  "413": "fame reason",
  "800": "position played",
  "69": "education place",
  "50": "author",
  "170": "creator",
  "106": "occupation",
  "131": "location",
  "17": "country located in",
  "175": "performer",
  "136": "music genre",
  "264": "music label",
  "495": "creation country"
}

# === Output Directories =====================
entities_analysis_file = f"{output_dir}/entities_analysis.json"
# Step 1
test_dir = f"{output_dir}/test" 
entity_dir = f"{output_dir}/entity" 
os.makedirs(test_dir, exist_ok=True)
os.makedirs(entity_dir, exist_ok=True)

# Step 2
corpus_sum_dir = f"{output_dir}/corpus_summary" 
qrels_sum_dir = f"{output_dir}/qrels_summary" 
corpus_all_dir = f"{output_dir}/corpus_all" 
qrels_all_dir = f"{output_dir}/qrels_all"

os.makedirs(corpus_sum_dir, exist_ok=True)
os.makedirs(qrels_sum_dir, exist_ok=True)
os.makedirs(corpus_all_dir, exist_ok=True)
os.makedirs(qrels_all_dir, exist_ok=True)

# Step 3
# === 3.1) QA Generation by T5-basd model
pl_train_dir = f"{output_dir}/pipeline/train" 
pl_dev_dir = f"{output_dir}/pipeline/dev" 
pl_qrels_train_dir = f"{output_dir}/pipeline/qrels-train" 

os.makedirs(f"{output_dir}/pipeline", exist_ok=True)
os.makedirs(pl_train_dir, exist_ok=True)
os.makedirs(pl_dev_dir, exist_ok=True)
os.makedirs(pl_qrels_train_dir, exist_ok=True)

# === 3.2) QA Generation by prompting Zephyr
pr_train_dir = f"{output_dir}/prompting/train" 
pr_dev_dir = f"{output_dir}/prompting/dev" 
pr_qrels_train_dir = f"{output_dir}/prompting/qrels-train" 

os.makedirs(f"{output_dir}/prompting", exist_ok=True)
os.makedirs(pr_train_dir, exist_ok=True)
os.makedirs(pr_dev_dir, exist_ok=True)
os.makedirs(pr_qrels_train_dir, exist_ok=True)

num_entities_per_relation = 20
dev_split = 0.1
max_tokens = 512

def extract_json_objects(text):
    pattern = r'\{[^{}]*\}'
    json_strings = re.findall(pattern, text)
    
    json_objects = []
    for json_str in json_strings:
        try:
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    
    return json_objects

def get_wikipedia_title_from_wikidata(wikidata_id):
    wikidata_url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={wikidata_id}&format=json&props=sitelinks"
    response = requests.get(wikidata_url)
    data = response.json()

    entities = data.get('entities', {})
    if wikidata_id in entities:
        wikipedia_title = entities[wikidata_id].get('sitelinks', {}).get('enwiki', {}).get('title')
        return wikipedia_title
    return None

def convert_to_url_format(text):
    text = text.replace(" ", "_")
    text = quote(text)
    text = text.replace("/", "%2F")
    return text

def get_pageviews(wiki_title):
    TOP_API_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.{project}/all-access/all-agents/{topic}/monthly/{date_from}/{date_to}"
    lang = 'en'
    project = 'wikipedia'
    date_from = "2022010100"
    date_to = "2022123000"
    all_views = 0
    
    edited_title = convert_to_url_format(wiki_title)
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

def remove_parentheses(input_text):
    cleaned_text = re.sub(r'\([^)]*\)\s*', '', input_text)
    return cleaned_text

def remove_infobox(input_text):
    infobox_pattern = re.compile(r'\{\{Infobox.*?\}\}', re.DOTALL)
    cleaned_text = re.sub(infobox_pattern, '', input_text)
    return cleaned_text

def extract_summary(text):
    text = remove_infobox(text)
    text = remove_parentheses(text)
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines if line != '']
    summary_lines = []
    
    for line in lines:
#         if not line or not line.endswith('.'):
        if len(line)>0 and line[-1].isalpha():
            break 
        else:
            summary_lines.append(line)
            
    summary = ' '.join(summary_lines)
    return summary

def count_tokens(sentence):
    return len(sentence.split())

def split_text_to_sentences(text, max_tokens):
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

def generate_qa_with_memory_handling(model, chunk, max_tokens):
    try:
        with torch.no_grad():
            return model.generate_qa(chunk), None
    except torch.cuda.OutOfMemoryError:
        print("Cuda error")
        torch.cuda.empty_cache()
        if max_tokens <= 1:  
            return [], "Text too small to split further"
        
        # Halve the token limit and split the chunk
        new_max_tokens = max_tokens // 2
        return None, split_text_to_sentences(chunk, new_max_tokens)

def create_test_and_entity_files():
    data_by_prop_id = {}
    with open(tsv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for idx, row in enumerate(reader):
            
            if idx != 0 and idx % 500 == 0:
                print(f"{idx} data are processed...")
            
            if dataset_name == 'popqa':
                prop_id = row['prop_id']
            elif dataset_name == 'witqa':
                prop_id = re.findall(r'\d+', row['predicate'])[0]

            if prop_id not in data_by_prop_id:
                data_by_prop_id[prop_id] = {'test': [], 'entity': [], 'counter': 0}

            query_id = f"{prop_id}_{data_by_prop_id[prop_id]['counter']}"
            data_by_prop_id[prop_id]['counter'] += 1

            # Append data to test and entity lists
            if dataset_name == 'popqa':
                data_by_prop_id[prop_id]['test'].append({
                    'query_id': query_id,
                    'question': row['question'],
                    'answers': ast.literal_eval(row['possible_answers']),
                    'pageviews': row['s_pop']
                })

                data_by_prop_id[prop_id]['entity'].append({
                    'query_id': query_id,
                    'wiki_title': row['s_wiki_title'],
                    'entity_id': row['s_uri'].split('/')[-1]
                })

            elif dataset_name == 'witqa':
                wikidata_id = row["subject"]
                try: 
                    wikipedia_title = get_wikipedia_title_from_wikidata(wikidata_id)
                    if wikipedia_title == None:
                        print(f"Return None for: {wikidata_id}")  
                except:
                    print(f"Error for: {wikidata_id}") 

                if wikipedia_title != None:
                    data_by_prop_id[prop_id]['test'].append({
                        'query_id': query_id,
                        'question': row['output_question'],
                        'answers': ast.literal_eval(row['expanded_object_label']),
                        'pageviews': row['s_pop']
                    })
                    data_by_prop_id[prop_id]['entity'].append({
                        'query_id': query_id,
                        'wiki_title': wikipedia_title,
                        'entity_id': wikidata_id
                    })

    for prop_id, content in data_by_prop_id.items():
        with open(f'{test_dir}/{prop_id}.test.json', 'w') as test_file:
            json.dump(content['test'], test_file, indent=4)

        with open(f'{entity_dir}/{prop_id}.entity.json', 'w') as entity_file:
            json.dump(content['entity'], entity_file, indent=4)

    print("Test and Entity creation complete.")

def create_test_and_entity_files_EQ():
    data_by_prop_id = {}
    for relation_dir in os.listdir(data_evidence_dir):
        prop_id = re.findall(r'\d+', relation_dir)[0]
        if prop_id not in data_by_prop_id:
            data_by_prop_id[prop_id] = {'test': [], 'entity': [], 'counter': 0}

        print("Processing relation: ", prop_id)
        file_path = f"{data_evidence_dir}/{relation_dir}/{relation_dir}.{sub_type}.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            for idx, item in enumerate(data):
    #             if idx == 10:
    #                 break
                # Get wikipedia title from wikidata
                query_id = f"{prop_id}_{data_by_prop_id[prop_id]['counter']}"
                data_by_prop_id[prop_id]['counter'] += 1

                wikidata_id = item["evidence"][0]["subject"]["uri"]
                try:             
                    wikipedia_title = get_wikipedia_title_from_wikidata(wikidata_id)
                    if wikipedia_title == None:
                        print(f"Return None for: {wikidata_id}")   
                    else:
                        pageviews = get_pageviews(wikipedia_title)
                        if pageviews == None:
                            print(f"Return pageviews None for: {wikidata_id}")   
                except:
                    print(f"Error for: {wikidata_id}")

                if wikipedia_title != None and pageviews != None:
                    data_by_prop_id[prop_id]['test'].append({
                        'query_id': query_id,
                        'question': item['question'],
                        'answers': item['answers'],
                        'pageviews': pageviews
                    })
                    data_by_prop_id[prop_id]['entity'].append({
                        'query_id': query_id,
                        'wiki_title': wikipedia_title,
                        'entity_id': wikidata_id
                    })
        
    for prop_id, content in data_by_prop_id.items():
        with open(f'{test_dir}/{prop_id}.test.json', 'w') as test_file:
            json.dump(content['test'], test_file, indent=4)

        with open(f'{entity_dir}/{prop_id}.entity.json', 'w') as entity_file:
            json.dump(content['entity'], entity_file, indent=4)

    print("Test and Entity creation complete.")

# Get wikipedia context via API
def create_corpus_and_qrels_files_via_api():
    for entity_file in os.listdir(entity_dir):
        if entity_file.endswith('.entity.json'):
            prop_id = entity_file.split('.')[0]
            print(f"Processing relation file: {prop_id}")
            
            with open(f'{entity_dir}/{entity_file}', 'r', encoding='utf-8') as ef:
                entities = json.load(ef)
                
                # Select a random subset of entities if applicable
                if len(entities) > num_entities_per_relation:
                    entities = random.sample(entities, num_entities_per_relation)
                
                print(f"Processing selected entities: {entities}")
                
                corpus_content = []
                qrels_content = []
                doc_counter = 0

                for entity in entities:
                    title = entity['wiki_title']
                    print(f"Fetching content for: {title}")
                    summary, paragraphs = get_wikipedia_summary_and_paragraphs(title)

                    summary_doc_id = f"{prop_id}_{doc_counter}"
                    doc_counter += 1
                    corpus_content.append({'doc_id': summary_doc_id, 'content': summary})
                    qrels_content.append({'query_id': entity['query_id'], 'doc_id': summary_doc_id, 'score': 1})

                    for paragraph in paragraphs:
                        paragraph_doc_id = f"{prop_id}_{doc_counter}"
                        doc_counter += 1
                        corpus_content.append({'doc_id': paragraph_doc_id, 'content': paragraph})
                        qrels_content.append({'query_id': entity['query_id'], 'doc_id': paragraph_doc_id, 'score': 0})

                with open(f'{corpus_dir}/{prop_id}.corpus.json', 'w', encoding='utf-8') as cf:
                    json.dump(corpus_content, cf)
                print(f"Corpus file created: {prop_id}.corpus.json")

                with open(f'{qrels_dir}/{prop_id}.qrels.json', 'w', encoding='utf-8') as qf:
                    json.dump(qrels_content, qf)
                print(f"Qrels file created: {prop_id}.qrels.json")
    
    print("Corpus and Qrels creation complete.")

# Get wikipedia context via HF datasets
def create_corpus_and_qrels_files_via_hf_datasets():
    
    entities = {}
    queries_id = {}
    corpus = {}
    qrels = {}
    
    row_dataset = load_dataset("wikipedia", "20220301.en", beam_runner='DirectRunner')
    
    for entity_file in os.listdir(entity_dir):
        if entity_file.endswith('.entity.json'):
            
            prop_id = entity_file.split('.')[0]
            # print(f"Processing relation file: {prop_id}")
            
            with open(f'{entity_dir}/{entity_file}', 'r', encoding='utf-8') as ef:
                data = json.load(ef)
                entities[prop_id] = [obj['wiki_title'] for obj in data]
                queries_id[prop_id] = [obj['query_id'] for obj in data]
                corpus[prop_id] = []
                qrels[prop_id] = []
        
            
    doc_counter = 0
    for i, item in enumerate(row_dataset['train']):
        
        if i % 100 == 0:
            break
        
        if i != 0 and i % 500000 == 0:
            print(f"{i} data are processed...")
        
        title = item['title']
        text = item['text']
        
        for idx, (relation_id, entity_list) in enumerate(entities.items()):
            if title in entity_list:
                index = entity_list.index(title)
                
                summary, paragraphs = extract_summary_paragraphs(text)
                
                
                corpus[relation_id].append({
                    'doc_id': f"{relation_id}_{doc_counter}",
                    'title': title,
                    # 'content': text
                    'content': extract_summary(text)
                })
                qrels[relation_id].append({
                    'query_id': queries_id[relation_id][index],
                    'doc_id': f"{relation_id}_{doc_counter}",
                    'score': 1
                })
                doc_counter += 1
                break

    for relation_id, value in corpus.items():
        filename = f"{relation_id}.corpus.json"
        filepath = f"{corpus_dir}/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as json_file:
            json.dump(value, json_file, indent=4) # ensure_ascii=False

    for relation_id, value in qrels.items():
        filename = f"{relation_id}.qrels.json"
        filepath = f"{qrels_dir}/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as json_file:
            json.dump(value, json_file, indent=4)
  
def check_entities():
    
    entities = {}
    all_org_entities = 0
    all_found_entities = 0
    all_not_found_entities = 0
    all_empty_found_entities = 0
    for entity_file in os.listdir(entity_dir):
        if entity_file.endswith('.entity.json'):
            relation_id = entity_file.split('.')[0]
            entities[relation_id] = {}
            
            with open(f'{entity_dir}/{entity_file}', 'r', encoding='utf-8') as ef, open(f'{corpus_sum_dir}/{relation_id}.corpus.json', 'r', encoding='utf-8') as cf:
                en_data = json.load(ef)
                cr_data = json.load(cf)
                
                org_entities_list = [obj['wiki_title'] for obj in en_data]
                entities[relation_id]["org_entities"] = org_entities_list
                entities[relation_id]["org_len"] = len(org_entities_list)
                all_org_entities += len(org_entities_list)
                
                found_entities_list = [obj['title'] for obj in cr_data]
                entities[relation_id]["fnd_entities"] = found_entities_list
                entities[relation_id]["fnd_len"] = len(found_entities_list)
                all_found_entities += len(found_entities_list)
                
                not_found_entities_list = [item for item in org_entities_list if item not in found_entities_list]
                entities[relation_id]["not_fnd"] = not_found_entities_list
                entities[relation_id]["len_not_fnd"] = len(not_found_entities_list)
                all_not_found_entities += len(not_found_entities_list)
                
                # not_exist_entities_list = org_entities_list - (found_entities_list + not_found_entities_list)
                not_exist_entities_list = [item for item in org_entities_list if item not in (found_entities_list + not_found_entities_list)]
                entities[relation_id]["not_exist"] = not_exist_entities_list
                entities[relation_id]["len_not_exist"] = len(not_exist_entities_list)
                
                entities[relation_id]['empty_fnd'] = [obj["title"] for obj in cr_data if obj['content'] == ""]
                entities[relation_id]['len_empty_fnd'] = len(entities[relation_id]['empty_fnd'])
                all_empty_found_entities += len(entities[relation_id]['empty_fnd'])
                            
                            
    print("Original entity num.: {}".format(all_found_entities))
    print("All found: {}".format(all_found_entities))
    print("All not found: {}".format(all_not_found_entities))
    print("All empty found: {}".format(all_empty_found_entities))
    
    with open(entities_analysis_file, 'w', encoding='utf-8') as ef:
        json.dump(entities, ef, indent=4)
  
def add_empty_entities():
    
    # def get_from_dataset(entity_name):
    #     title = item['title']
    #     text = item['text']
    #     for i, item in enumerate(row_dataset['train']):
    #         if title == entity_name:
    #             return text
        
    # with open(entities_analysis_file, 'r', encoding='utf-8') as eaf:
    #     en_data = json.load(eaf)
        
    #     for relation_id, value in en_data.items():
    #         empty_entities = value['empty_fnd']
    #         if empty_entities:
    #             with open(f'{corpus_dir}/{relation_id}.corpus.json', 'r', encoding='utf-8') as cf:
    #                 cr_data = json.load(cf)
                    
    #                 for item in cr_data:
    #                     if item['title'] in empty_entities:
    #                         new_content = get_from_dataset(item['title'])
    #                         item['content'] = extract_summary(new_content)
                    
    #             with open(f'{corpus_dir}/{relation_id}.corpus.json', 'w', encoding='utf-8') as cf:
    #                 json.dump(cr_data, cf, indent=4)
    
    #     entities_analysis_file
    
    with open('component0_preprocessing/generated_data/popQA_EQformat/entities_analysis.json', 'r') as f:
        data = json.load(f)
    
    all_empty_entities = {}
    for key, value in data.items():
        for entity in value["empty_fnd"]:
            all_empty_entities[entity] = {"relation_id": key, "content": None}    
    
    for i, item in enumerate(row_dataset['train']):
        if item in all_empty_entities.key():
            all_empty_entities[item]["content"] = extract_summary(item['text'])
        
    # convert 
    output_obj = {}
    for entity_name, details in all_empty_entities.items():
        relation_id = details["relation_id"]
        content = details["content"]

        if relation_id not in output_obj:
            output_obj[relation_id] = []

        output_obj[relation_id].append({"entity_name": entity_name, "content": content})
        
    
    for relation_id, value in output_obj.items():
        filename = f"{relation_id}.corpus.json"
        filepath = f"{corpus_dir}/{filename}"
        
        
        with open(filepath, 'r', encoding='utf-8') as json_file:
            data = json.dump(json_file)
        
        
        for item in value:
            for obj in data:
                if item['entity_name'] == obj['title']:
                    obj['content'] = item['content']
                    break 
        
        with open(f"{corpus_dir}/{relation_id}.new_corpus.json", 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)   

def create_train_and_dev_files_pipeline(args, relation_id=None):
    model = TransformersQG(
        language='en',
        model=args.qg_model,
        model_ae=args.ae_model,
        skip_overflow_error=True,
        drop_answer_error_text=True,
    )
    
    # V1
    # def create_train_and_dev_files_for_relation(relation_id):
    #     corpus_file = f"{relation_id}.corpus.json"
    #     print(f"Processing corpus file: {corpus_file}")
    #     query_id_counter = 0
        
    #     with open(f'{corpus_sum_dir}/{corpus_file}', 'r', encoding='utf-8') as cf:
    #         corpus_data = json.load(cf)
                
    #         all_qas = []
    #         qrels_train = []
    #         for item in corpus_data:
    #             content = remove_parentheses(item['content'])
    #             doc_id = item['doc_id']
                
    #             chunks_to_process = [(content, max_tokens)]
    #             while chunks_to_process:
                    
    #                 try: 
    #                     current_chunk, current_max_tokens = chunks_to_process.pop(0)
    #                     qas, split_chunks = generate_qa_with_memory_handling(model, current_chunk, current_max_tokens)
                        
    #                     if qas is not None:
    #                         print(qas)
    #                         for question, answer in qas:                                
                                
    #                             for question, answer in qas:
    #                                 qa_dict = {
    #                                     'query_id': f"qa_{relation_id}_{query_id_counter}",
    #                                     'question': question,
    #                                     'answers': [answer]
    #                                 }
    #                                 all_qas.append(qa_dict)
    #                                 qrels_train.append({
    #                                     'query_id': qa_dict['query_id'],
    #                                     'doc_id': doc_id,
    #                                     'score': 1
    #                                 })
    #                                 query_id_counter += 1
    #                             torch.cuda.empty_cache()
                                
    #                     elif split_chunks:
    #                         chunks_to_process.extend([(chunk, current_max_tokens) for chunk in split_chunks])
    #                     else:
    #                         print(f"Unable to process chunk due to memory constraints: {current_chunk}")

    #                 except AnswerNotFoundError:
    #                     print(f"Answer not found for passage: {content}")
    #                     continue
    #                 except ExceedMaxLengthError:
    #                     print(f"Input exceeded max length for passage: {content}")
    #                     continue
    #                 except ValueError as e:
    #                     print(f"For: {content}")
    #                     print(str(e))
    #                     continue
                
    #         random.shuffle(all_qas)
    #         split_index = int(len(all_qas) * dev_split)
    #         dev_qas = all_qas[:split_index]
    #         train_qas = all_qas[split_index:]

    #         with open(f'{train_dir}/{relation_id}.train.json', 'w', encoding='utf-8') as tf:
    #             json.dump(train_qas, tf, indent=4)

    #         with open(f'{dev_dir}/{relation_id}.dev.json', 'w', encoding='utf-8') as df:
    #             json.dump(dev_qas, df, indent=4)
            
    #         with open(f'{qrels_train_dir}/{relation_id}.qrels-train.json', 'w', encoding='utf-8') as qf:
    #             json.dump(qrels_train, qf, indent=4)

    # V2
    def create_train_and_dev_files_for_relation(relation_id):
        corpus_file = f"{relation_id}.corpus.json"
        print(f"Processing corpus file: {corpus_file}")
        query_id_counter = 0
        
        with open(f'{corpus_sum_dir}/{corpus_file}', 'r', encoding='utf-8') as cf:
            corpus_data = json.load(cf)
                
            all_qas = []
            qrels_train = []
            for item in corpus_data:
                context = remove_parentheses(item['content'])
                doc_id = item['doc_id']
                
                max_tokens = 512
                chunks = split_text_to_sentences(context, max_tokens)
                for chunk in chunks:
                    
                    try:
                        with torch.no_grad():
                            qas = model.generate_qa(chunk)
                    
                        if qas is not None:
                            print(qas)
                            for question, answer in qas:                                
                                
                                for question, answer in qas:
                                    qa_dict = {
                                        'query_id': f"qa_{relation_id}_{query_id_counter}",
                                        'question': question,
                                        'answers': [answer]
                                    }
                                    all_qas.append(qa_dict)
                                    qrels_train.append({
                                        'query_id': qa_dict['query_id'],
                                        'doc_id': doc_id,
                                        'score': 1
                                    })
                                    query_id_counter += 1
                    
                    except torch.cuda.OutOfMemoryError:
                        print("CUDA out of memory.")
                        continue
                    except AnswerNotFoundError:
                        print(f"Answer not found for passage: {chunk}")
                        continue
                    except ExceedMaxLengthError:
                        print(f"Input exceeded max length for passage: {chunk}")
                        continue
                    except ValueError as e:
                        print(f"For: {chunk}")
                        print(str(e))
                        continue
        
        random.shuffle(all_qas)
        split_index = int(len(all_qas) * dev_split)
        dev_qas = all_qas[:split_index]
        train_qas = all_qas[split_index:]

        with open(f'{train_dir}/{relation_id}.train.json', 'w', encoding='utf-8') as tf:
            json.dump(train_qas, tf, indent=4)

        with open(f'{dev_dir}/{relation_id}.dev.json', 'w', encoding='utf-8') as df:
            json.dump(dev_qas, df, indent=4)
        
        with open(f'{qrels_train_dir}/{relation_id}.qrels-train.json', 'w', encoding='utf-8') as qf:
            json.dump(qrels_train, qf, indent=4)
        

    if relation_id == None:
        for corpus_file in os.listdir(corpus_sum_dir):
            if corpus_file.endswith('.corpus.json'):
                
                prop_id = corpus_file.split('.')[0]
                create_train_and_dev_files_for_relation(prop_id)
    else: 
        create_train_and_dev_files_for_relation(relation_id)
             
    print("Train, Dev, and Qrels-Train creation complete.")

def create_train_and_dev_files_prompting(relation_id):
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    prompt_qa_generation = lambda context: f"""
    Example output: {{“question”: “”, “answer”: ""}}

    Context: {context}

    Step 1: Identify spans that are likely to be answers to questions, identify as many as possible.
    Step 2: For each identified span, generate a question.
    Step 3: Respond to the question in only a few tokens concisely.
    Step 4: Output in JSON format following the example above (i.e., `{{...}}`).
    Ensure that you distinctly label and delineate Steps 1, 2, 3, and 4. Let's think step by step:
    """.replace('    ', '')
    
    query_id_counter = 0
    
    with open(f'{corpus_sum_dir}/{relation_id}.corpus.json', 'r', encoding='utf-8') as cf:
        data = json.load(cf)
        
        all_qas = []
        qrels_train = []
        # for item in tqdm(data, desc=f"Processing {relation_id} ..."):
        for idx, item in enumerate(data):
            
            if idx == 5:
                break
            
            context = item['content']
            doc_id = item['doc_id']
            
            max_tokens = 256
            chunks = split_text_to_sentences(context, max_tokens)
            for chunk in chunks:
            
                _prompt = [
                    { "role": "system", "content": "You are a question-answer generator. Your goal is to generate question-answer pairs given the Context.\n"},
                    { "role": "user", "content": prompt_qa_generation(chunk)}
                ]
            
                prompt = pipe.tokenizer.apply_chat_template(_prompt, tokenize=False, add_generation_prompt=True)
                outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
                new_pt = outputs[0]["generated_text"]
                qas = extract_json_objects(new_pt)
            
                if qas is not None:
                    # print(qas)
                    for qa in qas:
                        if "question" in qa.keys() and "answer" in qa.keys():
                            # print("The question is: {}".format(qa["question"]))
                            # print("The answer is: {}".format(qa["answer"]))                     
                        
                            all_qas.append({
                                'query_id': f"qa_{relation_id}_{query_id_counter}",
                                'question': qa["question"],
                                'answers': [qa["answer"]]
                            })
                            qrels_train.append({
                                'query_id': f"qa_{relation_id}_{query_id_counter}",
                                'doc_id': doc_id,
                                'score': 1
                            })
                            query_id_counter += 1
                        else:
                            print("This QA object is missing either 'question' or 'answer' keys:", qa.keys())
    
    # Filtering step
    pattern = r'context\W'
    
    # filtered_qas = [qa for qa in all_qas if len(qa["question"].split()) >= 4]    
    filtered_qas = [
        qa for qa in all_qas 
        if isinstance(qa["question"], str) and isinstance(qa["answers"], list) and
            all(isinstance(answer, str) for answer in qa["answers"]) and
            len(qa["question"].split()) >= 4 and
            not re.search(pattern, qa["question"], re.IGNORECASE) and
            not any(re.search(pattern, answer, re.IGNORECASE) for answer in qa["answers"])
    ]
    
    random.shuffle(filtered_qas)
    split_index = int(len(filtered_qas) * dev_split)
    train_qas = filtered_qas[split_index:]
    dev_qas = filtered_qas[:split_index]

    with open(f'{pr_train_dir}/{relation_id}.train.json', 'w', encoding='utf-8') as tf:
        json.dump(train_qas, tf, indent=4)
    
    with open(f'{pr_dev_dir}/{relation_id}.dev.json', 'w', encoding='utf-8') as df:
        json.dump(dev_qas, df, indent=4)
    
    with open(f'{pr_qrels_train_dir}/{relation_id}.qrels-train.json', 'w', encoding='utf-8') as qf:
        json.dump(qrels_train, qf, indent=4)

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

def plot_bucket_num():
    split_points = [2, 3, 4, 5] # Good for popqa_pageviews
    # split_points = [3, 4, 5, 6] # Good for my pageviews
    
    # if dataset_name == 'popqa':
    #     fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 8))
    #     fig.delaxes(axes[0,1])  # Remove the first subplot (top-left)
    #     fig.delaxes(axes[0,2])  # Remove the third subplot (top-right)
    #     fig.delaxes(axes[0,3])  # Remove the third subplot (top-right)
    #     # fig.delaxes(axes[0,4])  # Remove the third subplot (top-right)
    
    # if dataset_name == 'eq':
    #     fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(12, 8))
    #     fig.delaxes(axes[0,1])  # Remove the first subplot (top-left)
    #     fig.delaxes(axes[0,2])  # Remove the third subplot (top-right)
    #     fig.delaxes(axes[0,3])  # Remove the third subplot (top-right)
    #     fig.delaxes(axes[0,4])  # Remove the third subplot (top-right)
    #     # fig.delaxes(axes[0,5])  # Remove the third subplot (top-right)
    
    
    all_queries = []
    for idx, filename in enumerate(os.listdir(test_dir)):
        if filename.endswith('.json'):
            relation_id = filename.split('.')[0]
            logging.info(f"Processing relation {relation_id}, {RELATIONS[relation_id]} ...")
            print(f"Processing relation {relation_id}, {RELATIONS[relation_id]} ...")
            
            # if dataset_name == 'popqa':
            #     row = (idx // 4) + 1
            #     col = idx % 4
            # if dataset_name == 'eq':
            #     row = (idx // 5) + 1
            #     col = idx % 5
            # ax = axes[row, col]
                        
            query_file_path = os.path.join(test_dir, filename)
            with open(query_file_path, 'r') as qf:
                q_rel_data = json.load(qf) 
            all_queries.extend(q_rel_data)
            
            bk_data = split_to_buckets(q_rel_data, split_points)
            counts = [len(bk) for bk in bk_data.values()]
            # ax.bar(["b1", "b2", "b3", "b4", "b5"], counts)
            # ax.set_title(RELATIONS[relation_id])
    
    # row = 0
    # col = 0
    # ax = axes[row, col]
    # bk_data = split_to_buckets(all_queries, split_points)
    # counts = [len(bk) for bk in bk_data.values()]
    # ax.bar(["b1", "b2", "b3", "b4", "b5"], counts)
    # ax.set_title('all')   
    
    # plt.tight_layout()
    # plt.show()
    
    
    # For only plotting the all queries
    plt.figure(figsize=(8, 5)) 
    font = {
        # 'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
    }
    
    bk_data = split_to_buckets(all_queries, split_points)
    counts = [len(bk) for bk in bk_data.values()]
    buckets = ["b1", "b2", "b3", "b4", "b5"]
    buckets = [f'$10^{i}$' for i in range(2, 7)]
    
    color_left = [0.69, 0.769, 0.871]  # lightsteelblue in RGB
    color_right = [0.255, 0.412, 0.882]  # royalblue in RGB
    interpolation_values = np.linspace(0, 1, len(buckets))
    # colors = [(color_left, color_right, value) for value in interpolation_values]
    
    for i, (bucket, value) in enumerate(zip(buckets, counts)):
        
        color = (1 - interpolation_values[i]) * np.array(color_left) + \
            interpolation_values[i] * np.array(color_right)
        
        plt.bar(bucket, value, color=color)
    # plt.bar(buckets, counts, color=colors)
    plt.xlabel("Popularity (pageviews)", fontdict=font)
    plt.xticks(fontsize=14)
    plt.ylabel("# Samples", fontdict=font)
    plt.yticks(fontsize=14)
    
    plt.yticks(rotation=45)
    plt.tight_layout()
    # plt.savefig(f"pop_bk", dpi=1000)
    plt.savefig('pop_bk.pdf', format='pdf', dpi=1000)
    
    plt.show()
           
def main(args):
    ### ==== Step 1: Creating test & entity files =====================
    # create_test_and_entity_files()
    
    ### ==== Step 1 (for EQ dataset): Creating test & entity files ====
    # create_test_and_entity_files_EQ() # run in Kaggle

    ### ==== Step 2: Creating corpus & qrels files ====================
    # create_corpus_and_qrels_files_via_api()
    # create_corpus_and_qrels_files_via_hf_datasets() # run in Kaggle
    # check_entities()
    # add_empty_entities()
    
    ### ==== Step 3: Creating train & dev & qrels-train files ============
    # Done: 
    # Doing: 
    # To Do:
    relation_id = relation_ids[0]
    print(relation_id)
    # create_train_and_dev_files_pipeline(args, relation_id=relation_id) # T5-based model
    create_train_and_dev_files_prompting(relation_id=relation_id)# Zephyr-based model
    
    ### ==== Plotting the distribution of the number of queries in each bucket
    # plot_bucket_num()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qg_model", type=str, required=True)
    parser.add_argument("--ae_model", type=str, required=True)
    
    args = parser.parse_args()
    main(args)