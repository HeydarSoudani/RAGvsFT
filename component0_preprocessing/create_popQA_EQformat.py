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
import matplotlib.pyplot as plt
import wikipediaapi
from lmqg import TransformersQG
from lmqg.exceptions import AnswerNotFoundError, ExceedMaxLengthError
from nltk.tokenize import sent_tokenize
from datasets import load_dataset

import nltk
nltk.download('punkt')

### === Constants =====================  
dataset_name = 'popqa'
tsv_file_path = "data/dataset/popQA/popQA.tsv"
output_dir = 'component0_preprocessing/generated_data/popQA_EQformat'

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

entities_analysis_file = f"{output_dir}/entities_analysis.json"
test_dir = f"{output_dir}/test" 
entity_dir = f"{output_dir}/entity" 
corpus_sum_dir = f"{output_dir}/corpus_summary" 
qrels_sum_dir = f"{output_dir}/qrels_summary" 
corpus_all_dir = f"{output_dir}/corpus_all" 
qrels_all_dir = f"{output_dir}/qrels_all"
os.makedirs(test_dir, exist_ok=True)
os.makedirs(entity_dir, exist_ok=True)
os.makedirs(corpus_sum_dir, exist_ok=True)
os.makedirs(qrels_sum_dir, exist_ok=True)
os.makedirs(corpus_all_dir, exist_ok=True)
os.makedirs(qrels_all_dir, exist_ok=True)


train_dir = f"{output_dir}/pipeline/train" 
dev_dir = f"{output_dir}/pipeline/dev" 
qrels_train_dir = f"{output_dir}/pipeline/qrels-train" 
os.makedirs(f"{output_dir}/pipeline", exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(dev_dir, exist_ok=True)
os.makedirs(qrels_train_dir, exist_ok=True)

num_entities_per_relation = 20
dev_split = 0.1
max_tokens = 512

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
        for row in reader:
            prop_id = row['prop_id']
            if prop_id not in data_by_prop_id:
                data_by_prop_id[prop_id] = {'test': [], 'entity': [], 'counter': 0}

            query_id = f"{prop_id}_{data_by_prop_id[prop_id]['counter']}"
            data_by_prop_id[prop_id]['counter'] += 1
            
            # Append data to test and entity lists
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

    for prop_id, content in data_by_prop_id.items():
        with open(f'{test_dir}/{prop_id}.test.json', 'w', encoding='utf-8') as test_file:
            json.dump(content['test'], test_file)

        with open(f'{entity_dir}/{prop_id}.entity.json', 'w', encoding='utf-8') as entity_file:
            json.dump(content['entity'], entity_file)
    
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
            
            with open(f'{entity_dir}/{entity_file}', 'r', encoding='utf-8') as ef, open(f'{corpus_dir}/{relation_id}.corpus.json', 'r', encoding='utf-8') as cf:
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

def create_train_and_dev_files(args, relation_id=None):
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
                
                max_tokens = 256
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
    
    
    all_queries = []
    for idx, filename in enumerate(os.listdir(test_dir)):
        if filename.endswith('.json'):
            relation_id = filename.split('.')[0]
            logging.info(f"Processing relation {relation_id}, {RELATIONS[relation_id]} ...")
            print(f"Processing relation {relation_id}, {RELATIONS[relation_id]} ...")
            
            if dataset_name == 'popqa':
                row = (idx // 4) + 1
                col = idx % 4
            if dataset_name == 'eq':
                row = (idx // 5) + 1
                col = idx % 5
            ax = axes[row, col]
                        
            query_file_path = os.path.join(test_dir, filename)
            with open(query_file_path, 'r') as qf:
                q_rel_data = json.load(qf) 
            all_queries.extend(q_rel_data)
            
            bk_data = split_to_buckets(q_rel_data, split_points)
            counts = [len(bk) for bk in bk_data.values()]
            ax.bar(["b1", "b2", "b3", "b4", "b5"], counts)
            ax.set_title(RELATIONS[relation_id])
    
    row = 0
    col = 0
    ax = axes[row, col]
    bk_data = split_to_buckets(all_queries, split_points)
    counts = [len(bk) for bk in bk_data.values()]
    ax.bar(["b1", "b2", "b3", "b4", "b5"], counts)
    ax.set_title('all')   
    
    plt.tight_layout()
    plt.show()
            

def main(args):
    ### ==== Creating test & entity files =================
    create_test_and_entity_files()

    ### ==== Creating corpus & qrels files ================
    # create_corpus_and_qrels_files_via_api()
    # create_corpus_and_qrels_files_via_hf_datasets()
    # check_entities()
    # add_empty_entities()
    
    ### ==== Creating train & dev & qrels-train files =====
    # Done: 106, 
    # Doing: 22, 182
    # To Do: 182, 218, 91, 257, 164, 526, 97, 533, 639, 472, 560, 484, 292, 422
    # relation_id = "182"
    # create_train_and_dev_files(args, relation_id=relation_id)
    
    ### ==== Plotting the distribution of the number of queries in each bucket
    plot_bucket_num()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--qg_model", type=str, required=True)
    # parser.add_argument("--ae_model", type=str, required=True)
    
    args = parser.parse_args()
    main(args)