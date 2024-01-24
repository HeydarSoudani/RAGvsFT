#!/usr/bin/env python3

import csv
import json
import os
import ast
import re
import torch
import argparse
import random
import wikipediaapi
from lmqg import TransformersQG
from lmqg.exceptions import AnswerNotFoundError, ExceedMaxLengthError
from nltk.tokenize import sent_tokenize

import nltk
nltk.download('punkt')


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



def main(args):
    dataset_name = 'popqa'
    tsv_file_path = "data/dataset/popQA/popQA.tsv"
    output_dir = 'component0_preprocessing/generated_data/popQA_EQformat'
    
    test_dir = f"{output_dir}/test" 
    entity_dir = f"{output_dir}/entity" 
    corpus_dir = f"{output_dir}/corpus" 
    qrels_dir = f"{output_dir}/qrels" 
    train_dir = f"{output_dir}/train" 
    dev_dir = f"{output_dir}/dev" 
    qrels_train_dir = f"{output_dir}/qrels-train" 
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(entity_dir, exist_ok=True)
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(qrels_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(qrels_train_dir, exist_ok=True)
    num_entities_per_relation = 20
    dev_split = 0.1
    max_tokens = 512

    ### ==== Creating test & entity files =====================
    # data_by_prop_id = {}
    # with open(tsv_file_path, 'r', encoding='utf-8') as file:
    #     reader = csv.DictReader(file, delimiter='\t')
    #     for row in reader:
    #         prop_id = row['prop_id']
    #         if prop_id not in data_by_prop_id:
    #             data_by_prop_id[prop_id] = {'test': [], 'entity': [], 'counter': 0}

    #         query_id = f"{prop_id}_{data_by_prop_id[prop_id]['counter']}"
    #         data_by_prop_id[prop_id]['counter'] += 1
            
    #         # Append data to test and entity lists
    #         data_by_prop_id[prop_id]['test'].append({
    #             'query_id': query_id,
    #             'question': row['question'],
    #             'answers': ast.literal_eval(row['possible_answers']),
    #             'pageviews': row['s_pop']
    #         })

    #         data_by_prop_id[prop_id]['entity'].append({
    #             'query_id': query_id,
    #             'wiki_title': row['s_wiki_title'],
    #             'entity_id': row['s_uri'].split('/')[-1]
    #         })

    # for prop_id, content in data_by_prop_id.items():
    #     with open(f'{test_dir}/{prop_id}.test.json', 'w', encoding='utf-8') as test_file:
    #         json.dump(content['test'], test_file)

    #     with open(f'{entity_dir}/{prop_id}.entity.json', 'w', encoding='utf-8') as entity_file:
    #         json.dump(content['entity'], entity_file)
    
    # print("Test and Entity creation complete.")

    ### ==== Creating corpus & qrels files =====================
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
  
    ### ==== Creating train & dev & qrels-train files =====================
    model = TransformersQG(
        language='en',
        model=args.qg_model,
        model_ae=args.ae_model,
        skip_overflow_error=True,
        drop_answer_error_text=True,
    )
    
    query_id_counter = 0
    for corpus_file in os.listdir(corpus_dir):
        if corpus_file.endswith('.corpus.json'):
            prop_id = corpus_file.split('.')[0]
            print(f"Processing corpus file: {corpus_file}")

            with open(f'{corpus_dir}/{corpus_file}', 'r', encoding='utf-8') as cf:
                corpus_data = json.load(cf)
                    
                all_qas = []
                qrels_train = []
                for item in corpus_data:
                    content = remove_parentheses(item['content'])
                    doc_id = item['doc_id']
                    
                    chunks_to_process = [(content, max_tokens)]
                    while chunks_to_process:
                        
                        try: 
                            current_chunk, current_max_tokens = chunks_to_process.pop(0)
                            qas, split_chunks = generate_qa_with_memory_handling(model, current_chunk, current_max_tokens)
                            
                            if qas is not None:
                                print(qas)
                                for question, answer in qas:                                
                                    
                                    for question, answer in qas:
                                        qa_dict = {
                                            'query_id': f"qa_{query_id_counter}",
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
                                    torch.cuda.empty_cache()
                                    
                            elif split_chunks:
                                chunks_to_process.extend([(chunk, current_max_tokens) for chunk in split_chunks])
                            else:
                                print(f"Unable to process chunk due to memory constraints: {current_chunk}")

                        except AnswerNotFoundError:
                            print(f"Answer not found for passage: {content}")
                            continue
                        except ExceedMaxLengthError:
                            print(f"Input exceeded max length for passage: {content}")
                            continue
                        except ValueError as e:
                            print(f"For: {content}")
                            print(str(e))
                            continue
                    
                    # if count_tokens(content) > max_tokens:
                    #     content_chunks = split_text_to_sentences(content, max_tokens)
                    # else:
                    #     content_chunks = [content]
                    
                    # for chunk in content_chunks:
                    #     try:
                    #         with torch.no_grad():
                    #             qas = model.generate_qa(content)
                                
                    #         print(qas)    
                    #         for question, answer in qas:
                    #             qa_dict = {
                    #                 'query_id': f"qa_{query_id_counter}",
                    #                 'question': question,
                    #                 'answers': [answer]
                    #             }
                    #             all_qas.append(qa_dict)
                    #             qrels_train.append({
                    #                 'query_id': qa_dict['query_id'],
                    #                 'doc_id': doc_id,
                    #                 'score': 1
                    #             })
                    #             query_id_counter += 1
                        
                    #         torch.cuda.empty_cache()                            
                        # except AnswerNotFoundError:
                        #     print(f"Answer not found for passage: {chunk}")
                        #     continue
                        # except ExceedMaxLengthError:
                        #     print(f"Input exceeded max length for passage: {chunk}")
                        #     continue
                        # except torch.cuda.OutOfMemoryError:
                        #     print("CUDA out of memory. Trying to free up memory.")
                        #     torch.cuda.empty_cache()  # Attempt to clear cache and continue
                        #     continue
                        # except ValueError as e:
                        #     print(f"For: {chunk}")
                        #     print(str(e))
                        #     continue
                    
                random.shuffle(all_qas)
                split_index = int(len(all_qas) * dev_split)
                dev_qas = all_qas[:split_index]
                train_qas = all_qas[split_index:]

                with open(f'{train_dir}/{prop_id}.train.json', 'w', encoding='utf-8') as tf:
                    json.dump(train_qas, tf, indent=4)

                with open(f'{dev_dir}/{prop_id}.dev.json', 'w', encoding='utf-8') as df:
                    json.dump(dev_qas, df, indent=4)
                
                with open(f'{qrels_train_dir}/{prop_id}.qrels-train.json', 'w', encoding='utf-8') as qf:
                    json.dump(qrels_train, qf, indent=4)

    print("Train, Dev, and Qrels-Train creation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qg_model", type=str, required=True)
    parser.add_argument("--ae_model", type=str, required=True)
    
    args = parser.parse_args()
    main(args)