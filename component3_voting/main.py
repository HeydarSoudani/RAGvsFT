import os
import json
import random
import logging
from tqdm import tqdm
from difflib import SequenceMatcher

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import numpy as np
from nltk.metrics.distance import edit_distance

from transformers import pipeline
from transformers import AutoTokenizer


base_path  = "component0_preprocessing/generated_data"
dataset_name = 'popQA'
target_relation_ids = 'all'
subset_percentage = 1.0
similarity_method = 'prompting' # 'seq_match', 'bert_score', 'cosine', 'jaccard', 'levenshtein', 'prompting'

gen_models = [
    "flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl",
    "stable_lm2", "tiny_llama", "MiniCPM",
    "llama2", "mistral", "zephyr"
]
    
retrieval_model = 'ideal'
model_name = gen_models[-1]

if model_name in ["flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl"]:
    model_type = 'flant5'
elif model_name in ["stable_lm2", "tiny_llama", "MiniCPM"]:
    model_type = 'slms'
elif model_name in ["llama2", "mistral", "zephyr"]:
    model_type = 'llms'

result_files = [
    {"title": "NoFT/NoRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_bf_norag_full_results.jsonl"},
    # {"title": f"NoFT/idealRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_bf_rag_{retrieval_model}_full_results.jsonl"},
    {"title": "FT/NoRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_af_norag_peft_results.jsonl"},
    # {"title": f"FT/idealRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_af_rag_{retrieval_model}_peft_results.jsonl"},
    # {"title": f"NoFT/dprRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{dataset_name}_{model_name}_bf_rag_dpr_full_results.jsonl"},
    # {"title": f"FT/dprRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{dataset_name}_{model_name}_af_rag_dpr_peft_results.jsonl"},
]

dataset_dir = 'component0_preprocessing/generated_data/{}_costomized'.format(dataset_name)
test_dir = f"{dataset_dir}/test"
out_results_path = f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_voting_{similarity_method}_results.jsonl"


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_relations_data():
    subfolders = ['test']
        
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = f"{base_path}/{dataset_name}_costomized/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                relation_id = file.split('.')[0]
                if relation_id not in relation_files:
                    relation_files[relation_id] = []
                relation_files[relation_id].append(os.path.join(subfolder_path, file))    

    # Select relations =================
    if target_relation_ids == "all":
        if dataset_name == 'popQA':
            test_relation_ids = ['22', '91', '97', '106', '164', '182', '218', '257', '292', '422', '472', '484', '526', '533', '560', '639']
        elif dataset_name == 'witQA':
            test_relation_ids = ['17', '19', '22', '25', '27', '36', '50', '57', '58', '69', '86', '106', '123', '136', '140', '149', '162', '184', '344', '452', '462', '641', '674', '1038', '1050', '1376', '1431', '1433', '2012', '2936', '3301', '4647']
        elif dataset_name == 'EQ':
            test_relation_ids = ['17', '19', '20', '26', '30', '36', '40', '50', '69', '106', '112', '127', '131', '136', '159', '170', '175', '176', '264', '276', '407', '413', '495', '740', '800']
    else:
        test_relation_ids = target_relation_ids
    
    test_files = {subfolder: [] for subfolder in subfolders}
    
    for subfolder in subfolders:
        subfolder_path = f"{base_path}/{dataset_name}_costomized/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_id = file.split('.')[0]
                if file_id in test_relation_ids:
                    test_files[subfolder].append(os.path.join(subfolder_path, file))

    print("Selected Relation ID:", test_relation_ids)
    logging.info(f"Selected Relation ID: {test_relation_ids}")

    return test_relation_ids, test_files, relation_files

def load_dataset(test_files):
    test_data = []
    for file in test_files['test']:
        relation_id = file.split('/')[-1].split('.')[0]
        data = load_json_file(file)
        for item in data:
            item['relation_id'] = relation_id
        test_data.extend(data) 
        
    test_subset_size = int(subset_percentage * len(test_data))
    subset_test_data = random.sample(test_data, test_subset_size)    
    test_questions = [(item['query_id'], item['question'], item['pageviews'], item['relation_id']) for item in subset_test_data]
    test_answers = [item['answers'] for item in subset_test_data]
    
    logging.info("Test dataset is loaded.")
    # print("Test dataset is loaded.")
    return test_questions, test_answers


def get_pred_values(query_id, file_list):
    # Dictionary to store results with filename as key and pred value as value
    results = {}

    # Loop over each file in the list
    for file_name in file_list:
        # Open and read the JSONL file
        with open(file_name, 'r') as file:
            # Read each line (each line is a JSON object)
            for line in file:
                # Convert JSON string to a Python dictionary
                data = json.loads(line)
                
                # Check if the current dictionary has the query_id we're looking for
                if data.get('query_id') == query_id:
                    # Extract the 'pred' value
                    pred_value = data.get('pred', 'Not available')
                    # Store the result
                    results[file_name] = pred_value
                    break  # Assuming only one matching queryId per file
            else:
                # If no match was found in the entire file, mark as not found
                results[file_name] = 'queryId not found in this file'

    return results

def seq_match_sim(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def bert_score_sim(s1, s2):
    candidate = word_tokenize(s1.lower())
    reference = word_tokenize(s2.lower())
    return sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))

def cos_sim(s1, s2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([s1, s2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    return cosine_sim

def jaccard_sim(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    jaccard_sim = 1 - jaccard(list(set1), list(set2))
    return jaccard_sim
    
def levenshtein_distance(s1, s2):
    edit_dist = edit_distance(s1, s2)
    return edit_dist

def vote(samples):
    """ Determine the sample with the highest total similarity score to all other samples. """
    # Initialize similarity scores dictionary
    similarity_scores = {sample: 0 for sample in samples}

    # Calculate similarity scores for each sample
    for i, si in enumerate(samples):
        for j, sj in enumerate(samples):
            if i != j:
                if similarity_method == 'seq_match':
                    similarity_scores[si] += seq_match_sim(si, sj) 
                
                elif similarity_method == 'bert_score':
                    similarity_scores[si] += bert_score_sim(si, sj)
                
                elif similarity_method == 'cosine':
                    similarity_scores[si] += cos_sim(si, sj)
                
                elif similarity_method == 'jaccard':
                    similarity_scores[si] += jaccard_sim(si, sj)
                
                elif similarity_method == 'levenshtein':
                    similarity_scores[si] += levenshtein_distance(si, sj)

    # Find the sample with the maximum similarity score
    final_answer = max(similarity_scores, key=similarity_scores.get)
    return final_answer


def prompting_final_answer(query, samples):
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-small",
        trust_remote_code=True
    )
    pipe = pipeline(
        task="text2text-generation",
        model="google/flan-t5-small",
        tokenizer=tokenizer,
        max_new_tokens = None
    )
    prompt = """Context: {context} \n Based on the provided context, answer the question: {question}"""
    prompt.format(context=' '.join(samples), question=query)
    
    result = pipe(prompt)[0]['generated_text']
    return result


def main():
    

    test_relation_ids, test_files, relation_files = load_relations_data()
    test_questions, test_answers = load_dataset(test_files)
    result_list = [result['filename'] for result in result_files]
    
    accuracy = []
    
    with open(out_results_path, 'w') as file:
        for idx, (query_id, query, query_pv, query_relation) in enumerate(tqdm(test_questions)):
            
            if idx == 10:
                break
        
            answers = get_pred_values(query_id, result_list)
            # pred = vote(list(answers.values()))
            pred = prompting_final_answer(query, list(answers.values()))
        
            is_correct = False
            for pa in test_answers[idx]:
                    if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                        is_correct = True
            accuracy.append(is_correct)
            
            if idx < 10 or idx % 300 == 0:
                print('\n')
                print(f"Query: {query}")
                print(f"Pred: {pred}")
                print(f"Labels: {test_answers[idx]}")
                print(f"Final decision: {is_correct}")
                print('====')
            
            item = {
                "query_id": query_id,
                "question": query,
                "possible_answers": test_answers[idx],
                "pred": pred,
                "is_correct": is_correct,
                "pageviews": query_pv
            }
            file.write(json.dumps(item) + '\n')

    acc = sum(accuracy) / len(accuracy)
    logging.info(f"Accuracy: {acc * 100:.2f}%")
    print(f"Accuracy: {acc * 100:.2f}%")


if __name__ == '__main__':
    main()