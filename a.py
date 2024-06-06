import json
import ast
import re
import os
import matplotlib.pyplot as plt
from collections import Counter


def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def count_ref_indexes(data):
    ref_indexes = [entry['ref_index'] for entry in data]
    return Counter(ref_indexes)

def find_ref_index(query_id, ref_doc_id, b_data):
    for index, entry in enumerate(b_data):
        if entry['id'] == query_id:
            for i, ctx in enumerate(entry['ctxs']):
                if ctx['id'] == ref_doc_id:
                    return i
            return "other"
    return "other"

def process_files(a_dir, b_dir, output_file):
    
    results = []    
    for a_filename in os.listdir(a_dir):
        if a_filename.endswith('.jsonl'):
            rel_id = a_filename.split('.')[0]
            a_file = os.path.join(a_dir, a_filename)
            b_file = os.path.join(b_dir, f"{rel_id}.dpr.ret_results.jsonl")
    
            a_data = load_jsonl(a_file)
            b_data = load_jsonl(b_file)

            for a_entry in a_data:
                query_id = a_entry['id']
                ref_doc_id = a_entry['sentences'][1]['ref_doc_id']
                ref_index = find_ref_index(query_id, ref_doc_id, b_data)
                results.append({"question_id": query_id, "ref_index": ref_index})

            with open(output_file, 'w') as out_file:
                for result in results:
                    out_file.write(json.dumps(result) + '\n')

def plot_ref_index_distribution(counts):
    indexes = list(range(5))  # ref_index can be 0, 1, 2, 3, 4
    values = [counts.get(i, 0) for i in indexes]
    
    for i, count in counts.items():
        print(f"ref_index: {i}, count: {count}")

    plt.bar(indexes, values, tick_label=indexes)
    plt.xlabel('ref_index')
    plt.ylabel('Number of occurrences')
    plt.title('Distribution of ref_index values')
    plt.show()

# Specify your file paths
a_dir = 'component0_preprocessing/generated_data/popQA_costomized/reranked_sentences/dpr_5'
b_dir = 'component0_preprocessing/generated_data/popQA_costomized/retrieved_passage/dpr_5'
a_file_path = 'component0_preprocessing/generated_data/popQA_costomized/reranked_sentences/dpr_5/22.dpr.set_reranked.jsonl'
b_file_path = 'component0_preprocessing/generated_data/popQA_costomized/retrieved_passage/dpr_5/22.dpr.ret_results.jsonl'
output_file_path = 'output_file.jsonl'
# Process the files
process_files(a_dir, b_dir, output_file_path)


# plot
data = load_jsonl(output_file_path)
ref_index_counts = count_ref_indexes(data)
plot_ref_index_distribution(ref_index_counts)


# # Function to safely convert the highlighted_text field to a dictionary
# def safe_convert_highlighted_text(highlighted_text_str):
#     try:
#         return json.loads(highlighted_text_str)
#     except json.JSONDecodeError:
#         # If there is an error, clean the sentences
#         highlighted_text_dict = json.loads('{"relation": "Unknown", "sentences": []}')
#         sentences_match = re.search(r'"sentences": \[(.*?)\]', highlighted_text_str)
#         if sentences_match:
#             sentences = sentences_match.group(1)
#             # Remove special characters and split into sentences
#             sentences_cleaned = re.sub(r'[^a-zA-Z0-9 ,.\-]', '', sentences)
#             sentences_list = [sentence.strip() for sentence in sentences_cleaned.split(',')]
#             highlighted_text_dict['sentences'] = sentences_list
#         return highlighted_text_dict

# # Open the input and output files
# with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
#     # Process each line in the input file
#     for line in input_file:
#         # Parse the line as a JSON object
#         record = json.loads(line)
        
#         # Convert the 'highlighted_text' field from a string to a dictionary
#         highlighted_text_str = record.get('highlighted_text', '{}')
#         record['highlighted_text'] = safe_convert_highlighted_text(highlighted_text_str)
        
#         # Write the updated record to the output file
#         output_file.write(json.dumps(record) + '\n')

# print(f"Processed records have been written to {output_file_path}")


# # Function to safely convert the highlighted_text field to a dictionary
# def safe_convert_highlighted_text(highlighted_text_str):
#     try:
#         return json.loads(highlighted_text_str)
#     except json.JSONDecodeError:
#         # If there is an error, clean the sentences
#         highlighted_text_dict = json.loads('{"relation": "Unknown", "sentences": []}')
#         sentences_match = re.search(r'"sentences": \[(.*?)\]', highlighted_text_str)
#         if sentences_match:
#             sentences = sentences_match.group(1)
#             # Remove special characters and split into sentences
#             sentences_cleaned = re.sub(r'[^a-zA-Z0-9 ,.\-]', '', sentences)
#             sentences_list = [sentence.strip() for sentence in sentences_cleaned.split(',')]
#             highlighted_text_dict['sentences'] = sentences_list
#         return highlighted_text_dict

# # Open the input and output files
# with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
#     # Process each line in the input file
#     for line in input_file:
#         # Parse the line as a JSON object
#         record = json.loads(line)
        
#         # Convert the 'highlighted_text' field from a list of strings to a list of dictionaries
#         highlighted_text_list = record.get('highlighted_text', [])
#         for item in highlighted_text_list:
#             highlighted_text_str = item.get('highlighted', '{}')
#             item['highlighted'] = safe_convert_highlighted_text(highlighted_text_str)
        
#         # Write the updated record to the output file
#         output_file.write(json.dumps(record) + '\n')

# print(f"Processed records have been written to {output_file_path}")
