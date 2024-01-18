
import json
import csv

def convert_jsonl_to_tsv(jsonl_file_path, tsv_file_path):
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file, \
         open(tsv_file_path, 'w', encoding='utf-8') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["query_id", "doc_id", "score"])

        for idx, line in enumerate(jsonl_file):
            data = json.loads(line)
            if data['score'] == 1:
                tsv_writer.writerow([data['query_id'], data['doc_id'], data['score']])

# Example usage
# jsonl_file_path = "component0_preprocessing/generated_data/popQA_religion/test/qrels_30ds_512tk.jsonl"
# tsv_file_path = "component0_preprocessing/generated_data/popQA_religion/test/qrels/qrels_30ds_512tk.tsv"

jsonl_file_path = "component1_retrieval/data/popqa_religion/qrels/gen_qrels.jsonl"
tsv_file_path = "component1_retrieval/data/popqa_religion/qrels/train.tsv"

convert_jsonl_to_tsv(jsonl_file_path, tsv_file_path)
print("Conversion complete.")
