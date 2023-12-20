
import json
import csv

def convert_jsonl_to_tsv(jsonl_file_path, tsv_file_path):
    """
    Converts a JSONL file to a TSV file. Assumes each JSON object in the JSONL file 
    contains 'id' and 'contents' keys.

    Args:
    jsonl_file_path (str): Path to the input JSONL file.
    tsv_file_path (str): Path to the output TSV file.
    """
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file, \
         open(tsv_file_path, 'w', encoding='utf-8') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        # Write the header
        tsv_writer.writerow(["id", "text"])

        # Process each line in the JSONL file
        for idx, line in enumerate(jsonl_file):
            
            if idx == 100:
                break
            
            # Load the JSON object from the line
            data = json.loads(line)

            # Extract the 'id' and 'contents' values
            data_id = data['id']
            data_cont = data['contents']

            # Write to the TSV file
            tsv_writer.writerow([data_id, data_cont])
            # tsv_file.write(f"{data_id}\t{data_text}\n")
            # tsv_writer.writerow([word, count])

# Example usage
jsonl_file_path = "data/generated/popQA_costomized/corpus.jsonl"
tsv_file_path = "data/generated/popQA_costomized/corpus.tsv"
convert_jsonl_to_tsv(jsonl_file_path, tsv_file_path)
print("Conversion complete.")


