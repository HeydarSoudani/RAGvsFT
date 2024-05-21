import os
import json

def merge_jsonl_files(input_directory, output_file):
    # Initialize an empty list to store the merged data
    merged_data = []

    # Iterate through each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, 'r') as file:
                # Read each line (JSON object) from the file and append it to the merged_data list
                for line in file:
                    merged_data.append(json.loads(line.strip()))

    # Write the merged data into the output file
    with open(output_file, 'w') as output:
        # Write each JSON object as a line in the output file
        for data in merged_data:
            output.write(json.dumps(data) + '\n')

# Example usage:
dataset_name = 'popQA'
model_name = 'stable_lm2'
input_directory_path = "component3_moe/naive_method/popQA_chunked_results"
output_file_path = f"component0_preprocessing/generated_data/popQA_costomized/results/{dataset_name}_{model_name}_moe_llm_results.jsonl"
merge_jsonl_files(input_directory_path, output_file_path)

