import json
import ast
import re

# Define the input and output file paths
input_file_path = 'component0_preprocessing/generated_data/popQA_costomized/retrieved_highlight/all.jsonl'
output_file_path = 'component0_preprocessing/generated_data/popQA_costomized/retrieved_highlight/all_out.jsonl'


# Function to safely convert the highlighted_text field to a dictionary
def safe_convert_highlighted_text(highlighted_text_str):
    try:
        return json.loads(highlighted_text_str)
    except json.JSONDecodeError:
        # If there is an error, clean the sentences
        highlighted_text_dict = json.loads('{"relation": "Unknown", "sentences": []}')
        sentences_match = re.search(r'"sentences": \[(.*?)\]', highlighted_text_str)
        if sentences_match:
            sentences = sentences_match.group(1)
            # Remove special characters and split into sentences
            sentences_cleaned = re.sub(r'[^a-zA-Z0-9 ,.\-]', '', sentences)
            sentences_list = [sentence.strip() for sentence in sentences_cleaned.split(',')]
            highlighted_text_dict['sentences'] = sentences_list
        return highlighted_text_dict

# Open the input and output files
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # Process each line in the input file
    for line in input_file:
        # Parse the line as a JSON object
        record = json.loads(line)
        
        # Convert the 'highlighted_text' field from a string to a dictionary
        highlighted_text_str = record.get('highlighted_text', '{}')
        record['highlighted_text'] = safe_convert_highlighted_text(highlighted_text_str)
        
        # Write the updated record to the output file
        output_file.write(json.dumps(record) + '\n')

print(f"Processed records have been written to {output_file_path}")



