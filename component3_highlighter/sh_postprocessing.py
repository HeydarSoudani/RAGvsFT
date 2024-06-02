import json
import re
import os


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


def main():
    input_dir = "component0_preprocessing/generated_data/popQA_costomized/highlighted_sentences/dpr_3/row_data"
    output_dir = "component0_preprocessing/generated_data/popQA_costomized/highlighted_sentences/dpr_3"
    
    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.jsonl'):
            relation_id = filename.split('.')[0]
            input_file = f"{input_dir}/{filename}"
            output_file = f"{output_dir}/{filename}"
            print(f"\nProcessing {relation_id}...")
            
            with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
                for line in f_in:
                    record = json.loads(line)
                    
                    sentences_list = record.get('sentences', [])
                    for item in sentences_list:
                        highlighted_text_str = item.get('sentence', '{}')
                        item['sentence'] = safe_convert_highlighted_text(highlighted_text_str)
                    
                    f_out.write(json.dumps(record) + '\n')


if __name__ == "__main__":
    main()