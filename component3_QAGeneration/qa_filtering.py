import argparse, json

def should_remove_entry(question, answers):

    if question.lower().startswith(("how old", "when", "how much", "how many", "in what year")):
        return True

    for answer in answers:
        if len(answer.split()) > 6:
            return True

    return False


def filter_jsonl(input_file_path, output_file_path):
    
    with open(input_file_path, 'r') as in_file, open(output_file_path, 'w') as out_file:
        for line in in_file:
            entry = json.loads(line.strip())

            question = entry.get("question", "")
            answers = entry.get("possible_answers", [])

            if not should_remove_entry(question, answers):
                json.dump(entry, out_file)
                out_file.write('\n')

def main(args):
    input_file_path = 'component0_preprocessing/generated_data/popQA_sm/filtered_qag_synthetic.jsonl'
    output_file_path = 'component0_preprocessing/generated_data/popQA_sm/typed_filtered_qag_synthetic.jsonl'

    filter_jsonl(input_file_path, output_file_path) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", type=str, required=True)
    # parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)