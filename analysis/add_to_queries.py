import json
import csv

# Step 1: Read the TSV file and create a dictionary
def read_tsv_to_dict(tsv_file, keyword):
    entity_relation_dict = {}
    with open(tsv_file, 'r') as file:
        tsv_reader = csv.DictReader(file, delimiter='\t')
        for row in tsv_reader:
            entity_id = row['s_uri'].split('/')[-1]
            # entity_id = row['entity_id']
            # relation_type = row['prop']
            # popqa_pop = row['s_pop']
            selected_kw = row[keyword]
            entity_relation_dict[entity_id] = selected_kw
    return entity_relation_dict

# Step 2: Read the JSONL file, update each JSON object, and store them
def update_jsonl_with_relations(jsonl_file, entity_relation_dict, output_file, keyword):
    with open(jsonl_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            json_obj = json.loads(line)
            entity_id = json_obj['entity_id']
            # json_obj['relation_type'] = entity_relation_dict.get(entity_id, None)
            json_obj[keyword] = entity_relation_dict.get(entity_id, None)
            json.dump(json_obj, outfile)
            outfile.write('\n')

# File paths (update these with your actual file paths)
tsv_file = 'data/dataset/popQA/popQA.tsv'
jsonl_file = 'component0_preprocessing/generated_data/popQA_costomized/queries.jsonl'
output_file = 'component0_preprocessing/generated_data/popQA_costomized/queries_new.jsonl'
keyword = "prop_id"

# Execute the functions
entity_relation_dict = read_tsv_to_dict(tsv_file, keyword)
update_jsonl_with_relations(jsonl_file, entity_relation_dict, output_file, keyword)
