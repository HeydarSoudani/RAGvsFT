import json


if __name__ == "__main__":

    PopQA = "data/generated/popQA_costomized/queries.jsonl"
    EQ_train = "data/generated/EntityQuestions_costomized/train/queries.jsonl"
    EQ_dev = "data/generated/EntityQuestions_costomized/dev/queries.jsonl"
    EQ_test = "data/generated/EntityQuestions_costomized/test/queries.jsonl"
    
    
    # Read entity_ids from the first file
    entity_ids_file1 = set()
    with open(PopQA, 'r') as f1:
        for line in f1:
            data = json.loads(line)
            entity_ids_file1.add(data.get('entity_id'))

    # Read entity_ids from the second file
    entity_ids_file2 = set()
    for split in ["train", "dev", "test"]:
        filename = "data/generated/EntityQuestions_costomized/{}/queries.jsonl".format(split)
        with open(filename, 'r') as f2:
            for line in f2:
                data = json.loads(line)
                entity_ids_file2.add(data.get('entity_id'))

    # Find common and unique entity_ids
    common_entity_ids = entity_ids_file1.intersection(entity_ids_file2)
    unique_entity_ids = entity_ids_file1.symmetric_difference(entity_ids_file2)
    
    print("Common Entity IDs:", len(common_entity_ids))
    print("Unique Entity IDs:", len(unique_entity_ids))
