import csv, os, json

def main():
    # corpus -> jsonl {"_id", "title", "text"}
    # queries -> jsonl {"_id", "text"}
    # qrels -> tsv {"quesry_id", "corpus_id", "score"}
    
    # Input files
    dataset_name = "popQA" # [popQA, witQA, EQ]
    dataset_dir = "component0_preprocessing/generated_data/{}_costomized".format(dataset_name)
    corpus_dir = f"{dataset_dir}/corpus_all"
    qrels_dir = f"{dataset_dir}/qrels_all"
    queries_dir = f"{dataset_dir}/test"
    
    # Output files  
    output_dataset_dir = "component1_retrieval/data/{}".format(dataset_name)
    output_corpus_file = f"{output_dataset_dir}/corpus.jsonl"
    output_queries_file = f"{output_dataset_dir}/queries.jsonl"
    output_qrels_dir = f"{output_dataset_dir}/qrels"
    output_qrels_file = f"{output_qrels_dir}/test.tsv"
    
    os.makedirs(output_dataset_dir, exist_ok=True)
    os.makedirs(output_qrels_dir, exist_ok=True)
    
    # Combine all corpus files into one
    if not corpus_dir.endswith('/'):
        corpus_dir += '/'
    with open(output_corpus_file, 'w') as outfile:
        for filename in os.listdir(corpus_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(corpus_dir, filename)
                with open(file_path, 'r') as infile:
                    data = json.load(infile)
                    
                    for item in data:
                        new_json = {
                            "_id": item["doc_id"],
                            "title": item["title"],
                            "text": item["content"]
                        }
                        json.dump(new_json, outfile)
                        outfile.write('\n')
    
    # Combine all quesries files into one
    if not queries_dir.endswith('/'):
        queries_dir += '/'
    with open(output_queries_file, 'w') as outfile:
        for filename in os.listdir(queries_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(queries_dir, filename)
                with open(file_path, 'r') as infile:
                    data = json.load(infile)
                    
                    for item in data:
                        new_json = {
                            "_id": item["query_id"],
                            "text": item["question"],
                            "pageviews": item["pageviews"]
                        }
                        json.dump(new_json, outfile)
                        outfile.write('\n')
    
    # Combine all qrels files into one
    headers = set()
    all_data = []
    for filename in os.listdir(qrels_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(qrels_dir, filename)
            with open(file_path, 'r') as infile:
                data = json.load(infile)
                for item in data:
                    all_data.append(item)
                    headers.update(item.keys())
    headers = sorted(list(headers))
    
    with open(output_qrels_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for item in all_data:
            writer.writerow(item)
    
    
    
    
if __name__ == "__main__":
    main()
