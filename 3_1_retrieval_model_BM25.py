from pyserini.index import IndexReader, IndexCollection
from pyserini.search import SimpleSearcher
import json

def convert_corpus_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            data = json.loads(line)
            new_data = {
                'id': data['corpus_id'],
                'contents': data['text']
            }
            output_file.write(json.dumps(new_data) + '\n')

    print(f"Conversion complete. Output file created at: {output_file_path}")

def load_queries(file_path):
    queries = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            queries[data['entity_id']] = data['question']
    return queries

def load_qrels(file_path):
    qrels = {}
    with open(file_path, 'r') as file:
        for line in file:
            query_id, _, corpus_id, score = line.split()
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][corpus_id] = int(score)
    return qrels

def search_queries(index_path, queries, k):
    searcher = SimpleSearcher(index_path)
    results = {}
    for query_id, query in queries.items():
        hits = searcher.search(query, k)
        results[query_id] = [(hit.docid, hit.score) for hit in hits]
    return results

def write_search_result(output_file_path, search_results):
    with open(output_file_path, 'w') as output_file:
        for query_id, docs in search_results.items():
            for doc_id, score in docs:
                output = {
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'score': score
                }
                output_file.write(json.dumps(output) + '\n')

    print(f"Results written to {output_file_path}")

if __name__ == "__main__":
    corpus_file = 'data/generated/popQA_costomized/corpus.jsonl'
    queries_file = 'data/generated/popQA_costomized/queries.jsonl'
    qrels_file = 'data/generated/popQA_costomized/qrels.jsonl'
    
    corpus_pyserini_file = 'data/generated/popQA_costomized/corpus_pyserini.jsonl'
    corpus_pyserini_index_path = 'data/generated/popQA_costomized/index'
    output_search_result_path = 'data/generated/bm25_search_results.jsonl'
    
    convert_corpus_file(corpus_file, corpus_pyserini_file)
    
    config = {
        'input': corpus_pyserini_file,
        'collection': 'JsonCollection',
        'generator': 'DefaultLuceneDocumentGenerator',
        'threads': 8,
        'index': corpus_pyserini_index_path,
        'storePositions': True,
        'storeDocvectors': True,
        'storeRaw': True
    }
    IndexCollection(config)
    
    queries = load_queries(queries_file)
    qrels = load_qrels(qrels_file)
    top_k = 5
    search_results = search_queries(corpus_pyserini_index_path, queries, top_k)
    write_search_result(output_search_result_path, search_results)
    
    
    # Evaluation
    