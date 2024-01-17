#!/usr/bin/env python3

import torch
import random
import argparse, json, os
from lmqg import TransformersQG
from lmqg.exceptions import AnswerNotFoundError, ExceedMaxLengthError

random.seed(0)
torch.manual_seed(0)


def main(args):
    
    if not os.path.exists(args.results_output_dir):
        os.makedirs(args.results_output_dir)
    
    queries_gen_path = os.path.join(args.results_output_dir, 'gen_queries.jsonl')
    qrels_gen_path = os.path.join(args.results_output_dir, 'gen_qrels.jsonl')
    
    model = TransformersQG(
        language='en',
        model=args.qg_model,
        model_ae=args.ae_model,
        skip_overflow_error=True,
        drop_answer_error_text=True,
    )
    
    with open(args.corpus_path, 'r') as in_file, open(queries_gen_path, 'w') as query_file, open(qrels_gen_path, 'w') as qrel_file:
        
        query_id_counter = 1
        for idx, line in enumerate(in_file):
            corpus_data = json.loads(line.strip())
            
            if idx == 2:
                break
            
            try:
                with torch.no_grad():
                    qas = model.generate_qa(corpus_data['contents'])
                torch.cuda.empty_cache()
                
                print(qas)
                for (question, answer) in qas:
                    qobj = json.dumps({
                        "query_id": "GQ_" + str(query_id_counter),
                        "question": question,
                        "possible_answers": [answer]
                    })
                    qrel_obj = json.dumps({
                        "query_id": "GQ_" + str(query_id_counter),
                        "doc_id": corpus_data["id"],
                        "score": 1
                    })
                    query_id_counter +=1
                    
                    query_file.write(qobj + '\n')
                    qrel_file.write(qrel_obj + '\n')
                    
            except AnswerNotFoundError:
                print(f"Answer not found for passage: {corpus_data['contents']}")
                continue
            except ExceedMaxLengthError:
                print(f"Input exceeded max length for passage: {corpus_data['contents']}")
                continue
            except ValueError as e:
                print(f"For: {corpus_data['contents']}")
                print(str(e))
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qg_model", type=str, required=True)
    parser.add_argument("--ae_model", type=str, required=True)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--results_output_dir", type=str)
    
    args = parser.parse_args()
    main(args)

