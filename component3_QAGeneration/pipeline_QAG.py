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
    results_save_path = os.path.join(args.results_output_dir, args.results_output_filename)
    
    model = TransformersQG(
        language='en',
        model=args.qg_model,
        model_ae=args.ae_model,
        skip_overflow_error=True,
        drop_answer_error_text=True,
    )
    
    with open(args.corpus_path, 'r') as in_file, open(results_save_path, 'w') as out_file:
        for idx, line in enumerate(in_file):
            if idx > 8841:
                passage = json.loads(line.strip())['contents']
                try:
                    with torch.no_grad():
                        qas = model.generate_qa(passage)
                    torch.cuda.empty_cache()
                    
                    print(qas)
                    for (question, answer) in qas:
                        obj = json.dumps({
                            "question": question,
                            "possible_answers": [answer]
                        })
                        out_file.write(obj + '\n')
                except AnswerNotFoundError:
                    print(f"Answer not found for passage: {passage}")
                    continue
                except ExceedMaxLengthError:
                    print(f"Input exceeded max length for passage: {passage}")
                    continue
                except ValueError as e:
                    print(f"For: {passage}")
                    print(str(e))
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qg_model", type=str, required=True)
    parser.add_argument("--ae_model", type=str, required=True)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--results_output_dir", type=str)
    parser.add_argument("--results_output_filename", type=str)
    
    args = parser.parse_args()
    main(args)

