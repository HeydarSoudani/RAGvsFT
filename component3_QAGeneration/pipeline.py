
import argparse, json, os
from lmqg import TransformersQG


def main(args):
    
    if not os.path.exists(args.results_output_dir):
        os.makedirs(args.results_output_dir)
    results_save_path = os.path.join(args.results_output_dir, args.results_output_filename)
    
    model = TransformersQG(
        model=args.qg_model,
        model_ae=args.ae_model
    )
    
    with open(args.corpus_path, 'r') as in_file, open(results_save_path, 'w') as out_file:
        for idx, line in enumerate(in_file):
            if idx == 100:
                break
            passage = json.loads(line.strip())['contents']
            qas = model.generate_qa(passage)
            print(qas)
            for (question, answer) in qas:
                obj = json.dumps({
                    "question": question,
                    "possible_answers": [answer] 
                })
                out_file.write(obj + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qg_model", type=str, required=True)
    parser.add_argument("--ae_model", type=str, required=True)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--results_output_dir", type=str)
    parser.add_argument("--results_output_filename", type=str)
    
    args = parser.parse_args()
    main(args)

