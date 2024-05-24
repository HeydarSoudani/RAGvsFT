#!/usr/bin/env python3

import argparse

def summary_generation_for_retrieved_context(args):
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # === Retrieved context ===
    ret_results = {}
    ret_results_dir = f"{args.data_dir}/retrieved/{args.retrieval_method}"
    
    for test_relation_id in test_relation_ids:
        ret_results_path = f"{ret_results_dir}/{test_relation_id}.{args.retrieval_method}.ret_results.jsonl"
        with open (ret_results_path, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                ret_results[data['id']] = data
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    summary_generation_for_retrieved_context(args)