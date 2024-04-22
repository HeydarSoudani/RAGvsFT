import json
import os



# === Datasets variables ========================
dataset_name = 'EQ' # [popQA, witQA, EQ]
retrieval_models = ["bm25", "contriever", "rerank", "dpr"]
gen_models = [
    "flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl",
    "stable_lm2", "tiny_llama", "MiniCPM",
    "llama2", "mistral", "zephyr"
]
dataset_dir = 'component0_preprocessing/generated_data/{}_costomized'.format(dataset_name)
test_dir = f"{dataset_dir}/test"

model_name = gen_models[5]

if model_name in ["flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl"]:
    model_type = 'flant5'
elif model_name in ["stable_lm2", "tiny_llama", "MiniCPM"]:
    model_type = 'slms'
elif model_name in ["llama2", "mistral", "zephyr"]:
    model_type = 'llms'

base_path  = "component0_preprocessing/generated_data"
retrieval_model = 'ideal'
results_files = [
        {"id": 1, "title": "NoFT/NoRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_bf_norag_full_results.jsonl"},
        {"id": 2, "title": f"NoFT/idealRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_bf_rag_{retrieval_model}_full_results.jsonl"},
        {"id": 3, "title": "FT/NoRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_af_norag_peft_results.jsonl"},
        {"id": 4, "title": f"FT/idealRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_af_rag_{retrieval_model}_peft_results.jsonl"},
        # {"title": f"voting", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_voting_results.jsonl"},
        # {"title": f"voting_2", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_voting_2_results.jsonl"},
        # {"title": f"NoFT/bm25RAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_bf_rag_bm25_full_results.jsonl"},
        # {"title": f"NoFT/dprRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_bf_rag_dpr_full_results.jsonl"},
        # {"title": f"FT/dprRAG", "filename": f"{base_path}/{dataset_name}_costomized/results/{dataset_name}_{model_name}_af_rag_dpr_peft_results.jsonl"},
    ]

output_file = f"{base_path}/{dataset_name}_costomized/results/{model_type}/{dataset_name}_{model_name}_moe_ideal_results.jsonl"

def load_results_data(results_files):
    results_data = {}
    for result_file in results_files:
        file_id = result_file['id']
        with open(result_file['filename'], 'r') as file:
            for line in file:
                result = json.loads(line)
                query_id = result['query_id']
                result_obj = {
                    "file_id": file_id,
                    "result": result
                }
                
                if query_id in results_data:
                    results_data[query_id].append(result_obj)
                else:
                    results_data[query_id] = [result_obj]
    return results_data

def main():
    
    results_data = load_results_data(results_files)

    with open(output_file, 'w') as output:
        
        # Iterate over each file in the test directory
        for filename in os.listdir(test_dir):
            if filename.endswith('.test.json'):
                test_file_path = os.path.join(test_dir, filename)
                relation_id = filename.split('.')[0]
                print(f"Processing relation: {relation_id} ...")
                with open(test_file_path, 'r') as file:
                    test_queries = json.load(file)
                
                for test_data in test_queries:
                    query_id = test_data['query_id']
                    question = test_data['question']
                    answers = test_data['answers']
                    pageviews = test_data['pageviews']
                    files_with_correct = []
                    
                    query_results = results_data.get(query_id, [])
                    
                    for result in query_results:
                        file_id = result['file_id']
                        if result['result']['is_correct']:
                            files_with_correct.append(file_id)
                        
                        # with open(result_file['filename'], 'r') as file:
                        #     for line in file:
                        #         result_data = json.loads(line)
                        #         if result_data['query_id'] == query_id and result_data['is_correct']:
                        #             files_with_correct.append(file_id)
                        #             break
                        
                    is_correct = len(files_with_correct) != 0 
                
                    output.write(json.dumps({
                        "query_id": query_id,
                        "question": question,
                        "answers": answers,
                        "pageviews": pageviews,
                        "is_correct": is_correct,
                        "files_with_correct": files_with_correct
                    }) + '\n')
                                    


if __name__ == "__main__":
    main()