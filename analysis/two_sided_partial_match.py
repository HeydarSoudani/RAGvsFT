import json, os

# === Datasets variables ========================
dataset_name = 'EQ' # [popQA, witQA, EQ]
model_idx = 10
result_idx = 4

dataset_dir = 'component0_preprocessing/generated_data/{}_costomized'.format(dataset_name)

gen_models = [
    "flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl",
    "tiny_llama", "stable_lm2", "MiniCPM",
    "llama2", "mistral", "zephyr"
]
model_name = gen_models[model_idx]
    
if model_name in ["flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl"]:
    model_type = 'flant5'
elif model_name in ["stable_lm2", "tiny_llama", "MiniCPM"]:
    model_type = 'slms'
elif model_name in ["llama2", "mistral", "zephyr"]:
    model_type = 'llms'

retrieval_model = 'ideal'

result_type = result_idx # [1, 2, 3, 4]
if result_type == 1:
    result_text = 'bf_norag_full'
elif result_type == 2:
    result_text = 'af_norag_peft'
elif result_type == 3:
    result_text = 'bf_rag_ideal_full'
elif result_type == 4:
    result_text = f'af_rag_{retrieval_model}_peft'

input_file = f"{dataset_dir}/results/{model_type}/{dataset_name}_{model_name}_{result_text}_results.jsonl"

output_dir = f"{dataset_dir}/results_two_side/"
output_file = f"{output_dir}/{model_type}/{dataset_name}_{model_name}_{result_text}_results.jsonl"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/{model_type}", exist_ok=True)



input_file = 'component0_preprocessing/generated_data/popQA_costomized/results/popQA_MiniCPM_5pcent_h_0r_0p_bf_rag_dpr_peft_results.jsonl'
output_file = 'component0_preprocessing/generated_data/popQA_costomized/results/popQA_MiniCPM_5pcent_h_0r_0p_bf_rag_dpr_peft_results_1side.jsonl'

def main():
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    accuracy = []
    with open(output_file, 'w', encoding='utf-8') as of:
        for item in data:
            possible_answers = item['possible_answers']
            pred = item['pred']
            
            is_correct = False
            for pa in possible_answers:
                if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                    is_correct = True
                    break
                
            # pred_parts = pred.split()
            # for pa in possible_answers:
            #     for part in pred_parts:
            #         if part in pa or part.lower() in pa or part.capitalize() in pa:
            #             is_correct = True
            #             break  
            
            accuracy.append(is_correct)
            
            new_item = {
                "query_id": item['query_id'],
                "question": item['question'],
                "possible_answers": item['possible_answers'],
                "pred": pred,
                "is_correct": is_correct,
                "has_context": item['has_context'],
                "pageviews": item['pageviews'],
            }
            of.write(json.dumps(new_item) + '\n')
    
    acc = sum(accuracy) / len(accuracy)
    print(f"Accuracy: {acc * 100:.2f}%")
        


if __name__ == '__main__':
    main()