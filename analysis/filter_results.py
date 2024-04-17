import os
import json

def update_results_from_tests(test_dir, results_file, output_dir):
    # Dictionary to store query_id and new pageviews from test files
    test_data = {}

    # Read all test files and collect query IDs and their respective pageviews
    for filename in os.listdir(test_dir):
        if filename.endswith('.test.json'):
            with open(os.path.join(test_dir, filename), 'r') as file:
                data = json.load(file)
                for item in data:
                    test_data[item['query_id']] = item['pageviews']

    
    if results_file.endswith('.jsonl'):
        # Open the results.jsonl file and filter/update entries
        updated_results = []
        with open(results_file, 'r') as file:
            for line in file:
                result = json.loads(line.strip())
                query_id = result['query_id']
                if query_id in test_data:
                    # Update pageviews and add to the results to keep
                    result['pageviews'] = test_data[query_id]
                    updated_results.append(result)

        # Write the updated results to a new file
        output_file_path = f"{output_dir}/{results_file.split('/')[-1]}"
        with open(output_file_path, 'w') as file:
            for result in updated_results:
                file.write(json.dumps(result) + '\n')

# Define your directories and files
dataset_name = 'EQ'
base_dir = 'component0_preprocessing/generated_data/{}_costomized'.format(dataset_name)
test_directory = f"{base_dir}/test" 
results_file = f'{base_dir}/results/EQ_tiny_llama_af_norag_peft_results.jsonl'
output_directory = f'{base_dir}/results/slms'

# os.makedirs('component0_preprocessing/generated_data/EQ_costomized/retrieved', exist_ok=True)
# os.makedirs('component0_preprocessing/generated_data/EQ_costomized/retrieved/bm25', exist_ok=True)


# Call the function
update_results_from_tests(test_directory, results_file, output_directory)








# def update_ret_results_from_tests(test_dir, results_dir, output_dir):
    
#     # Process each results file in the results directory
#     for results_file in os.listdir(results_dir):
#         if results_file.endswith('.ret_results.jsonl'):
#             relation_id = results_file.split('.')[0]
#             test_file_path = os.path.join(test_dir, f"{relation_id}.test.json")
#             results_file_path = os.path.join(results_dir, results_file)
#             output_file_path = os.path.join(output_dir, results_file)

#             if os.path.exists(test_file_path):
#                 # Load test data for the current relation
#                 with open(test_file_path, 'r') as file:
#                     test_data = json.load(file)
#                 test_data_map = {item['query_id']: item['pageviews'] for item in test_data}

#                 # Open the results file and filter/update entries
#                 updated_results = []
#                 with open(results_file_path, 'r') as file:
#                     for line in file:
#                         result = json.loads(line.strip())
#                         query_id = result['id']
#                         if query_id in test_data_map:
#                             # Update pageviews and add to the results to keep
#                             # result['pageviews'] = test_data_map[query_id]
#                             updated_results.append(result)

#                 # Write the updated results to a new file in the output directory
#                 with open(output_file_path, 'w') as file:
#                     for result in updated_results:
#                         file.write(json.dumps(result) + '\n')

# # Define your directories
# dataset_name = 'EQ'
# ret_method = 'rerank'
# output_dir = 'component0_preprocessing/generated_data/{}_costomized'.format(dataset_name)
# test_directory = f"{output_dir}/test" 
# results_directory = f'component0_preprocessing/generated_data/EQ_costomized/retrieved_base/{ret_method}'
# output_directory = f'component0_preprocessing/generated_data/EQ_costomized/retrieved/{ret_method}'

# os.makedirs('component0_preprocessing/generated_data/EQ_costomized/retrieved', exist_ok=True)
# os.makedirs(f'component0_preprocessing/generated_data/EQ_costomized/retrieved/{ret_method}', exist_ok=True)

# # Call the function
# update_ret_results_from_tests(test_directory, results_directory, output_directory)








