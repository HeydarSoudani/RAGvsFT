import json, os
import pandas as pd
import matplotlib.pyplot as plt

# RELATIONS = {
#     '22': 'Occupation',
#     '91': 'Genre',
#     '97': 'Capital of',
#     '106': 'Religion',
#     '164': 'Producer',
#     '182': 'Country',
#     '218': 'Place of birth',
#     '257': 'Father',
#     '292': 'Mother',
#     '422': 'Capital',
#     '472': 'Color',
#     '484': 'Author',
#     '526': 'Director',
#     '533': 'Screenwriter',
#     '560': 'Sport',
#     '639': 'Composer'
# }
RELATIONS = {
    '17': 'country',
    '19': 'place of birth',
    '22': 'father',
    '25': 'mother',
    '27': 'country of citizenship',
    '36': 'capital',
    '50': 'author',
    '57': 'director',
    '58': 'screenwriter',
    '69': 'educated at',
    '86': 'composer',
    '106': 'occupation',
    '123': 'publisher',
    '136': 'genre',
    '140': 'religion',
    '149': 'architectural style',
    '162': 'producer',
    '184': 'doctoral advisor',
    '344': 'director of photography',
    '452': 'industry',
    '462': 'color',
    '641': 'sport',
    '674': 'characters',
    '1038': 'relative',
    '1050': 'medical condition',
    '1376': 'capital of',
    '1431': 'executive producer',
    '1433': 'published in',
    '2012': 'cuisine',
    '2936': 'language used',
    '3301': 'broadcast by',
    '4647': 'location of first performance'
}

def incorrect_per_relation():
    # file_path = 'component0_preprocessing/generated_data/popQA_costomized/results/slms/popQA_stable_lm2_af_rag_ideal_peft_results.jsonl'
    file_path = 'analysis/on_false_results/two_sided_partial_match_witqa.jsonl'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)

    df['relation_id'] = df['query_id'].apply(lambda x: x.split('_')[0])
    df['relation_name'] = df['relation_id'].map(RELATIONS)
    df_incorrect = df[df['is_correct'] == False]

    relation_counts = df_incorrect['relation_name'].value_counts(normalize=True) * 100

    plt.figure(figsize=(10, 6))
    relation_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Relation ID')
    plt.ylabel('Percentage')
    plt.title('Percentage of Each Relation in Incorrect Predictions')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

def write_incorrect_data_per_relation():
    
    # input_file_path = 'component0_preprocessing/generated_data/popQA_costomized/results/slms/popQA_stable_lm2_af_rag_ideal_peft_results.jsonl'
    # output_file_path = 'analysis/on_false_results/incorrect_results_by_relation.json'    
    
    input_file_path = 'analysis/on_false_results/two_sided_partial_match_witqa.jsonl'
    output_file_path = 'analysis/on_false_results/incorrect_results_by_relation_2side_witqa.json'
    
    incorrect_results = {relation: [] for relation in RELATIONS.values()}
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            sample = json.loads(line)
            if not sample['is_correct']:
                relation_id = sample['query_id'].split('_')[0]
                relation_name = RELATIONS.get(relation_id)
                if relation_name:
                    incorrect_results[relation_name].append(sample)
    
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(incorrect_results, outfile, indent=4)

def finetuning_diff():
    w_ft_file = 'component0_preprocessing/generated_data/popQA_costomized/results_two_side/slms/popQA_MiniCPM_af_rag_ideal_peft_results.jsonl'
    wo_ft_file = 'component0_preprocessing/generated_data/popQA_costomized/results_two_side/slms/popQA_MiniCPM_bf_rag_ideal_full_results.jsonl'
    output_file = 'analysis/on_false_results/compare_ft_wo_ft.json'
    
    # === Load retrieved pasage
    ret_results = {}
    ret_results_dir = "component0_preprocessing/generated_data/popQA_costomized/retrieved/ideal"
    for filename in os.listdir(ret_results_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(ret_results_dir, filename)
            print(f"Processing file: {file_path}")
            with open (file_path, 'r') as file:
                for line in file:
                    data = json.loads(line.strip())
                    ret_results[data['id']] = data
    
    with open(w_ft_file, 'r') as file:
        lines = file.readlines()
    w_ft_data = [json.loads(line.strip()) for line in lines]

    with open(wo_ft_file, 'r') as file:
        lines = file.readlines()
    wo_ft_data = [json.loads(line.strip()) for line in lines]
    wo_ft_dict = {item['query_id']: item for item in wo_ft_data}
    
    results_by_relation = {}
    for item in w_ft_data:
        if item['is_correct'] and not wo_ft_dict[item['query_id']]['is_correct']:
            relation_id = item['query_id'].split('_')[0]
            
            if RELATIONS[relation_id] not in results_by_relation:
                results_by_relation[RELATIONS[relation_id]] = []
            
            results_by_relation[RELATIONS[relation_id]].append({
                'query_id': item['query_id'],
                'question': item['question'],
                'possible_answers': item['possible_answers'],
                'pageviews': item['pageviews'],
                'ret_context': ret_results[item['query_id']]['ctxs'][0]['text'],
                'ft_pred': item['pred'],
                'wo_ft_pred': wo_ft_dict[item['query_id']]['pred'],
            })
    
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(results_by_relation, outfile, indent=4)
    
def get_relevant_qa_pairs():

    # Load JSON files
    def load_json(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    
    dataset = 'witQA'
    
    relation_ids = list(RELATIONS.keys())
    for relation_id in relation_ids:
    # relation_id = '526'
        base_dir = f'component0_preprocessing/generated_data/{dataset}_costomized'
        q_test = load_json(f'{base_dir}/test/{relation_id}.test.json')
        qrel = load_json(f'{base_dir}/qrels_all/{relation_id}.qrels.json')
        doc = load_json(f'{base_dir}/corpus_all/{relation_id}.corpus.json')
        qrel_train = load_json(f'{base_dir}/prompting/qrels-train/{relation_id}.qrels-train.json')
        q_train = load_json(f'{base_dir}/prompting/train/{relation_id}.train.json')

        output_path = f"{base_dir}/retrieved_qa_pairs/ideal/{relation_id}.retrieved_qa_pairs.jsonl"
        
        os.makedirs(f'{base_dir}/retrieved_qa_pairs', exist_ok=True)
        os.makedirs(f'{base_dir}/retrieved_qa_pairs/ideal', exist_ok=True)
        
        qrel_mapping = {}
        for entry in qrel:
            query_id = entry['query_id']
            doc_id = entry['doc_id']
            if query_id not in qrel_mapping:
                qrel_mapping[query_id] = []
            qrel_mapping[query_id].append(doc_id)

        qrel_train_mapping = {}
        for entry in qrel_train:
            doc_id = entry['doc_id']
            query_id = entry['query_id']
            if doc_id not in qrel_train_mapping:
                qrel_train_mapping[doc_id] = []
            qrel_train_mapping[doc_id].append(query_id)

        q_train_dict = {entry['query_id']: entry for entry in q_train}

        results = []
        for test_query in q_test:
            test_query_id = test_query['query_id']
            relevant_docs = qrel_mapping.get(test_query_id, [])
            relevant_train_queries = []
            for doc_id in relevant_docs:
                relevant_train_queries.extend(qrel_train_mapping.get(doc_id, []))
            relevant_train_questions = [q_train_dict[query_id] for query_id in relevant_train_queries if query_id in q_train_dict]
            results.append({
                'query_id': test_query_id,
                'question': test_query['question'],
                'relevant_train_questions': relevant_train_questions
            })

        with open(output_path, 'w', encoding='utf-8') as outfile:
            for result in results:
                outfile.write(json.dumps(result) + '\n')


def main():
    # incorrect_per_relation()
    # write_incorrect_data_per_relation()
    # finetuning_diff()
    get_relevant_qa_pairs()


if __name__ == '__main__':
    main()
    