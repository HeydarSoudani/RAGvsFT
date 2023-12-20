import csv
import json

def get_column_from_tsv(tsv_file, column_index):
    column_data = []

    with open(tsv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')  # Specify the delimiter as tab
        for row in reader:
            # Check if the row has enough columns
            if len(row) > column_index:
                column_data.append(row[column_index])
    return column_data

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_json_file_and_return_list(file_path):
    json_objects = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as JSON
            try:
                json_object = json.loads(line)
                # Append the JSON object to the list
                json_objects.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

    return json_objects

def write_to_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)

def overlap_words(list1, list2):
    # Convert lists to sets for faster intersection operation
    set1 = set(list1)
    set2 = set(list2)

    # Find the overlapping words
    overlap = set1.intersection(set2)

    return overlap

if __name__ == "__main__":
    entities_lists = {}

    # ### == Get PopQA entities ========
    # tsv_file_path = './data/popQA.tsv'
    # column_index_to_extract = 1 
    # popqa_data = list(set(get_column_from_tsv(tsv_file_path, column_index_to_extract)))[1:]
    # print("PopQA words:", len(popqa_data))
    # entities_lists["popqa"] = popqa_data


    # ### == Get Compmix entities ======
    # train_path = './data/Compmix_QA/compmix_trainset.json'
    # dev_path = './data/Compmix_QA/compmix_devset.json'
    # test_path = './data/Compmix_QA/compmix_testset.json'
    # train_list = load_json_file(train_path)
    # dev_list = load_json_file(dev_path)
    # test_list = load_json_file(test_path)
    # concatenated_list = train_list + dev_list + test_list
    # compmix_data = list(set([obj["entities"][0]["label"] for obj in concatenated_list]))
    # print("Compmix words:", len(compmix_data))
    # entities_lists["compmix"] = compmix_data


    # ### == Get Convmix entities ========
    # train_path = './data/Convmix_ConvQA/trainset/train_set_ALL.json'
    # dev_path = './data/Convmix_ConvQA/devset/dev_set_ALL.json'
    # test_path = './data/Convmix_ConvQA/testset/test_set_ALL.json'
    # train_list = load_json_file(train_path)
    # dev_list = load_json_file(dev_path)
    # test_list = load_json_file(test_path)
    # concatenated_list = train_list + dev_list + test_list

    # entities_list = []
    # for obj in concatenated_list:
    #     for question in obj["questions"]:
    #         entities_list.extend([entity["label"] for entity in question["entities"]])
    # convmix_data = list(set(entities_list))
    # print("Convmix words:", len(convmix_data))
    # entities_lists["convmix"] = convmix_data

    # data = load_json_file('./analysis/data/entities_list.json')
    # data["convmix"] = convmix_data
    # write_to_json_file('./analysis/data/entities_list.json', data)


    # ### == Get TGConv entities ======
    # train_path = './analysis/data/tgconv/train_concepts_nv.json'
    # dev_path = './analysis/data/tgconv/dev_concepts_nv.json'
    # test_path = './analysis/data/tgconv/test_concepts_nv.json'
    # train_list = load_json_file(train_path)
    # dev_list = load_json_file(dev_path)
    # test_list = load_json_file(test_path)
    # concatenated_list = train_list + dev_list + test_list
    # tgconv_hard_tgt_data = list(set([obj["hard_target"] for obj in concatenated_list]))
    # tgconv_easy_tgt_data = list(set([obj["easy_target"] for obj in concatenated_list]))
    # print("TGConv (easy) words:", len(tgconv_easy_tgt_data))
    # print("TGConv (hard) words:", len(tgconv_hard_tgt_data))
    # entities_lists["tgconv_easy"] = tgconv_easy_tgt_data
    # entities_lists["tgconv_hard"] = tgconv_hard_tgt_data


    # ### == Get WoW entities ========
    # topic_list = []
    # with open('./analysis/data/WoW/topic_splits.json', "r") as f:
    #     a = json.load(f)
    #     for item in a.keys():
    #         topic_list.extend(a[item]) 
    
    # wow_data = list(set(topic_list))
    # print("WoW words:", len(wow_data))
    # entities_lists["wow"] = wow_data
    

    ### == ECQG =====================
    train_path = './data/ECQG/train.json'
    dev_path = './data/ECQG/val.json'
    test_path = './data/ECQG/test.json'
    train_list = read_json_file_and_return_list(train_path)
    dev_list = read_json_file_and_return_list(dev_path)
    test_list = read_json_file_and_return_list(test_path)
    concatenated_list = train_list + dev_list + test_list
    ECQG_data = list(set([obj["entity"] for obj in concatenated_list]))
    print(ECQG_data)
    


    data = load_json_file('./data/entities_list.json')
    data["ECQG"] = ECQG_data
    write_to_json_file('./data/entities_list.json', data)
    
    ### == Check the overlap entities 
    # common_words = list(overlap_words(compmix_data, popqa_data))
    # print("Overlapping words (CompMix & PopQA):", len(common_words))

    
    ### == Write the lists of words to the JSON file
    # write_to_json_file('./analysis/data/entities_list.json', entities_lists)

