import os
import json
from urllib.parse import quote

def convert_to_url_format(text):
    # text = text.replace(" ", "_")
    text = quote(text)
    # text = text.replace("/", "%2F")
    return text

# title = "Darreh Sib"
# print(convert_to_url_format(title))


def update_pageviews(entity_dir, test_dir, freq_file):
    # Load the frequency data from the json file
    with open(freq_file, 'r') as file:
        freq_data = json.load(file)

    # Iterate over the entity files
    all_data = 0
    all_not_found = 0
    for idx, filename in enumerate(os.listdir(entity_dir)):
        
        # if idx == 2:
        #     break
        
        not_found_count = 0
        if filename.endswith('.entity.json'):
            relation_id = filename.split('.')[0]
            entity_path = os.path.join(entity_dir, filename)
            test_path = os.path.join(test_dir, f"{relation_id}.test.json")
            
            print("Processing relation: ", relation_id)
            # Read the entity file
            with open(entity_path, 'r') as file:
                entity_data = json.load(file)
            
            # Read the test file
            with open(test_path, 'r') as file:
                test_data = json.load(file)

            # Create new lists for cleaned data
            new_entity_data = []
            new_test_data = []

            # Update pageviews in the test file based on entity data
            for entity in entity_data:
                wiki_title = convert_to_url_format(entity['wiki_title'])
                # Find the pageviews from freq_data
                if wiki_title in freq_data:
                    new_entity_data.append(entity)
                    for test in test_data:
                        if test['query_id'] == entity['query_id']:
                            test['pageviews'] = freq_data[wiki_title]
                            new_test_data.append(test)
                else:
                    not_found_count += 1
                    # print(f"Pageviews not found for {wiki_title}")

            all_data += len(entity_data)
            all_not_found += not_found_count
            print(f"all entities: {len(entity_data)}")
            print(f"Pageviews not found for {not_found_count} entities")
            
            # Write the updated test data back to the file
            if new_entity_data:
                with open(entity_path, 'w') as file:
                    json.dump(new_entity_data, file, indent=4)
            if new_test_data:
                with open(test_path, 'w') as file:
                    json.dump(new_test_data, file, indent=4)

    print(f"The precentage of pageviews not found: {all_not_found/all_data*100}%")

# Define your directories and frequency file path
dataset_name = 'EQ'
output_dir = 'component0_preprocessing/generated_data/{}_costomized'.format(dataset_name)
test_directory = f"{output_dir}/test" 
entity_directory = f"{output_dir}/entity" 
frequency_file = 'data/dataset/entity_questions_dataset/entity_frequency.json'

# Call the function
update_pageviews(entity_directory, test_directory, frequency_file)



