import os
import json
import requests
from datetime import datetime
from urllib.parse import quote
import urllib.request as urllib2

def convert_to_url_format(text):
    text = text.replace(" ", "_")
    text = quote(text)
    text = text.replace("/", "%2F")
    return text

def fetch_wikipedia_pageviews(wiki_title):
    TOP_API_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.{project}/all-access/all-agents/{topic}/monthly/{date_from}/{date_to}"
    lang = 'en'
    project = 'wikipedia'
    # date_from = "2022010100"
    # date_to = "2022123000"
    date_from = "2021010100"
    date_to = "2021103100"
    all_views = 0
    
    edited_title = convert_to_url_format(wiki_title)
    url = TOP_API_URL.format(lang=lang,
                            project = project,
                            topic = edited_title,
                            date_from = date_from,
                            date_to = date_to)
    try:
        resp = urllib2.urlopen(url)
        resp_bytes = resp.read()
        data = json.loads(resp_bytes)
        all_views = sum([item['views'] for item in data['items']]) 
        # print("Target: {:<15}, Views: {}".format(edited_title, all_views))
    except urllib2.HTTPError as e:
        # print(e.code)
        print("Target: {:<20}, does not have a wikipedia page".format(edited_title))
    except urllib2.URLError as e:
        print(e.args)
    
    return all_views

def update_test_files(entity_dir, test_dir):
    """ Update test files with pageviews from Wikipedia. """
    for filename in os.listdir(entity_dir):
        if filename.endswith('.entity.json'):
            relation_id = filename.split('.')[0]
            entity_file_path = os.path.join(entity_dir, filename)
            test_file_path = os.path.join(test_dir, f"{relation_id}.test.json")
            
            print("Processing relation: ", relation_id)
            # Read entity file
            with open(entity_file_path, 'r') as file:
                entities = json.load(file)
            
            # Process each wiki_title in entity file
            if os.path.exists(test_file_path):
                with open(test_file_path, 'r') as file:
                    tests = json.load(file)
                
                for entity in entities:
                    wiki_title = entity['wiki_title']
                    pageviews = fetch_wikipedia_pageviews(wiki_title.replace(' ', '_'))  # Wikipedia titles are underscored
                    
                    # Update corresponding test entry
                    for test in tests:
                        if test['query_id'] == entity['query_id']:
                            test['pageviews'] = pageviews
                
                # Save updated test file
                with open(test_file_path, 'w') as file:
                    json.dump(tests, file, indent=4)

# Set the paths to your directories

dataset_name = 'EQ'
output_dir = 'component0_preprocessing/generated_data/{}_costomized'.format(dataset_name)
test_dir = f"{output_dir}/test" 
entity_dir = f"{output_dir}/entity" 


# Call the function
update_test_files(entity_dir, test_dir)
