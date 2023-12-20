import urllib.request as urllib2
from urllib.parse import quote
import json, os, ast
import requests
import re
import wikipediaapi
import pandas as pd

reg_templates = {
    "P17": 'Which\\ country\\ is\\ (.*) located\\ in\\?',
    "P19": "Where\\ was\\ (.*) born\\?",
    "P20": "Where\\ did\\ (.*) die\\?",
    "P26": "Who\\ is\\ (.*) married\\ to\\?",
    "P30": "Which\\ continent\\ is\\ (.*) located\\?",
    "P36": "What\\ is\\ the\\ capital\\ of\\ (.*)\\?",
    "P40": "Who\\ is\\ (.*)'s\\ child\\?",
    "P50": "Who\\ is\\ the\\ author\\ of\\ (.*)\\?",
    "P69": "Where\\ was\\ (.*) educated\\?",
    "P106": "What\\ kind of\\ work\\ does\\ (.*) do\\?",
    "P112": "Who\\ founded\\ (.*)\\?",
    "P127": "Who\\ owns\\ (.*)\\?",
    "P131": "Where\\ is\\ (.*) located\\?",
    "P136": "What\\ type\\ of\\ music\\ does\\ (.*) play\\?",
    "P159": "Where\\ is\\ the\\ headquarter\\ of\\ (.*)?",
    "P170": "Who\\ was\\ (.*) created\\ by\\?",
    "P175": "Who\\ performed\\ (.*)\\?",
    "P176": "Which\\ company\\ is\\ (.*) produced\\ by\\?",
    "P264": "What\\ music\\ label\\ is\\ (.*) represented\\ by\\?",
    "P276": "Where\\ is\\ (.*) located\\?",
    "P407": "Which\\ language\\ was\\ (.*) written\\ in\\?",
    "P413": "What\\ position\\ does\\ (.*) play\\?",
    "P495": "Which\\ country\\ was\\ (.*) created\\ in\\?",
    "P740": "Where\\ was\\ (.*) founded\\?",
    "P800": "What\\ is\\ (.*) famous\\ for\\?",
    "P19_similar": "What is the birthplace of (.*)?",
    "P159_similar": "Where is (.*) headquartered?",
    "P170_similar": "Who is the creator of (.*)?"
}

relation_types = {
    "P17": "country",
    "P19": "place of birth",
    "P20": "place of death",
    "P26": "spouse",
    "P30": "continent",
    "P36": "capital",
    "P40": "child",
    "P50": "author",
    "P69": "educated at",
    "P106": "occupation",
    "P112": "founded by",
    "P127": "owned by",
    "P131": "located in",
    "P136": "genre",
    "P159": "headquarter",
    "P170": "creator",
    "P175": "performer",
    "P176": "manufacturer",
    "P264": "record label",
    "P276": "location",
    "P407": "language",
    "P413": "position",
    "P495": "origin",
    "P740": "formation",
    "P800": "notable work"
}

def convert_to_url_format(text):
    text = text.replace(" ", "_")
    text = quote(text)
    text = text.replace("/", "%2F")
    return text

def get_pageviews(wiki_title):
    TOP_API_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.{project}/all-access/all-agents/{topic}/monthly/{date_from}/{date_to}"
    lang = 'en'
    project = 'wikipedia'
    date_from = "2023010100"
    date_to = "2023121400"
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

def get_wikidata_id(wikipedia_title):
    formatted_title = wikipedia_title.replace(" ", "_")
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": formatted_title,
        "prop": "pageprops"
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Extract page details
    if 'query' in data and 'pages' in data['query']:
        pages = data['query']['pages']
        page_id = next(iter(pages))
        if 'pageprops' in pages[page_id] and 'wikibase_item' in pages[page_id]['pageprops']:
            wikidata_id = pages[page_id]['pageprops']['wikibase_item']
            return wikidata_id
        else:
            return None
    else:
        return None
    
def get_wikipedia_title_from_wikidata(wikidata_id):
    wikidata_url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={wikidata_id}&format=json&props=sitelinks"
    response = requests.get(wikidata_url)
    data = response.json()

    entities = data.get('entities', {})
    if wikidata_id in entities:
        wikipedia_title = entities[wikidata_id].get('sitelinks', {}).get('enwiki', {}).get('title')
        return wikipedia_title
    return None

def get_wikipedia_summary_and_paragraphs(title):
    inappropriate_sections = [
        'References', 'Resources', 'Sources', 'Notes', 'External links', 'See also', 'Further reading',
        'Gallery', 'Ranks', 'Awards', 'Awards and nominations', 'Television', 'Covers and tributes',
        'Filmography', 'Discography', 'Selected discography', 'Bibliography', 'Songs', 'cast', 'Accolades',
        'Selected compositions'
    ]
    wiki_wiki = wikipediaapi.Wikipedia('my_project', 'en')
    page = wiki_wiki.page(title)

    if not page.exists():
        return "Page does not exist", []

    summary = page.summary
    paragraphs = [section.text for section in page.sections if section.title not in inappropriate_sections]
    paragraphs = [item for item in paragraphs if item != ""]

    return summary, paragraphs


if __name__ == "__main__":
    
    ### === Input files & directories
    entity_freq_file = "data/dataset/EntityQuestions/entity_frequency.json"
    data_evidence_path = "data/dataset/EntityQuestions/data_evidence_costomized" 
    relation_temp_file = "data/dataset/EntityQuestions/relation_query_templates.json"
    data_path = 'data/dataset/EntityQuestions'
    
    ### === output files
    queries_file = "data/generated/EntityQuestions_costomized/{}/queries.jsonl"
    corpus_file = "data/generated/EntityQuestions_costomized/{}/corpus.jsonl"
    qrels_file = "data/generated/EntityQuestions_costomized/{}/qrels.jsonl"
    
    
    ### === Read freq file  
    # with open(entity_freq_path, 'r', encoding='utf-8') as file:
    #     data = json.load(file)
    # print(len(data.keys()))
    # for idx, (entity, freq) in enumerate(data.items()):
    #     if idx == 10:
    #         break
    #     print("{}: {}".format(urllib.parse.unquote(entity), freq))
    # wikipedia_title = "Free association (communism and anarchism)"
    # wikidata_id = get_wikidata_id(wikipedia_title)
    # print(wikidata_id)
    # wiki_context = get_wikipedia_summary_and_paragraphs(wikipedia_title)
    # print(wiki_context)
    
    # wikipedia_title = get_wikipedia_title_from_wikidata(wikidata_id)
    # no_wikipedia_page = []
    # # it usually means that there is no direct Wikidata item linked to that specific Wikipedia page
    # if wikipedia_title == None:
    #     no_wikipedia_page.append({title: wikidata_id})
    # print(wikipedia_title)
    
    
    ### === Create Query file by API ===========
    ### === First try and wrong ================
    # with open(relation_temp_file, 'r', encoding='utf-8') as file:
    #     relation_template = json.load(file)
     
    # deleted_entities = {}
    # for split in (['test', 'dev', 'train',]):
    #     deleted_entities[split] = []
    #     split_path = os.path.join(data_path, split)
        
    #     with open(queries_file.format(split), 'w') as jsonl:
        
    #         for filename in os.listdir(split_path):
    #             relation_code = filename.split('.')[0]
                
    #             file_path = os.path.join(split_path, filename)
    #             if os.path.isfile(file_path):
    #                 print(f"Found file: {file_path}")
    #                 with open(file_path, 'r', encoding='utf-8') as file:
                        
    #                     reg_template = reg_templates[relation_code]
    #                     data = json.load(file)
    #                     for idx, query_obj in enumerate(data):
                            
    #                         if (idx != 0) and (idx % 200 == 0):
    #                             print('{} processed'.format(idx))
                            
    #                         question = query_obj['question']
    #                         possible_answers = query_obj['answers']
    #                         wikidata_label = re.findall(reg_template, query_obj['question'])[0]
    #                         entity_id = get_wikidata_id(wikidata_label)
    #                         relation_type = relation_types[relation_code]
                            
    #                         wiki_title = get_wikipedia_title_from_wikidata(entity_id)
    #                         if wiki_title != None:
    #                             # print('wikiID: {}, title: {}'.format(entity_id, wiki_title))
    #                             pageviews = get_pageviews(wiki_title)
    #                             temp = {
    #                                 "entity_id": entity_id,
    #                                 'relation_type': relation_type,
    #                                 "pageviews": pageviews,
    #                                 "question": question,
    #                                 "possible_answers": possible_answers
    #                             }
    #                             jsonl.write(json.dumps(temp) + '\n')
                            
    #                         else:
    #                             deleted_entities[split].append(entity_id)
    
    
    ### === Create Query file by API ===========
    ### === Second try =========================
    with open(relation_temp_file, 'r', encoding='utf-8') as file:
        relation_template = json.load(file)
    
    
    for split in (['train', 'test', 'dev']): 
        data_evidence_path = "data/dataset/EntityQuestions/data_evidence_costomized"
        split_path = os.path.join(data_evidence_path, split)
        
        with open(queries_file.format(split), 'w') as jsonl:
        
            for filename in os.listdir(split_path):
                relation_code = filename.split('.')[0]
                
                file_path = os.path.join(split_path, filename)
                if os.path.isfile(file_path):
                    print(f"Found file: {file_path}")
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                        
                        reg_template = reg_templates[relation_code]
                        data = json.load(file)
                        for idx, query_obj in enumerate(data):
                            
                            # if idx == 10:
                            #     break
                            
                            if (idx != 0) and (idx % 200 == 0):
                                print('{} processed'.format(idx))
                            
                            question = query_obj['question']
                            possible_answers = query_obj['answers']
                            # wikidata_label = re.findall(reg_template, query_obj['question'])[0]
                            wikidata_id = query_obj['evidence'][0]['subject']["uri"]
                            # entity_id = get_wikidata_id(wikidata_label)
                            relation_type = relation_types[relation_code]
                            
                            wiki_title = get_wikipedia_title_from_wikidata(wikidata_id)
                            # print('wikiID: {}, title: {}'.format(wikidata_id, wiki_title))
                            if wiki_title != None:
                                
                                pageviews = get_pageviews(wiki_title)
                                temp = {
                                    "entity_id": wikidata_id,
                                    'relation_type': relation_type,
                                    "pageviews": pageviews,
                                    "question": question,
                                    "possible_answers": possible_answers
                                }
                                jsonl.write(json.dumps(temp) + '\n')
    