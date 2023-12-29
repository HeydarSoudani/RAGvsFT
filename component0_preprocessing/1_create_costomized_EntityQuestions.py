import urllib.request as urllib2
from urllib.parse import quote
import json, os
import requests
import wikipediaapi



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
    "P800": "notable work",
    "a":"b"
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

def create_queries_file(data_evidence_path, queries_file):
    relation_temp_file = "data/dataset/EntityQuestions/relation_query_templates.json"
    with open(relation_temp_file, 'r', encoding='utf-8') as file:
        relation_template = json.load(file)
    
    for split in (['train', 'test', 'dev']): 
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

def create_corpus_qrels_files(queries_file, corpus_file, qrels_file):
   
   corpus_id_counter = 2708
   for split in (['train']):  # 'test', 'dev'
        
        split_queries_path = queries_file.format(split)
        split_corpus_path = corpus_file.format(split)
        split_qrels_path = qrels_file.format(split)
        
        with open(split_queries_path, 'r') as queries, open(split_corpus_path, 'w') as corpus, open(split_qrels_path, 'w') as qrels:
            for idx, line in enumerate(queries):
                if (idx+1)%300 == 0:
                    print("# processed queries:", idx+1)
                
                # if idx == 40:
                #     break
                
                data = json.loads(line.strip())
                wikidata_id = data['entity_id']
                wikipedia_title = get_wikipedia_title_from_wikidata(wikidata_id)
                
                if wikipedia_title:
                    summary, paragraphs = get_wikipedia_summary_and_paragraphs(wikipedia_title)
                    
                    corpus_data1 = {
                        'id': str(corpus_id_counter),
                        'contents': summary
                    }
                    qrels_data1 = {
                        'query_id': wikidata_id,
                        'doc_id': str(corpus_id_counter),
                        'score': 1
                    }
                    corpus_id_counter += 1
                    
                    corpus_jsonl_line = json.dumps(corpus_data1)
                    corpus.write(corpus_jsonl_line + '\n')
                    qrels_jsonl_line = json.dumps(qrels_data1)
                    qrels.write(qrels_jsonl_line + '\n')
                    
                    for paragraph in paragraphs:
                        corpus_data2 = {
                            'id': str(corpus_id_counter),
                            'contents': paragraph
                        }
                        qrels_data2 = {
                            'query_id': wikidata_id,
                            'doc_id': str(corpus_id_counter),
                            'score': 0
                        }
                        corpus_id_counter += 1
                        
                        corpus_jsonl_line = json.dumps(corpus_data2)
                        corpus.write(corpus_jsonl_line + '\n')
                        qrels_jsonl_line = json.dumps(qrels_data2)
                        qrels.write(qrels_jsonl_line + '\n')
            

if __name__ == "__main__":
    
    ### === Input files & directories
    entity_freq_file = "data/dataset/EntityQuestions/entity_frequency.json"
    data_evidence_path = "data/dataset/EntityQuestions/data_evidence_costomized" 
    
    ### === output files
    queries_file = "data/generated/EntityQuestions_costomized/{}/queries.jsonl"
    corpus_file = "data/generated/EntityQuestions_costomized/{}/corpus.jsonl"
    qrels_file = "data/generated/EntityQuestions_costomized/{}/qrels.jsonl"
    
    ### === Create Query file by API ===========
    # create_queries_file(data_evidence_path, queries_file)
    
    ### === Create Corpus file by API ===========
    create_corpus_qrels_files(queries_file, corpus_file, qrels_file)
    