import requests, uuid, json, ast
import urllib.request as urllib2
from urllib.parse import quote
import wikipediaapi
import pandas as pd

def convert_to_url_format(text):
    # Use urllib.parse.quote to convert text to URL format
    url_formatted_text = quote(text)
    return url_formatted_text

def get_pageviews(wiki_title):
    TOP_API_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.{project}/all-access/all-agents/{topic}/monthly/{date_from}/{date_to}"
    lang = 'en'
    project = 'wikipedia'
    date_from = "2023010100"
    date_to = "2023121400"
    all_views = 0
    
    edited_title = convert_to_url_format(wiki_title.replace(" ", "_"))
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
        'Selected compositions', 'Select bibliography',
        'Historic population', 'Family tree', 'Table',
        'Selected works', 'Quotes', 'Literary awards', 'Select critical works',
        'Interview'
    ]
    wiki_wiki = wikipediaapi.Wikipedia('my_project', 'en')
    page = wiki_wiki.page(title)

    if not page.exists():
        return "Page does not exist", []

    summary = page.summary
    paragraphs = [section.text for section in page.sections if section.title not in inappropriate_sections]
    paragraphs = [item for item in paragraphs if item != ""]

    return summary, paragraphs

def create_queries_file(tsv_file, jsonl_file):
    
    df = pd.read_csv(tsv_file, sep='\t')
    possible_answers = [ast.literal_eval(item) for item in df['possible_answers']]
    question = list(df['question'])
    relation_type = list(df['prop'])
    entity_id = [item.split('/')[-1] for item in df['s_uri']]
    
    with open(jsonl_file, 'w') as jsonl:
        for idx, item in enumerate(entity_id):
            wiki_title = get_wikipedia_title_from_wikidata(item)
            
            if wiki_title != None:
                print('Id: {}, wikiID: {}, title: {}'.format(idx, item, wiki_title))
                pageviews = get_pageviews(wiki_title)
                temp = {
                    "entity_id": item,
                    'relation_type': relation_type[idx],
                    "pageviews": pageviews,
                    "question": question[idx],
                    "possible_answers": possible_answers[idx]
                }
                jsonl.write(json.dumps(temp) + '\n')

def create_corpus_qrels_files(queries_file, corpus_file, qrels_file):
    with open(queries_file, 'r') as queries, open(corpus_file, 'w') as corpus, open(qrels_file, 'w') as qrels:
        corpus_id_counter = 1
        for idx, line in enumerate(queries):
            if (idx+1)%300 == 0:
                print("# processed queries:", idx+1)
            # if idx == 400:
            #     break
            
            data = json.loads(line.strip())
            wikidata_id = data['entity_id']
            wikipedia_title = get_wikipedia_title_from_wikidata(wikidata_id)
            
            if wikipedia_title:
                summary, paragraphs = get_wikipedia_summary_and_paragraphs(wikipedia_title)
                
                # Write summary in files
                # corpus_id = str(uuid.uuid4())
                corpus_data1 = {
                    'id': str(corpus_id_counter),
                    'contents': summary
                    # 'corpus_id': corpus_id,
                    # 'title': wikipedia_title,
                    # 'entityID': wikidata_id,
                    # 'text': summary,
                    # 'has_answer': True
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
                
                
                # Write paragraphs in files
                for paragraph in paragraphs:
                    # corpus_id = str(uuid.uuid4())
                    corpus_data2 = {
                        'id': str(corpus_id_counter),
                        'contents': paragraph
                        # 'corpus_id': corpus_id,
                        # 'title': wikipedia_title,
                        # 'entityID': wikidata_id,
                        # 'text': paragraph,
                        # 'has_answer': False
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
    
    popQA_input_file = "data/dataset/popQA.tsv"
    queries_file = "data/generated/popQA_costomized/queries.jsonl"
    # create_queries_file(popQA_input_file, queries_file)
    
    corpus_file = "data/generated/popQA_costomized/corpus.jsonl"
    qrels_file = "data/generated/popQA_costomized/qrels.jsonl"
    create_corpus_qrels_files(queries_file, corpus_file, qrels_file)


