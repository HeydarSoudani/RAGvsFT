import numpy as np
import requests
import json
import pandas as pd

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
    
def update_jsonl_with_relations(jsonl_file, entity_relation_dict, output_file):
    with open(jsonl_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            json_obj = json.loads(line)
            entity_id = json_obj['entity_id']
            # json_obj['relation_type'] = entity_relation_dict.get(entity_id, None)
            json_obj['el_pop'] = entity_relation_dict.get(entity_id, None)
            json.dump(json_obj, outfile)
            outfile.write('\n')
            

if __name__ == "__main__":

    path = 'analysis/some_data/pretraining_entities/wikipedia_entity_map.npz'
    queries_file = "data/generated/popQA_costomized/queries.jsonl"
    output_path = "data/generated/popQA_costomized/el_popularity.json"
      
    # with np.load(path) as data:
    #     entities_list = []
    #     for idx, (url, value) in enumerate(data.items()):
    #         # if idx == 10:
    #         #     break
    #         if (idx+1)%1000 == 0:
    #             print("{} data are processed".format(idx+1))
            
    #         wikidata_title = url.split('/')[-1]
    #         # wikidata_id = get_wikidata_id(wikidata_title)
    #         # print('{}: {}'.format(wikidata_id, wikidata_title))
    #         # if wikidata_id:
    #         entities_list.append({
    #             # 'wd_id': wikidata_id,
    #             'wp_title': wikidata_title,
    #             'doc_num': len(value)
    #         })
        
    #     with open(output_path, 'w', encoding='utf-8') as file:
    #         json.dump(entities_list, file, indent=4)
    #     # with open(output_path, 'w') as f:
    #     #     json.dump(entities_list, f)
    
    
    
    with open(output_path, 'r') as file:
        el_popularity = json.load(file)
        
    queries_el_pop = {}
    with open(queries_file, 'r') as queries:
        for idx, line in enumerate(queries):
            
            if (idx+1)%300 == 0:
                print("# processed queries:", idx+1)
            
            data = json.loads(line.strip())
            entity_id = data['entity_id']
            wiki_title = data['wiki_title']
    
            for obj in el_popularity:
                queries_el_pop[entity_id] = None
                if obj.get('wp_title').replace("_", " ").lower() == wiki_title.lower():
                    queries_el_pop[entity_id] = obj.get('doc_num')
                    break
                
    # print(queries_el_pop)
    jsonl_file = 'data/generated/popQA_costomized/queries.jsonl'
    output_file = 'data/generated/popQA_costomized/new_queries.jsonl'
    update_jsonl_with_relations(jsonl_file, queries_el_pop, output_file)
    
            
    
        

