
import urllib.request as urllib2
from urllib.parse import quote
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math

from utils import load_json_file, write_to_json_file

def convert_to_url_format(text):
    # Use urllib.parse.quote to convert text to URL format
    url_formatted_text = quote(text)
    return url_formatted_text

def get_pageviews(entities_list):
    TOP_API_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.{project}/all-access/all-agents/{topic}/monthly/{date_from}/{date_to}"
    lang = 'en'
    project = 'wikipedia'
    date_from = "2018010100"
    date_to = "2022123100"

    topic_count = []
    for topic in entities_list:
        edited_topic = convert_to_url_format(topic.replace(" ", "_"))
        url = TOP_API_URL.format(lang=lang,
                                project = project,
                                topic = edited_topic,
                                date_from = date_from,
                                date_to = date_to)
        try:
            resp = urllib2.urlopen(url)
            resp_bytes = resp.read()
            data = json.loads(resp_bytes)
            all_views = sum([item['views'] for item in data['items']]) 
            # print("Target: {:<15}, Views: {}".format(topic, all_views))
            topic_count.append(all_views)
        except urllib2.HTTPError as e:
            # print(e.code)
            print("Target: {:<20}, does not have a wikipedia page".format(edited_topic))
        except urllib2.URLError as e:
            print(e.args)
    
    return topic_count

def get_logs(input_list, base=10):
    # Calculate logarithm for each element in the list
    logs = [math.log(x, base) for x in input_list]
    return logs

def get_less_popular_entities(dataset, coef):
    pageviews = load_json_file('./data/pageview_list.json')
    entities_lists = load_json_file("./data/entities_list.json")

    pv = get_logs(pageviews[dataset])
    threshold = int((max(pv) - min(pv))*coef) + min(pv)
    result_dict = dict(zip(entities_lists["popqa"], pv))
    filtered = {obj[0]: obj[1] for obj in result_dict.items() if obj[1] < threshold}

    return filtered


if __name__ == "__main__":
    
    ### Get pageviews
    # entities_lists = load_json_file("./data/entities_list.json")
    # pageviews = {}
    # for key, value in entities_lists.items():
    #     print('{} ...'.format(key))
    #     pageviews[key] = get_pageviews(value)
    
    ### == Write the lists of words to the JSON file
    # write_to_json_file('./data/pageview_list.json', pageviews)    


    ### Plot popularity
    # pageviews = load_json_file('./data/pageview_list.json')
    # for key, value in pageviews.items():
    #     sns.kdeplot(get_logs(value), label=key, fill=True, common_norm=False)
    # sns.kdeplot(get_logs(pageviews['popqa']), label='popqa', fill=True, common_norm=False)
    # plt.axvline(x=threshold)
    # plt.xlabel("Popularity")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.show()


    filtered = get_less_popular_entities("popqa", 0.3) # 0.4 -> 119
    print(filtered)
    print(len(filtered))
    write_to_json_file('./data/filtered_entities_list.json', filtered)

    ### === get wikipedia context




    # data = load_json_file('./data/pageview_list.json')
    # data["ECQG"] = get_pageviews(entities_lists['ECQG'])
    # write_to_json_file('./data/pageview_list.json', data)


    

