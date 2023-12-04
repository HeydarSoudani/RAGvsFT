import wikipediaapi
import random
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm

from utils import read_tsv_column, write_to_json_file

random.seed(10)

#TODO: Add Wikiextractor


def check_token_length(text, limitation=300):
    limited_text = []
    for item in text:
        tokens_ctx = word_tokenize(item)
        if len(tokens_ctx) < limitation:
            limited_text.append(item)
        else:
            limited_text.append(' '.join(tokens_ctx[:limitation]))
    return limited_text

def get_wikipedia_page_paragraphs(page_title, context_type="summary", language='en'):
    # context_type: ["summary", "paragraphs"]
    wiki_wiki = wikipediaapi.Wikipedia("MyProjectName", language)
    page = wiki_wiki.page(page_title)
    if not page.exists():
        return f"Wikipedia page with title '{page_title}' not found."
    
    if context_type == "summary":
        summary = [page.summary]
        return summary
    else:
        paragraphs = [section.text for section in page.sections]
        return paragraphs


def get_wiki_context_by_api(title_list, context_type="summary"):
    res = {}
    progress_bar = tqdm(range(len(title_list)))
    for title in title_list:
        # print('{} ...'.format(title))
        out = get_wikipedia_page_paragraphs(title, context_type)
        if type(out) == list:
            res[title] = out
            # res[title] = check_token_length(out)
        progress_bar.update(1)
                
    return res

def get_wiki_context_by_api_by_title(title, context_type="summary"):
    out = get_wikipedia_page_paragraphs(title, context_type)
    if type(out) == list:
        return out


def get_wiki_context_from_dump():
    pass


if __name__ == "__main__":
    dataset_path = "data/dataset/popQA.tsv"
    # entity_list = read_tsv_column(dataset_path, "subj")
    wiki_title_list = read_tsv_column(dataset_path, "s_wiki_title")
    context_sum = get_wiki_context_by_api(wiki_title_list, context_type="summary")

    # wiki_title_list_rnd = random.sample(wiki_title_list, 200)
    # context_sum = get_wiki_context_by_api(wiki_title_list_rnd, context_type="summary")
    
    ### if you want to add more context
    # for title, value in context_sum.items():
    #     context[title] = value[0]

    #     tokens_ctx = word_tokenize(context[title])
    #     print(len(tokens_ctx))
    #     if len(tokens_ctx) < 150:
    #         context_pra = get_wiki_context_by_api_by_title(title, context_type="paragraphs")
    #         if len(word_tokenize(context_pra[0])) < 300:
    #             context[title] += context_pra[0]      
    #     print(len(word_tokenize(context[title])))    

    context_path = "data/generated/wikiapi_results.json"
    write_to_json_file(context_path, context_sum)