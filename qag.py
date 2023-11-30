from lmqg import TransformersQG
import spacy
from utils import load_json_file, write_to_json_file

spacy.load('en_core_web_sm')

#TODO: Comstomize QA generation process

def generate_qa_by_lmqg(context):
    res = {}
    # model = TransformersQG(language="en")
    model = TransformersQG(model='lmqg/t5-base-squad-qg', model_ae='lmqg/t5-base-squad-ae')
    for title, value in context.items():
        qa_list = model.generate_qa(value[0])
        res[title] = qa_list
    return res
    


if __name__ == "__main__":
    wiki_context = load_json_file("data/generated/wikiapi_results.json")
    # print(wiki_context)

    ### 1) generate QA pairs    
    qa = generate_qa_by_lmqg(wiki_context)

    qa_path = "data/generated/qag_results.json"
    write_to_json_file(qa_path, qa)
    ### 2) Filter

    ### 3) Change dataset format
    # {}