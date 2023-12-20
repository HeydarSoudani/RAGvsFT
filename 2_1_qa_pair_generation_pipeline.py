from lmqg import TransformersQG
import spacy, json
import nltk
from nltk.tokenize import sent_tokenize

spacy.load('en_core_web_sm')
nltk.download('punkt')

from utils import load_json_file, write_to_json_file


#TODO: Comstomize QA generation process

# def generate_qa_by_lmqg(context):
#     res = {}
#     # model = TransformersQG(language="en")
#     model = TransformersQG(model='lmqg/t5-base-squad-qg', model_ae='lmqg/t5-base-squad-ae')
#     for title, value in context.items():
#         qa_list = model.generate_qa(value[0])
#         res[title] = qa_list
#     return res
    


if __name__ == "__main__":
    wiki_corpus_path = "data/generated/popQA_costomized/corpus.jsonl"
    
    # read model
    # model = TransformersQG(
    #     model='lmqg/t5-base-squad-qg',
    #     model_ae='lmqg/t5-base-squad-ae'
    # )
    
    # Read corpus and split per sentences
    with open(wiki_corpus_path, 'r') as file:
        for idx, line in enumerate(file):
            if idx == 2:
                break
            
            # Convert each line from JSON string to Python dictionary
            passage = json.loads(line.strip())['text']
            sentences = sent_tokenize(passage)
            
            for sentence in sentences:
                print(sentence)        
    
    
    # wiki_context = load_json_file("data/generated/wikiapi_results.json")
    # # print(wiki_context)

    # ### 1) generate QA pairs    
    # qa = generate_qa_by_lmqg(wiki_context)

    # qa_path = "data/generated/qag_results.json"
    # write_to_json_file(qa_path, qa)
    ### 2) Filter

    ### 3) Change dataset format
    # {}