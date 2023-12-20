
import wikipediaapi
import nltk
from nltk.tokenize import word_tokenize
from lmqg import TransformersQG

from utils import load_json_file, write_to_json_file

def get_wikipedia_page_paragraphs(page_title, language='en'):
    wiki_wiki = wikipediaapi.Wikipedia("MyProjectName",language)

    page = wiki_wiki.page(page_title)
    if not page.exists():
        return f"Wikipedia page with title '{page_title}' not found."

    paragraphs = [section.text for section in page.sections]
    
    # for section in page.sections:
    #     print(section.keys()) 
    
    return paragraphs

    # # Print the result
    # print(f"Paragraphs from the Wikipedia page '{wikipedia_page_title}':")
    # for i, paragraph in enumerate(paragraphs, start=1):
    #     print(f"Paragraph {i}:\n{paragraph}\n")

def remove_empty_sentences(sentence_list):
    # Use list comprehension to filter out empty sentences
    non_empty_sentences = [sentence for sentence in sentence_list if sentence.strip() != ""]
    return non_empty_sentences

def unicode_to_text(unicode_string):
    try:
        # Decode the Unicode string
        text = unicode_string.encode('utf-8').decode('unicode_escape')
        return text
    except UnicodeDecodeError:
        return "Invalid Unicode input. Please provide a valid Unicode string."


def get_by_tokens(passage, num):
    concatenated_sentence = ' '.join(passage)
    tokens = word_tokenize(concatenated_sentence)
    print(len(tokens))
    print(len(tokens[:num]))
    text_by_tokens = [unicode_to_text(tk) for tk in tokens[:num]]

    concatenated_passage = ' '.join(text_by_tokens)

    print(passage)
    print(concatenated_passage)
    return concatenated_passage

# Example usage:
if __name__ == "__main__":
    
    # Get paragraphs from the Wikipedia page
    entities_lists = load_json_file("./data/filtered_entities_list.json")
    wiki_context = {}
    for key, value in entities_lists.items():
        paragraphs = get_wikipedia_page_paragraphs(key)
        # pre-processing
        non_empty_paragraphs = remove_empty_sentences(paragraphs)
        if len(non_empty_paragraphs) > 1:
            wiki_context[key] = get_by_tokens(non_empty_paragraphs, 300)
    write_to_json_file("./data/entities_wiki_context.json", wiki_context)

    # QAs generation form context
    wiki_context = load_json_file("./data/entities_wiki_context.json")
    entities_qa = {}
    model = TransformersQG(language="en")
    for key, value in wiki_context.items():
        print('Generating QA for {} ...'.format(key))
        qa = model.generate_qa(unicode_to_text(value))
        print(qa)
        entities_qa[key] = qa
    write_to_json_file("/content/drive/MyDrive/QA_generation/entities_qa.json", entities_qa)


