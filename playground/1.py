import wikipediaapi

def get_wikipedia_summary_and_paragraphs(title):
    inappropriate_sections = [
        'References', 'Sources', 'External links', 'See also', 'Gallery', 
        'Television', 'Filmography', 'Discography', 'Bibliography'
    ]
    wiki_wiki = wikipediaapi.Wikipedia('my_project', 'en')
    page = wiki_wiki.page(title)

    if not page.exists():
        return "Page does not exist", []


    summary = page.summary
    print(summary)
    # paragraphs = [section.text for section in page.sections if section.title not in inappropriate_sections]
    for section in page.sections:
        if section.title not in inappropriate_sections:
            print(section.title)
            print(section.text)
    # paragraphs = [item for item in paragraphs if item != ""]

    # return summary, paragraphs

if __name__ == "__main__":
    title = "Thomas McMurtry"
    get_wikipedia_summary_and_paragraphs(title)