import re
# import spacy
from enum import StrEnum

class CorpusLanguage(StrEnum):
    """
        ISO 639 language codes, see more at
            https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
    """
    FR = "fr"
    EN = "fr"


class Corpus:
    def __init__(self, text: str, lang: CorpusLanguage=CorpusLanguage.EN):
        self.raw_text = text
        self.lang = lang
        self.clean_text = self.remove_markdown_links(self.raw_text)
        self.clean_text = self.remove_urls(self.clean_text)
        # self.nlp = spacy.load('en_core_web_trf')
        # self.paragraphs = [p for p in self.paragraphs if self.contains_verb(p)]
        
    def remove_pattern(self, pattern, text):
        return re.sub(pattern, "", text)

    def remove_markdown_links(self,text):
        pattern = r"(?:\[(?P<text>.*?)\])\((?P<link>.*?)\)"
        return self.remove_pattern(pattern, text)

    def remove_urls(self, text):
        pattern = r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        return self.remove_pattern(pattern, text)

    def contains_verb(self, paragraph):
        # naive assumption, a paragraph is valid if it contains a verb
        doc = self.nlp(paragraph)
        return any(token.pos_=="VERB" for token in doc)
