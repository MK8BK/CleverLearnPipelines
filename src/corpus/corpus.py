import re
import spacy

class Corpus:
    def __init__(self, path: str=None, text: str=None):
        self.raw_text = None
        if not path:
            assert text is not None, "no path to file and no str text provided"
            self.raw_text = text
            return
        # eventually add try except block here
        with open(path, "r", encoding="utf-8") as f:
            self.raw_text = f.read()
        self.nlp = spacy.load('en_core_web_trf')
        self.clean_text = self.remove_markdown_links(self.raw_text)
        self.clean_text = self.remove_urls(self.clean_text)
        self.paragraphs = self.split_into_paragraphs(self.clean_text)
        # self.paragraphs = [p for p in self.paragraphs if self.contains_verb(p)]
        self.clean_text = "\n\n".join(self.paragraphs)

    def split_into_paragraphs(self, text):
        # naive approach for now
        paragraphs = text.split("\n\n")
        return paragraphs
        
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

if __name__=="__main__":
    # c = Corpus("../scraping/data/wiki/Microeconomics.md")
    c = Corpus("../scraping/data/wiki/Paraguay.md")
    print(c.clean_text)
