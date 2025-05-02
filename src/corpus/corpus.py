import re
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
        self.clean_text = self._remove_markdown_links(self.raw_text)
        self.clean_text = self._remove_urls(self.clean_text)

    def _remove_markdown_links(self,text):
        pattern = r"(?:\[(?P<text>.*?)\])\((?P<link>.*?)\)"
        # return self._remove_pattern(pattern, text)
        return re.sub(pattern, r"\g<text>", text)

    def _remove_urls(self, text):
        pattern = r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        return re.sub(pattern, "", text)

