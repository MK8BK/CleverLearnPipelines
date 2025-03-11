# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
# https://github.com/matthewwithanm/python-markdownify

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


class WikiScraper:
    def __init__(self):
        self._cache = []

    def scrape(self, url):
        self.url = url
        self.url_last_path = self.url.split('wikipedia.org/wiki/')[-1]
        # TODO: improve and clean later
        assert all((c in "()-_=+-[]" or c.isalnum())
                   and c != "/" for c in self.url_last_path)
        try:
            r = requests.get(url)
            if r.status_code != 200:
                return None
        except requests.ConnectionError:
            return None
        self.html_text = r.text
        self._clean()
        self.html_text = str(self.soup)
        self.md = md(self.html_text)
        return self # builder design pattern

    def _clean(self):
        def drop_all(**kwargs):
            all_tags = self.soup.find_all(**kwargs)
            if not all_tags:
                return
            for tag in all_tags:
                tag.decompose()
        self.soup = BeautifulSoup(self.html_text, "html.parser")
        self.soup = self.soup.find(name="body")
        self.soup.find(name="header").decompose()
        # self.soup.find(name="footer").decompose()
        self.soup.find(class_="vector-menu-content").decompose() # language menu
        drop_all(name="nav")
        drop_all(name="script")
        drop_all(name="img")
        drop_all(name="style")
        drop_all(name="input")
        drop_all(name="label")
        drop_all(name="audio")
        drop_all(name="figure")
        drop_all(name="button")

    def save(self, root_path: str = "data/wiki"):
        with open(f"{root_path}/{self.url_last_path}.md", "w", encoding="utf-8") as f:
            f.write(self.md)
        return self


if __name__ == "__main__":
    paraguay_article = "https://en.wikipedia.org/wiki/Paraguay"
    microecon_article = "https://en.wikipedia.org/wiki/Microeconomics"
    napoleon_article = "https://en.wikipedia.org/wiki/Napoleon"
    w = WikiScraper()
    w.scrape(paraguay_article).save()
    w.scrape(microecon_article).save()
    w.scrape(napoleon_article).save()
