# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
# https://github.com/matthewwithanm/python-markdownify

import requests
import os
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from logutils import get_logger


class WikiScraper:
    def __init__(self):
        self._cache = []
        self.logger = get_logger(WikiScraper.__name__)

    def scrape(self, url):
        self.url = url
        self.url_last_path = self.url.split('wikipedia.org/wiki/')[-1]
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
        self.logger.info(f"Successfully scraped page at {url}. See the latest markdown file at test_data/wiki/")
        return self.md

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
