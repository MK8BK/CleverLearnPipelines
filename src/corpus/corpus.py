

class Corpus:
    def __init__(self, path: str=None, text: str=None):
        self.text = None
        if path:
            with open(path, "r") as f:
                self.text = f.read()
        else:
            self.text = text


if __name__=="__main__":
    c = Corpus("../scraping/data/wiki/Microeconomics.md")
    print(c.text)
