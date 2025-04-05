import argparse
import sys
import pathlib
import json
from index import WikiTestDataIndex
from scraping.scraper import WikiScraper
from corpus.corpus import Corpus
# import llms.__init__
from llms.quiz_generator import QuizGenerator, PipelineValidationError

MAIN_PATH = pathlib.Path(__file__)
SRC_DIR_PATH = MAIN_PATH.parent
ROOT_PROJECT_PATH = SRC_DIR_PATH.parent
DATA_PATH = ROOT_PROJECT_PATH.joinpath("test_data")
# LOGS_DATA_PATH = DATA_PATH.joinpath("logs")

index = WikiTestDataIndex(DATA_PATH)

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--scrape-wikipedia", help="scrape a wikipedia page")
parser.add_argument("-g", "--generate-quiz",
                    help="generate a quiz for a wikipedia page, use cached version if already scraped")
parser.add_argument(
    "-l", "--logs", help="write pipeline logs to specified file")


def scrape(url):
    md = WikiScraper().scrape(url)
    index.add_document(url, md)

def main():
    args = parser.parse_args()
    if args.scrape_wikipedia:
        url = args.scrape_wikipedia
        scrape(url)
    if args.generate_quiz:
        url = args.generate_quiz
        if not index.already_scraped(url):
            scrape(url)
        corpus = Corpus(index.retrieve_doc(url))
        quiz_generator = QuizGenerator(corpus)
        try:
            quiz = quiz_generator.generate()
            str_quiz = json.dumps(quiz.model_dump())
            index.add_quiz(url, str_quiz)
            print("done, see last json file generated at test_data/quizzes")
            return
        except PipelineValidationError as pve:
            print("Could not generate quiz, validation error.")
        except Exception as e:
            print(f"Could not generate quiz. {e}")
        finally:
            sys.exit(1)
        # print(corpus.clean_text)
        

        
# paraguay_article = "https://en.wikipedia.org/wiki/Paraguay"
# microecon_article = "https://en.wikipedia.org/wiki/Microeconomics"
# napoleon_article = "https://en.wikipedia.org/wiki/Napoleon"

if __name__=="__main__":
    main()
else:
    raise RuntimeError("main is imported")

