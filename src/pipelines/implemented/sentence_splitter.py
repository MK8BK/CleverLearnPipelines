
from typing import List
from nltk import tokenize

# Temporary -----------------------------------------------------
# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------
from pipelines.base_pipeline import Pipeline


class SentenceSplitter(Pipeline):
    """
        Split the text into sentences
    """
    title = "sentence_splitter_pipeline"

    def __init__(self, *args, **kwargs):
        super().__init__(SentenceSplitter.title)

    def _process(self, input_data: str) -> List[str]:
        return tokenize.sent_tokenize(input_data, language="english")

    def _validate(self, input_data, output_data):
        return True


if __name__ == "__main__":
    from corpus.corpus import Corpus
    from test_helpers import TEST_DATA_PATH, measure_time
    from index import WikiTestDataIndex

    index = WikiTestDataIndex(TEST_DATA_PATH)

    ss = SentenceSplitter()
    index.ensure_pipeline_dir(ss.title)

    with open(str(TEST_DATA_PATH.joinpath("wiki").joinpath("1.md")),
              "r", encoding="utf8") as f:
        corpus = Corpus(f.read())

    from json import dumps

    @measure_time
    def p(t):
        return ss.process(t)
    pipeline_output = p(corpus.clean_text)
    pipeline_output = dumps(pipeline_output)
    index.store_pipeline_output(ss.title, pipeline_output, "out1.json")
