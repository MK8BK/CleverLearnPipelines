from typing import List
from time import time
import numpy as np
from sentence_transformers import SentenceTransformer

# Temporary -----------------------------------------------------
# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------
from pipelines.base_pipeline import Pipeline
import logutils

class Sentence2Vec(Pipeline):
    """
        Transform a set of sentences into a set of vectors.
        Method retained: average the word2vec score
    """
    title = "sentence_to_vec_pipeline"

    def __init__(self, *args, **kwargs):
        super().__init__(Sentence2Vec.title)
        start = time()
        self.logger = logutils.get_logger(Sentence2Vec.title)
        # TODO: update to handle multiple languages later
        self.model_name = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        end = time()
        self.logger.debug(f"Pipeline setup done in {end-start}s")

    def _process(self, sentences: List[str]) -> np.ndarray:
        embeddings = self.model.encode(sentences)
        return embeddings

    def _validate(self, input_data, output_data):
        return True


if __name__ == "__main__":
    from test_helpers import TEST_DATA_PATH, measure_time
    from index import WikiTestDataIndex
    from sentence_splitter import SentenceSplitter
    from corpus.corpus import Corpus

    index = WikiTestDataIndex(TEST_DATA_PATH)

    ss = SentenceSplitter()
    s2v = Sentence2Vec()

    input_file_path = TEST_DATA_PATH.joinpath("wiki").joinpath("1.md")

    with open(str(input_file_path), "r", encoding="utf8") as f:
        c = Corpus(f.read())

    sentences = ss.process(c.clean_text)

    @measure_time
    def p(input_: List[str]):
        return s2v.process(input_)
    sentence_vectors = p(sentences)
    print(sentence_vectors)

    # index.store_pipeline_output(s2v.title, pipeline_output, "out1.json")
