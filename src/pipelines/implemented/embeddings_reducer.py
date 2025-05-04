from typing import List
from time import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Temporary -----------------------------------------------------
# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------
from pipelines.base_pipeline import Pipeline
import logutils

class EmbeddingsReducer(Pipeline):
    """
        Use Principle Component Analysis to reduce the number of dimensions of 
        a set of embeddings
    """
    title = "embeddings_reducer_pipeline"

    def __init__(self, *args, **kwargs):
        super().__init__(EmbeddingsReducer.title)
        start = time()
        self.logger = logutils.get_logger(EmbeddingsReducer.title)
        # 80% of the variance of the embeddings should be explained by the kept
        # dimensions
        self.model = PCA(n_components=0.80)
        self.scaler = StandardScaler()
        end = time()
        self.logger.debug(f"Pipeline setup done in {end-start}s")

    def _process(self, sentence_embeddings: np.ndarray) -> np.ndarray:
        scaled_embeddings = self.scaler.fit_transform(sentence_embeddings)
        reduced_embeddings = self.model.fit_transform(scaled_embeddings)
        self.logger.debug(f"{reduced_embeddings.shape[1]}/{sentence_embeddings.shape[1]} dimensions kept")
        return reduced_embeddings

    def _validate(self, input_data, output_data):
        return True


if __name__ == "__main__":
    LOGGER = logutils.get_logger(__name__)
    from test_helpers import TEST_DATA_PATH, measure_time
    from index import WikiTestDataIndex
    from sentence_splitter import SentenceSplitter
    from sentence_to_vec import Sentence2Vec
    from corpus.corpus import Corpus

    index = WikiTestDataIndex(TEST_DATA_PATH)

    ss = SentenceSplitter()
    s2v = Sentence2Vec()
    er = EmbeddingsReducer()

    input_file_path = TEST_DATA_PATH.joinpath("wiki").joinpath("1.md")

    with open(str(input_file_path), "r", encoding="utf8") as f:
        c = Corpus(f.read())

    sentences = ss.process(c.clean_text)
    sentence_vectors = s2v.process(sentences)

    @measure_time
    def p(input_: List[List[str]]):
        return er.process(input_)
    p(sentence_vectors)
    
