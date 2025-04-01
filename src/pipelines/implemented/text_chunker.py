from typing import List
from itertools import chain

# Temporary -----------------------------------------------------
# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------

from pipelines.base_pipeline import Pipeline

"""
Pipeline 1: Split the corpus into paragraphs
    TODO: find a way to implement semantic paragraph splitting
"""

# https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html


class TextChunker(Pipeline):
    """
        Splits input text into chunks of size at least `minimum_chunk_length`.
        Splits on new line characters.
    """
    title = "text_chunker_pipeline"

    def __init__(self, *args, **kwargs):
        super().__init__(TextChunker.title)
        self.minimum_chunk_length = self.prompt_store["minimum_chunk_length"]

    def _process(self, input_data: str) -> List[str]:
        n = len(input_data)
        paragraphs = []
        index = 0
        while index < n:
            new_index = input_data.find("\n", index+self.minimum_chunk_length)
            if new_index == -1:
                paragraphs.append(input_data[index:])
                break
            paragraphs.append(input_data[index:new_index+1])  # exclusive
            index = new_index + 1
        assert len(input_data) == sum(len(p) for p in paragraphs)
        return paragraphs

    def _validate(self, input_data, output_data):
        return True


if __name__ == "__main__":

    from corpus.corpus import Corpus
    from test_helpers import TEST_DATA_PATH, measure_time
    from index import WikiTestDataIndex
    import llms
    from pathlib import Path

    index = WikiTestDataIndex(TEST_DATA_PATH)

    tc = TextChunker()
    index.ensure_pipeline_dir(tc.title)

    with open(str(TEST_DATA_PATH.joinpath("wiki").joinpath("1.md")),
              "r", encoding="utf8") as f:
        corpus = Corpus(f.read())

    from json import dumps

    @measure_time
    def p(t):
        return tc.process(t)
    pipeline_output = p(corpus.clean_text)
    # print(len(pipeline_output))
    pipeline_output = dumps(pipeline_output)
    index.store_pipeline_output(tc.title, pipeline_output, "out1.json")
