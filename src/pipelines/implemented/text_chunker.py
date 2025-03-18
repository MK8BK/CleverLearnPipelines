from typing import List
from itertools import chain

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

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
    title = "text_chunker_pipeline"
    def __init__(self, *args, **kwargs):
        super().__init__(TextChunker.title)
        self.minimum_chunk_length = self.prompt_store["minimum_chunk_length"]
        self.maximum_chunk_length = self.prompt_store["maximum_chunk_length"]

        self.semantic_text_splitter = SemanticChunker(OpenAIEmbeddings())
        self.length_based_text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=self.maximum_chunk_length,
            chunk_overlap=10,
        )

    def _process(self, input_data: str) -> List[str]:
        docs = self.semantic_text_splitter.create_documents([input_data])
        # this semantic text splitter tends to produce short docs
        # when the text is not a proper paragraph, real paragraphs tend to be
        # grouped together i.e. large ... TODO: review
        long_enough_docs = ([doc.page_content] for doc in filter(
            lambda doc: len(doc.page_content) > self.minimum_chunk_length,
            docs))
        texts = (self.length_based_text_splitter.split_text(doc[0])
                    if len(doc[0])>self.maximum_chunk_length
                 else doc
                 for doc in long_enough_docs)
        texts = list(chain.from_iterable(texts))
        return texts

    def _validate(self, input_data, output_data):
        # TODO: implement later
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

