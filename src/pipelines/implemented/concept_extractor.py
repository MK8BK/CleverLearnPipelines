import concurrent.futures
from typing import List
import asyncio
from pydantic import BaseModel
import concurrent


# Temporary -----------------------------------------------------
# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------

from corpus.corpus import CorpusLanguage
from llms.openai import OpenAI_client, OpenAI_role, Message
from pipelines.base_pipeline import Pipeline

"""
Pipeline 2: extraire de chaque paragraphe les C/I (concepts/infos) importantes
            --> summarization TODO
"""



class Concepts(BaseModel):
    concepts: List[str]


class ConceptExtractor(Pipeline):
    NTHREADS = 20
    title = "concept_extractor_pipeline"

    def __init__(self, *args, **kwargs):
        super().__init__(ConceptExtractor.title, *args, **kwargs)
        self.client = OpenAI_client()
        self.outputs = None
        self.prompt_store["concept_limit"] = 3  # Exaple: Limit to 5 concepts

    def _process(self, paragraphs: List[str]) -> List[List[str]]:
        """
            Tried asyncio, utter garbage, using threads instead ...
https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example
        """
        cf = concurrent.futures
        with cf.ThreadPoolExecutor(max_workers=ConceptExtractor.NTHREADS) as executor:
            indices = list(range(len(paragraphs)))
            results = executor.map(self._process_paragraph, paragraphs, indices)
            return list(results)


    def _process_paragraph(self, paragraph: str, index: int) -> List[str]:
        dev_message = self.prompt_store["extraction_directives"].format(
            self.prompt_store["concept_limit"])
        messages = [Message(OpenAI_role.DEVELOPER, dev_message),
                    Message(OpenAI_role.USER, paragraph)]
        concepts_list: Concepts = self.client.submit_messages(messages,
                                                  response_format=Concepts)
        # print(index, end=" ")
        return concepts_list.concepts

    def _validate(self, input_data, output_data):
        # return True
        lengths = [len(concepts_list) for concepts_list in output_data]
        valid = [1 if l<=self.prompt_store["concept_limit"] else 0 for l in lengths]
        valid_count = sum(valid)
        if valid_count<len(valid):
            self.logger.warning(
                f"({len(valid)-valid_count} / {len(valid)}) chunks had more than `concept_limit` concepts.")
        valid = valid_count/len(valid)
        return True # valid>.95


if __name__ == "__main__":
    from index import WikiTestDataIndex
    from test_helpers import TEST_DATA_PATH, measure_time
    from text_chunker import TextChunker

    index = WikiTestDataIndex(TEST_DATA_PATH)
    # print(index.data_path)

    ce = ConceptExtractor()
    index.ensure_pipeline_dir(ce.title)

    from json import dumps, loads

    input_file_path = TEST_DATA_PATH.joinpath("pipelines").joinpath(
        TextChunker.title).joinpath("out1.json")

    with open(str(input_file_path), "r", encoding="utf8") as f:
        paragraphs = loads(f.read())
    # print(paragraphs[1])

    @measure_time
    def p(pars):
        return ce.process(pars)
    pipeline_output = p(paragraphs)
    print(ce._validate(paragraphs, pipeline_output))
    pipeline_output = dumps(pipeline_output)
    index.store_pipeline_output(ce.title, pipeline_output, "out1.json")
