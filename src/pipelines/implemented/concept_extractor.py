from typing import List
from pydantic import BaseModel


# Temporary -----------------------------------------------------
# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------

from pipelines.base_pipeline import Pipeline
from llms.openai import OpenAI_client, OpenAI_role
from corpus.corpus import CorpusLanguage

"""
Pipeline 2: extraire de chaque paragraphe les C/I (concepts/infos) importantes
            --> summarization TODO
"""


class Concepts(BaseModel):
    concepts: List[str]


class ConceptExtractor(Pipeline):
    title = "concept_extractor_pipeline"
    def __init__(self, *args, **kwargs):
        super().__init__(ConceptExtractor.title)
        self.client = OpenAI_client()

    def _process(self, paragraphs: List[str]) -> List[List[str]]:
        all_concepts = []
        self.client.add_message(OpenAI_role.DEVELOPER,
                                self.prompt_store["extraction_directives"])
        for i, paragraph in enumerate(paragraphs):
            concepts_list = self._process_paragraph(paragraph)
            all_concepts.append(concepts_list)
            print(i)
            if i>5:
                return all_concepts
        return all_concepts

    def _process_paragraph(self, paragraph: str)->List[str]:
            self.client.add_message(
                OpenAI_role.USER,
                f"{self.prompt_store["list_concepts"]} ```{paragraph}```")
            concepts = self.client.submit_messages(response_format=Concepts)
            pydantic_concepts = concepts.choices[0].message.parsed
            concepts_list = pydantic_concepts.concepts
            return concepts_list

    def _validate(self, input_data, output_data):
        # TODO: implement later
        return True
    


if __name__ == "__main__":
    from corpus.corpus import Corpus
    from index import WikiTestDataIndex
    from pathlib import Path
    from test_helpers import TEST_DATA_PATH, measure_time
    from text_chunker import TextChunker


    index = WikiTestDataIndex(TEST_DATA_PATH)
    # print(index.data_path)

    ce = ConceptExtractor()
    index.ensure_pipeline_dir(ce.title)


    from json import dumps, loads

    input_file_path = TEST_DATA_PATH.joinpath("pipelines").joinpath(TextChunker.title).joinpath("out1.json")

    with open(str(input_file_path), "r", encoding="utf8") as f:
        paragraphs = loads(f.read())
    # print(paragraphs[1])

    @measure_time
    def p(pars):
        return ce.process(pars)
    pipeline_output = p(paragraphs)
    pipeline_output = dumps(pipeline_output)
    index.store_pipeline_output(ce.title, pipeline_output, "out1.json")
