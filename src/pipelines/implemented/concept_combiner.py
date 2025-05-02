from typing import List


# Temporary -----------------------------------------------------
# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------
from pipelines.base_pipeline import Pipeline 

"""
Pipeline 3: combiner tous les C/I en eliminant les redondances
            --> mesure de distance entre les concepts
"""

class ConceptCombiner(Pipeline):
    title = "concept_combiner_pipeline"
    """
        sbert.net/docs/quickstart.html
    """
    def __init__(self, *args, **kwargs):
        super().__init__(ConceptCombiner.title)

    def _process(self, input_data: List[List[str]]) -> List[str]:
        return [l for lst in input_data for l in lst]

    def _validate(self, input_data, output_data):
        # TODO: implement later
        return True

def main():
    from index import WikiTestDataIndex
    from test_helpers import TEST_DATA_PATH, measure_time
    from concept_extractor import ConceptExtractor

    index = WikiTestDataIndex(TEST_DATA_PATH)
    # print(index.data_path)

    cc = ConceptCombiner()
    index.ensure_pipeline_dir(cc.title)

    from json import dumps, loads

    input_file_path = TEST_DATA_PATH.joinpath("pipelines").joinpath(
        ConceptExtractor.title).joinpath("out1.json")

    with open(str(input_file_path), "r", encoding="utf8") as f:
        paragraphs = loads(f.read())

    @measure_time
    def p(input_: List[List[str]]):
        return cc.process(input_)
    pipeline_output = p(paragraphs)
    print(cc._validate(paragraphs, pipeline_output))
    pipeline_output = dumps(pipeline_output)
    index.store_pipeline_output(cc.title, pipeline_output, "out1.json")

if __name__=="__main__":
    main()