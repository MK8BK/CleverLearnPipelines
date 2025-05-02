from typing import List, Tuple

# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------

from pipelines.base_pipeline import Pipeline 
from llms.openai import OpenAI_client, OpenAI_role, Message
from pipelines.implemented.concept_combiner import ConceptCombiner

from pydantic import BaseModel
class QuestionAnswer(BaseModel):
    question: str
    answer: str

class QuestionAnswerGenerator(Pipeline):
    title = "question_answer_generator_pipeline"
    def __init__(self, *args, **kwargs):
        super().__init__(QuestionAnswerGenerator.title)
        self.client = OpenAI_client()
    def _process(self, concepts_list: List[str])->List[Tuple[str, str]]:
        dev_messages = [Message(OpenAI_role.DEVELOPER, self.prompt_store["generation_directives"])]*len(concepts_list)
        user_messages = [Message(OpenAI_role.USER, concept) for concept in concepts_list]
        messages = list(zip(dev_messages, user_messages))
        qas = self.client.concurrent_submit_messages(messages, response_format=QuestionAnswer)
        qas = [(qa.question, qa.answer) for qa in qas]
        return qas


    def _validate(self, input_data, output_data):
        # TODO: implement later
        return True


def main():
    from index import WikiTestDataIndex
    from test_helpers import TEST_DATA_PATH, measure_time
    from concept_combiner import ConceptCombiner

    index = WikiTestDataIndex(TEST_DATA_PATH)

    qag = QuestionAnswerGenerator()
    index.ensure_pipeline_dir(qag.title)

    from json import dumps, loads

    input_file_path = TEST_DATA_PATH.joinpath("pipelines").joinpath(
        ConceptCombiner.title).joinpath("out1.json")

    with open(str(input_file_path), "r", encoding="utf8") as f:
        paragraphs = loads(f.read())

    @measure_time
    def p(input_: List[List[str]]):
        return qag.process(input_)
    pipeline_output = p(paragraphs)
    print(qag._validate(paragraphs, pipeline_output))
    pipeline_output = [qa.model_dump() for qa in pipeline_output]
    pipeline_output = dumps(pipeline_output)
    index.store_pipeline_output(qag.title, pipeline_output, "out1.json")



if __name__=="__main__":
    main()