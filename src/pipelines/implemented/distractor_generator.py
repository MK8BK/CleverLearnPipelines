from typing import Tuple, List
from pydantic import BaseModel

# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------

from corpus.quiz import Quiz, MultipleChoiceQuestion
from llms.openai import OpenAI_client, OpenAI_role, Message
from pipelines.base_pipeline import Pipeline


class Distractors(BaseModel):
    distractors: List[str]


class DistractorGenerator(Pipeline):
    title = "distractor_generator_pipeline"

    def __init__(self, *args, **kwargs):
        super().__init__(DistractorGenerator.title)
        self.client = OpenAI_client()

    def _process(self, question_answers: List[Tuple[str, str]]) -> Quiz:
        messages = [
            (Message(OpenAI_role.DEVELOPER,
                     self.prompt_store["directives"].format(self.prompt_store["min_distractor_number"])),
             Message(OpenAI_role.USER,
                     f"Question: {m[0]}\nCorrect answer: {m[1]}")) for m in question_answers
        ]
        distractors: List[Distractors] = self.client.concurrent_submit_messages(
            messages, response_format=Distractors)
        mcqs: List[MultipleChoiceQuestion] = [
            MultipleChoiceQuestion(
                question=question_answers[i][0], answer=question_answers[i][1], distractors=distractors[i].distractors)
            for i in range(len(question_answers))]
        quiz: Quiz = Quiz(mcqs=mcqs)
        return quiz

    def _validate(self, input_data, output_data):
        # TODO: implement later
        return True


def main():
    from index import WikiTestDataIndex
    from test_helpers import TEST_DATA_PATH, measure_time
    from question_answer_generator import QuestionAnswerGenerator

    index = WikiTestDataIndex(TEST_DATA_PATH)

    dg = DistractorGenerator()
    index.ensure_pipeline_dir(dg.title)

    from json import dumps, loads

    input_file_path = TEST_DATA_PATH.joinpath("pipelines").joinpath(
        QuestionAnswerGenerator.title).joinpath("out1.json")

    with open(str(input_file_path), "r", encoding="utf8") as f:
        content = loads(f.read())

    @measure_time
    def p(input_: List[List[str]]):
        return dg.process(input_)
    content = [(q["question"], q["answer"]) for q in content]
    quiz: Quiz = p(content)
    pipeline_output = dumps(quiz.model_dump())
    index.store_pipeline_output(dg.title, pipeline_output, "out1.json")


if __name__ == "__main__":
    main()
