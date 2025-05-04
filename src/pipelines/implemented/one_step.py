from corpus.corpus import CorpusLanguage
from pipelines.base_pipeline import Pipeline
from llms.openai import OpenAI_client, OpenAI_role, Message
from corpus.quiz import Quiz


class OneStepPipeline(Pipeline):
    title = "one_step_pipeline"

    def __init__(self, language=CorpusLanguage.EN):
        super().__init__(OneStepPipeline.title, language)
        self.client = OpenAI_client()

    def _process(self, corpus: str) -> Quiz:
        messages = [Message(OpenAI_role.DEVELOPER,
                        self.prompt_store["directives"].format(self.prompt_store["min_question_number"])),
                    Message(OpenAI_role.USER, corpus)]
        return self.client.submit_messages(messages, response_format=Quiz)
    def _validate(self, input_data, output_data):
        return True


