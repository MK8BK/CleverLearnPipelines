from pipelines.base_pipeline import Pipeline 

class QuestionAnswerGenerator(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__("question_answer_generator_pipeline")
    def _process(self, input_data: str):
        # TODO: implement later
        return input_data.split("\n\n")
    def _validate(self, input_data, output_data):
        # TODO: implement later
        return True