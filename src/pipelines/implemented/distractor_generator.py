from pipelines.base_pipeline import Pipeline 


class DistractorGenerator(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__("distractor_generator_pipeline")
    def _process(self, paragraph: str):
        pass
    def _validate(self, input_data, output_data):
        # TODO: implement later
        return True