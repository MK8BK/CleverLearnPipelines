from pipelines.base_pipeline import Pipeline 


"""
Pipeline 3: combiner tous les C/I en eliminant les redondances
            --> mesure de distance entre les concepts
"""

class ConceptCombiner(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__("concept_combiner_pipeline")
    def _process(self, input_data: str):
        # TODO: implement later
        return input_data.split("\n\n")
    def _validate(self, input_data, output_data):
        # TODO: implement later
        return True