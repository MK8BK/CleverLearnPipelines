from typing import List
from corpus.corpus import Corpus
from corpus.quiz import Quiz

from pipelines.base_pipeline import PipelineValidationError, Pipeline
from pipelines.implemented.text_chunker import TextChunker
from pipelines.implemented.concept_extractor import ConceptExtractor
from pipelines.implemented.concept_combiner import ConceptCombiner
from pipelines.implemented.question_answer_generator import QuestionAnswerGenerator
from pipelines.implemented.distractor_generator import DistractorGenerator


QUIZ_PIPELINE_1 = [TextChunker(), ConceptExtractor(),
                   ConceptCombiner(), QuestionAnswerGenerator(),
                   DistractorGenerator()]


class QuizGenerator:
    def __init__(self, corpus: Corpus, mcq_number: int, pipelines: List[Pipeline] = QUIZ_PIPELINE_1):
        self.context = dict()
        self.context["article"] = corpus.clean_text
        self.context["mcq_number"] = mcq_number
        self.piplines: List[Pipeline] = pipelines

    def generate(self) -> Quiz:
        tmp = self.context["article"]
        for pipeline in self.piplines:
            print(f"Executing {pipeline.title} pipeline.")
            try:
                tmp = pipeline.process(tmp)
            except PipelineValidationError as pve:
                raise pve
        return tmp
