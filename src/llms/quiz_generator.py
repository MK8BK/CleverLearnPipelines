from typing import List
from corpus.corpus import Corpus
from corpus.quiz import Quiz

from pipelines.base_pipeline import PipelineValidationError, Pipeline
from pipelines.implemented.text_chunker import TextChunker
from pipelines.implemented.semantic_text_chunker import SemanticTextChunker
from pipelines.implemented.concept_extractor import ConceptExtractor
from pipelines.implemented.concept_combiner import ConceptCombiner
from pipelines.implemented.question_answer_generator import QuestionAnswerGenerator
from pipelines.implemented.distractor_generator import DistractorGenerator
from pipelines.implemented.one_step import OneStepPipeline
from logutils import get_logger


MVP_PIPELINE = [OneStepPipeline()]
PIPELINE1 = [TextChunker(), ConceptExtractor(),
                   ConceptCombiner(), QuestionAnswerGenerator(),
                   DistractorGenerator()]
PIPELINE2 = [SemanticTextChunker(), ConceptExtractor(),
                   ConceptCombiner(), QuestionAnswerGenerator(),
                   DistractorGenerator()]
                

class QuizGenerator:
    def __init__(self, corpus: Corpus, mcq_number: int=20, pipelines: List[Pipeline] = PIPELINE2):
        self.context = dict()
        self.context["article"] = corpus.clean_text
        # self.context["mcq_number"] = mcq_number
        self.piplines: List[Pipeline] = pipelines
        self.logger = get_logger(QuizGenerator.__name__)

    def generate(self) -> Quiz:
        tmp = self.context["article"]
        for pipeline in self.piplines:
            try:
                tmp = pipeline.process(tmp)
            except PipelineValidationError as pve:
                self.logger.error(f"failed quiz generation")
                raise pve
        self.logger.info("Successful quiz generation. See the latest json file at test_data/quizzes/")
        return tmp
