from typing import List
from corpus.corpus import Corpus
from corpus.quiz import Quiz
from json import dumps

from pipelines.base_pipeline import PipelineValidationError, Pipeline
from pipelines.implemented.text_chunker import TextChunker
from pipelines.implemented.semantic_text_chunker import SemanticTextChunker
from pipelines.implemented.concept_extractor import ConceptExtractor
from pipelines.implemented.concept_combiner import ConceptCombiner
from pipelines.implemented.question_answer_generator import QuestionAnswerGenerator
from pipelines.implemented.distractor_generator import DistractorGenerator
from pipelines.implemented.one_step import OneStepPipeline
from pipelines.implemented.concept_cluster_combiner import ConceptClusterCombiner
from logutils import get_logger
from index import WikiTestDataIndex


MVP_PIPELINE = [OneStepPipeline()]
PIPELINE1 = [TextChunker(), ConceptExtractor(),
             ConceptCombiner(), QuestionAnswerGenerator(),
             DistractorGenerator()]
PIPELINE2 = [SemanticTextChunker(), ConceptExtractor(),
             ConceptCombiner(), QuestionAnswerGenerator(),
             DistractorGenerator()]
PIPELINE3 = [SemanticTextChunker(), ConceptExtractor(),
             ConceptClusterCombiner(), QuestionAnswerGenerator(),
             DistractorGenerator()]


class QuizGenerator:
    def __init__(self, corpus: Corpus, mcq_number: int = 30,
                 pipelines: List[Pipeline] = PIPELINE3,
                 index: WikiTestDataIndex = None,
                 store_intermediate: bool = True):
        self.context = dict()
        self.context["article"] = corpus.clean_text
        self.context["mcq_number"] = mcq_number
        self.piplines: List[Pipeline] = pipelines
        self.logger = get_logger(QuizGenerator.__name__)
        self.store_intermediate = store_intermediate
        self.index = index
        for p in self.piplines:
            p.set_context(self.context)

    def generate(self) -> Quiz:
        tmp = self.context["article"]
        for pipeline in self.piplines:
            try:
                tmp = pipeline.process(tmp)
            except PipelineValidationError as pve:
                self.logger.error(f"failed quiz generation")
                raise pve
            if self.store_intermediate:
                if isinstance(tmp, Quiz):
                    output = dumps(tmp.model_dump())
                else:
                    try:
                        output = dumps(tmp)
                    except:
                        self.logger.warning(f"Can't store intermediate pipeline output for {pipeline.title}.")
                self.index.store_pipeline_output(
                    pipeline.title, output, "intermediate.json")
        self.logger.info(
            "Successful quiz generation. See the latest json file at test_data/quizzes/")
        return tmp
