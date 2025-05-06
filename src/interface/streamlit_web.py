import sys
sys.path.insert(0, '..')
import streamlit as st
from streamlit_ace import st_ace
from pipelines.base_pipeline import Pipeline

# eventually generalize this
from pipelines.implemented.concept_cluster_combiner import ConceptClusterCombiner
from pipelines.implemented.concept_combiner import ConceptCombiner
from pipelines.implemented.concept_extractor import ConceptExtractor
from pipelines.implemented.distractor_generator import DistractorGenerator
from pipelines.implemented.embeddings_reducer import EmbeddingsReducer
from pipelines.implemented.one_step import OneStepPipeline
from pipelines.implemented.question_answer_generator import QuestionAnswerGenerator
from pipelines.implemented.semantic_text_chunker import SemanticTextChunker
from pipelines.implemented.sentence_splitter import SentenceSplitter
from pipelines.implemented.sentence_to_vec import Sentence2Vec
from pipelines.implemented.text_chunker import TextChunker

class PipelinePage:
    def __init__(self, pipeline: Pipeline):
        self.title = pipeline.title
    def page(self):
        st.write(f"# {self.title}")

@st.cache_resource
def load_pipelines():
    ALL_PIPELINE_CLASSES = [ConceptClusterCombiner, ConceptCombiner, ConceptExtractor
        , DistractorGenerator, EmbeddingsReducer, OneStepPipeline,
        QuestionAnswerGenerator, SemanticTextChunker, SentenceSplitter,
        Sentence2Vec, TextChunker]
    ALL_PIPELINE_INSTANCES = [cls() for cls in ALL_PIPELINE_CLASSES]
    ALL_PIPELINE_PAGES = [st.Page(PipelinePage(pipeline).page,
                                  title=pipeline.title, url_path=pipeline.title)
        for pipeline in ALL_PIPELINE_CLASSES]
    return ALL_PIPELINE_CLASSES, ALL_PIPELINE_INSTANCES, ALL_PIPELINE_PAGES

ALL_PIPELINE_CLASSES, ALL_PIPELINE_INSTANCES, ALL_PIPELINE_PAGES = load_pipelines()

def quiz_page():
    st.write("Quiz Page")

pg = st.navigation([*ALL_PIPELINE_PAGES, quiz_page])

pg.run()


