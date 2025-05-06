"""
    Prompt engineering guide:
        https://platform.openai.com/docs/guides/prompt-engineering
"""

from abc import ABC, abstractmethod
from pathlib import Path
from json import loads
from time import time
import os
from corpus.corpus import CorpusLanguage
from logutils import get_logger


class PipelineValidationError(RuntimeError):
    def __init__(self, *args):
        super().__init__(*args)


PIPELINE_PROMPTS_PATH = Path(__file__).parent.joinpath(
    "implemented").joinpath("prompts")

PIPELINE_NAMES = set(p.split(".")[0]
                     for p in os.listdir(PIPELINE_PROMPTS_PATH))


class Pipeline(ABC):
    def __init__(self, pipeline_title: str, language: CorpusLanguage = CorpusLanguage.EN
                 ,extra_context: dict=None):
        if pipeline_title not in PIPELINE_NAMES:
            raise RuntimeError(f"no file {pipeline_title}.json exists")
        self.title = pipeline_title
        self.language = language
        self.context = extra_context

        prompts_path = PIPELINE_PROMPTS_PATH.joinpath(self.title+".json")
        with open(prompts_path, "r", encoding="utf8") as f:
            self.prompt_store = loads(f.read())
        self.logger = get_logger(self.title)

    def set_context(self, extra_context: dict):
        self.context = extra_context
    def process(self, data):
        """
            Process the input data, type unspecified
        """
        start = time()
        output_data = self._process(data)
        valid = self._validate(data, output_data)
        end = time()
        duration = end-start
        self.logger.info(f"execution time {duration}s")
        if not valid:
            self.logger.error("invalid pipeline output")
            raise PipelineValidationError(f"[{self.__class__}]-[{self.title}]")
        return output_data

    @abstractmethod
    def _process(self, input_data):
        pass

    @abstractmethod
    def _validate(self, input_data, output_data) -> bool:
        """
            Validate the output_data given the input_data
            i.e. the processing yielded sensible results
        """
        pass
