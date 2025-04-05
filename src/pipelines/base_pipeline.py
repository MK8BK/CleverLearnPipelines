
"""
    Prompt engineering guide:
        https://platform.openai.com/docs/guides/prompt-engineering
"""

from abc import ABC, abstractmethod
from pathlib import Path
from json import loads
import os
from corpus.corpus import CorpusLanguage


class PipelineValidationError(RuntimeError):
    def __init__(self, *args):
        super().__init__(*args)


PIPELINE_PROMPTS_PATH = Path(__file__).parent.joinpath(
    "implemented").joinpath("prompts")
PIPELINE_NAMES = set(p.split(".")[0]
                     for p in os.listdir(PIPELINE_PROMPTS_PATH))


class Pipeline(ABC):
    def __init__(self, pipeline_title: str, language: CorpusLanguage = CorpusLanguage.EN):
        if pipeline_title not in PIPELINE_NAMES:
            raise RuntimeError(f"no file {pipeline_title}.json exists")
        self.title = pipeline_title
        self.language = language

        prompts_path = PIPELINE_PROMPTS_PATH.joinpath(self.title+".json")
        with open(prompts_path, "r", encoding="utf8") as f:
            self.prompt_store = loads(f.read())

    def process(self, data):
        """
            Process the input data, type unspecified
        """
        output_data = self._process(data)
        valid = self._validate(data, output_data)
        if not valid:
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
