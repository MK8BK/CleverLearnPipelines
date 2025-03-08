from typing import Optional, List
from pydantic import BaseModel, conlist

class MultipleChoiceQuestion(BaseModel):
    """ an MCQ standard form class
    """
    question: str # stem
    answer: str # key
    distractors: List[str]
    # distractors: conlist(str, min_length=1)
    # https://docs.pydantic.dev/latest/api/types/#pydantic.types.conlist

class Quiz(BaseModel):
    mcqs: List[MultipleChoiceQuestion]
    # mcqs: conlist(MultipleChoiceQuestion, min_length=1)

