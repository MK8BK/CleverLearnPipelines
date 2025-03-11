from typing import Optional, List, Annotated
from annotated_types import Len
from pydantic import BaseModel 

class MultipleChoiceQuestion(BaseModel):
    """ an MCQ standard form class
    """
    question: str # stem
    answer: str # key
    distractors: List[str]

class Quiz(BaseModel):
    mcqs: List[MultipleChoiceQuestion]

