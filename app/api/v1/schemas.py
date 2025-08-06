from pydantic import BaseModel, HttpUrl
from typing import List

class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerResponse(BaseModel):
    answer: str
    source_clauses: List[str]
    rationale: str

class RunResponse(BaseModel):
    answers: List[AnswerResponse]