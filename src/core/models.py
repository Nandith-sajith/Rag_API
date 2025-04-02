from pydantic import BaseModel

class PromptRequest(BaseModel):
    query: str

class PromptResponse(BaseModel):
    answer: str
    confidence: float
    evaluation: dict