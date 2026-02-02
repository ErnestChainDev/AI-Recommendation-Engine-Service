from pydantic import BaseModel, Field

class RecommendIn(BaseModel):
    user_id: int
    attempt_id: int
    score: int = Field(ge=0)
    total: int = Field(gt=0)

    # optional: category breakdown (future-proof)
    logic: int = 0
    programming: int = 0
    networking: int = 0
    design: int = 0

class RecommendOut(BaseModel):
    program: str
    confidence: int
    rationale: str