from pydantic import BaseModel, Field
from typing import List, Optional

class ActionCandidate(BaseModel):
    action: str = Field(
        description="One of ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7"
    )
    x: Optional[int] = Field(
        default=None,
        description="x coordinate if action is ACTION6, else null"
    )
    y: Optional[int] = Field(
        default=None,
        description="y coordinate if action is ACTION6, else null"
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Short explanation for why this action may help"
    )

class CandidateActions(BaseModel):
    candidates: List[ActionCandidate] = Field(
        description="Top 3 candidate next actions"
    )
