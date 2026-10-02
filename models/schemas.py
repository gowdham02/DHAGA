"""Pydantic schemas used to validate every live LLM response."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ReturnCategory(str, Enum):
    FIT = "FIT"
    QUALITY = "QUALITY"
    PRODUCT = "PRODUCT"
    DELIVERY = "DELIVERY"
    OTHER = "OTHER"


class FitSubcategory(str, Enum):
    TOO_TIGHT = "TOO_TIGHT"
    TOO_LOOSE = "TOO_LOOSE"
    SIZE_MISMATCH = "SIZE_MISMATCH"
    SHOULDER_FIT = "SHOULDER_FIT"
    LENGTH_ISSUE = "LENGTH_ISSUE"
    OTHER_FIT = "OTHER_FIT"


class ReturnClassification(BaseModel):
    """Classification produced by the inexpensive first model."""

    category: ReturnCategory
    subcategory: str = Field(description="FIT detail when category is FIT; otherwise a concise label.")
    confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=3, max_length=500)

    @field_validator("subcategory")
    @classmethod
    def normalize_subcategory(cls, value: str) -> str:
        value = value.strip().upper().replace(" ", "_")
        if not value:
            raise ValueError("subcategory cannot be empty")
        return value


class FitAnalysis(BaseModel):
    """Detailed classification created only for returns routed to FIT."""

    category: Literal[ReturnCategory.FIT] = ReturnCategory.FIT
    subcategory: FitSubcategory
    confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=3, max_length=500)


class EvaluationResult(BaseModel):
    """Independent evaluator verdict from the stronger model."""

    approved: bool
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=3, max_length=500)
    requires_human_review: bool

    @model_validator(mode="after")
    def rejected_cases_require_review(self) -> "EvaluationResult":
        if not self.approved and not self.requires_human_review:
            raise ValueError("A rejected classification must require human review")
        return self
