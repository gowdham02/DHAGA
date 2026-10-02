import pytest
from pydantic import ValidationError

from chains.router import build_return_router
from models.schemas import EvaluationResult, ReturnClassification


def test_pydantic_classification_normalizes_subcategory():
    result = ReturnClassification(category="FIT", subcategory="too tight", confidence=0.9, explanation="The garment is tight.")
    assert result.subcategory == "TOO_TIGHT"


def test_rejected_evaluation_must_require_review():
    with pytest.raises(ValidationError):
        EvaluationResult(approved=False, confidence=0.3, reason="Unsupported by text.", requires_human_review=False)


def test_runnable_branch_routes_fit_only():
    router = build_return_router(lambda payload: {**payload, "route": "FIT_ANALYSIS"})
    fit = router.invoke({"classification": ReturnClassification(category="FIT", subcategory="FIT", confidence=0.8, explanation="fit issue")})
    other = router.invoke({"classification": ReturnClassification(category="OTHER", subcategory="UNCLEAR", confidence=0.5, explanation="unclear")})
    assert fit["route"] == "FIT_ANALYSIS"
    assert other["route"] == "NON_FIT"
