"""Safe CSV validation/cleaning and orchestration of the LangChain workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

from chains.classifier import build_classifier_chain
from chains.config import ModelConfig
from chains.evaluator import build_evaluator_chain
from chains.fit_analyzer import build_fit_analysis_chain
from chains.router import build_return_router
from models.schemas import EvaluationResult, FitAnalysis, FitSubcategory, ReturnCategory, ReturnClassification


REQUIRED_COLUMNS = {
    "order_id", "sku", "product_name", "vendor", "size", "return_reason", "return_reason_detail"
}
TEXT_COLUMNS = {"return_reason", "return_reason_detail"}


class CsvValidationError(ValueError):
    """A user-facing, recoverable problem with an uploaded CSV."""


@dataclass
class CleaningReport:
    input_rows: int
    duplicate_rows_removed: int
    blank_text_rows: int


@dataclass
class ProcessingSummary:
    processed: int = 0
    failed: int = 0
    classifier_calls: int = 0
    fit_calls: int = 0
    evaluator_calls: int = 0
    mode: str = ""

    @property
    def total_llm_calls(self) -> int:
        return self.classifier_calls + self.fit_calls + self.evaluator_calls


def read_and_validate_csv(uploaded: Any) -> tuple[pd.DataFrame, CleaningReport]:
    """Load, validate, normalize and de-duplicate user CSV data using Pandas only."""
    try:
        raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        if not raw:
            raise CsvValidationError("The uploaded CSV is empty.")
        df = pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False)
    except UnicodeDecodeError as exc:
        raise CsvValidationError("Could not decode this CSV. Please upload a UTF-8 encoded file.") from exc
    except pd.errors.EmptyDataError as exc:
        raise CsvValidationError("The uploaded CSV has no header or records.") from exc
    except Exception as exc:
        if isinstance(exc, CsvValidationError):
            raise
        raise CsvValidationError(f"Could not read CSV: {exc}") from exc

    df.columns = [str(column).strip() for column in df.columns]
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise CsvValidationError("Missing required columns: " + ", ".join(missing))
    if df.empty:
        raise CsvValidationError("The uploaded CSV has a header but no return records.")

    input_rows = len(df)
    for column in REQUIRED_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()
    duplicate_rows_removed = int(df.duplicated(subset=list(REQUIRED_COLUMNS), keep="first").sum())
    df = df.drop_duplicates(subset=list(REQUIRED_COLUMNS), keep="first").copy()
    blank_text_rows = int((df["return_reason"].eq("") & df["return_reason_detail"].eq("")).sum())
    df["original_return_text"] = df.apply(
        lambda row: " | ".join(part for part in [row["return_reason"], row["return_reason_detail"]] if part), axis=1
    )
    return df.reset_index(drop=True), CleaningReport(input_rows, duplicate_rows_removed, blank_text_rows)


def _text(payload: dict) -> str:
    return f"{payload.get('return_reason', '')} {payload.get('return_reason_detail', '')}".lower()


def _demo_classify(payload: dict) -> ReturnClassification:
    """Transparent deterministic fallback for the offline synthetic walkthrough."""
    text = _text(payload)
    if not text.strip():
        return ReturnClassification(category="OTHER", subcategory="MISSING_TEXT", confidence=0.0,
                                    explanation="No return reason text was supplied.")
    categories = {
        "DELIVERY": ["late", "delivery", "delivered", "courier", "shipping", "parcel", "delay", "arrived"],
        "QUALITY": ["quality", "stitch", "tear", "torn", "fabric", "material", "defect", "damaged", "fade", "colour bleed"],
        "PRODUCT": ["colour", "color", "image", "different", "wrong product", "not as shown", "design"],
        "FIT": ["tight", "loose", "fit", "fitting", "small", "smaller", "large", "bada", "chota", "chhota", "size", "shoulder", "sleeve", "length", "chart", "romba"],
    }
    matched = {category: sum(term in text for term in terms) for category, terms in categories.items()}
    category = max(matched, key=matched.get)
    if matched[category] == 0:
        return ReturnClassification(category="OTHER", subcategory="UNCLEAR", confidence=0.38,
                                    explanation="The comment does not state a clear return issue.")
    confidence = 0.88 if matched[category] > 1 else 0.72
    subcategory = "FIT_RELATED" if category == "FIT" else f"{category}_ISSUE"
    return ReturnClassification(category=category, subcategory=subcategory, confidence=confidence,
                                explanation=f"Demo rules found {category.lower()}-related language in the comment.")


def _demo_fit(payload: dict) -> FitAnalysis:
    text = _text(payload)
    if "shoulder" in text:
        label, confidence = "SHOULDER_FIT", 0.90
    elif any(term in text for term in ["length", "sleeve", "short", "long", "lamba"]):
        label, confidence = "LENGTH_ISSUE", 0.88
    elif any(term in text for term in ["tight", "chota", "chhota", "small"]):
        label, confidence = "TOO_TIGHT", 0.90
    elif any(term in text for term in ["loose", "bada", "large"]):
        label, confidence = "TOO_LOOSE", 0.88
    elif any(term in text for term in ["chart", "size", "expected"]):
        label, confidence = "SIZE_MISMATCH", 0.76
    else:
        label, confidence = "OTHER_FIT", 0.48
    return FitAnalysis(subcategory=label, confidence=confidence,
                       explanation=f"Demo fit rules identified {label.lower()} signals.")


def _demo_evaluate(payload: dict) -> EvaluationResult:
    confidence = float(payload["classifier_confidence"])
    weak = not _text(payload).strip() or confidence < 0.65 or payload["subcategory"] in {"OTHER_FIT", "UNCLEAR"}
    return EvaluationResult(
        approved=not weak,
        confidence=0.45 if weak else min(0.97, confidence + 0.05),
        reason="Ambiguous or weak evidence needs human review." if weak else "The proposed classification is supported by the comment.",
        requires_human_review=weak,
    )


def _invoke_with_retry(chain: Any, payload: dict, schema: type, attempts: int = 2):
    error: Exception | None = None
    for _ in range(attempts):
        try:
            result = chain.invoke(payload)
            return result if isinstance(result, schema) else schema.model_validate(result)
        except (ValidationError, Exception) as exc:  # live provider/format errors are surfaced per record
            error = exc
    raise RuntimeError(f"Structured-output validation failed after {attempts} attempts: {error}")


def analyze_returns(df: pd.DataFrame, config: ModelConfig, progress_callback=None) -> tuple[pd.DataFrame, ProcessingSummary]:
    """Run classification → RunnableBranch routing → evaluation for each return record."""
    summary = ProcessingSummary(mode="Live LangChain" if config.has_live_llm else "Deterministic demo (no LLM key)")
    if config.has_live_llm and config.classifier_model == config.evaluator_model:
        raise ValueError("CLASSIFIER_MODEL and EVALUATOR_MODEL must be different models.")

    classifier = build_classifier_chain(config) if config.has_live_llm else None
    fit_chain = build_fit_analysis_chain(config) if config.has_live_llm else None
    evaluator = build_evaluator_chain(config) if config.has_live_llm else None

    def fit_handler(payload: dict) -> dict:
        fit_result = _invoke_with_retry(fit_chain, payload, FitAnalysis) if fit_chain else _demo_fit(payload)
        summary.fit_calls += 1
        return {**payload, "fit_analysis": fit_result, "route": "FIT_ANALYSIS"}

    router = build_return_router(fit_handler)
    output_rows: list[dict] = []
    total = len(df)
    for index, row in df.iterrows():
        payload = row.to_dict()
        try:
            if not str(payload.get("original_return_text", "")).strip():
                classification = ReturnClassification(category="OTHER", subcategory="MISSING_TEXT", confidence=0,
                                                      explanation="No return reason text was supplied.")
            else:
                classification = _invoke_with_retry(classifier, payload, ReturnClassification) if classifier else _demo_classify(payload)
                summary.classifier_calls += 1
            routed = router.invoke({**payload, "classification": classification})
            final = routed.get("fit_analysis", classification)
            evaluator_payload = {
                **payload,
                "category": final.category.value if hasattr(final.category, "value") else str(final.category),
                "subcategory": final.subcategory.value if hasattr(final.subcategory, "value") else str(final.subcategory),
                "classifier_confidence": final.confidence,
                "explanation": final.explanation,
            }
            evaluation = _invoke_with_retry(evaluator, evaluator_payload, EvaluationResult) if evaluator else _demo_evaluate(evaluator_payload)
            summary.evaluator_calls += 1
            review = evaluation.requires_human_review or final.confidence < 0.65 or not evaluation.approved
            output_rows.append({**payload, "category": final.category.value if hasattr(final.category, "value") else str(final.category),
                                "subcategory": final.subcategory.value if hasattr(final.subcategory, "value") else str(final.subcategory),
                                "classification_confidence": final.confidence, "classification_explanation": final.explanation,
                                "route": routed["route"], "evaluation_status": "APPROVED" if evaluation.approved else "REVIEW_REQUIRED",
                                "evaluator_confidence": evaluation.confidence, "evaluator_reason": evaluation.reason,
                                "requires_human_review": review, "processing_error": ""})
            summary.processed += 1
        except Exception as exc:
            summary.failed += 1
            output_rows.append({**payload, "category": "FAILED", "subcategory": "FAILED", "classification_confidence": 0.0,
                                "classification_explanation": "", "route": "FAILED", "evaluation_status": "FAILED",
                                "evaluator_confidence": 0.0, "evaluator_reason": str(exc), "requires_human_review": True,
                                "processing_error": str(exc)})
        if progress_callback:
            progress_callback((index + 1) / total, index + 1, total)
    return pd.DataFrame(output_rows), summary
