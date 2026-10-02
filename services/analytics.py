"""Pandas-only dashboard calculations; no LLM is used for aggregation."""

from __future__ import annotations

import pandas as pd

from chains.config import ModelConfig
from services.data_processor import ProcessingSummary


FIT_ISSUES = ["TOO_TIGHT", "TOO_LOOSE", "SIZE_MISMATCH", "SHOULDER_FIT", "LENGTH_ISSUE", "OTHER_FIT"]
CATEGORIES = ["FIT", "QUALITY", "PRODUCT", "DELIVERY", "OTHER", "FAILED"]


def calculate_analytics(results: pd.DataFrame) -> dict:
    total = len(results)
    fit_mask = results["category"].eq("FIT")
    fit_total = int(fit_mask.sum())
    category_distribution = results["category"].value_counts().reindex(CATEGORIES, fill_value=0).rename_axis("category").reset_index(name="returns")
    fit_distribution = (results.loc[fit_mask, "subcategory"].value_counts().reindex(FIT_ISSUES, fill_value=0)
                        .rename_axis("fit_issue").reset_index(name="returns"))

    sku = results.assign(fit_return=fit_mask).groupby(["sku", "product_name"], as_index=False).agg(
        total_returns=("order_id", "size"), fit_related_returns=("fit_return", "sum")
    )
    sku["fit_return_rate"] = sku["fit_related_returns"] / sku["total_returns"]
    sku = sku.sort_values(["fit_return_rate", "fit_related_returns"], ascending=False).reset_index(drop=True)
    sku_issues = (results.loc[fit_mask].groupby(["sku", "subcategory"]).size().reset_index(name="count")
                  .sort_values(["sku", "count"], ascending=[True, False]).drop_duplicates("sku")
                  .set_index("sku")["subcategory"].to_dict())

    vendor = results.assign(fit_return=fit_mask).groupby("vendor", as_index=False).agg(
        total_returns=("order_id", "size"), fit_related_returns=("fit_return", "sum")
    )
    vendor["fit_return_rate"] = vendor["fit_related_returns"] / vendor["total_returns"]
    vendor = vendor.sort_values(["fit_return_rate", "fit_related_returns"], ascending=False).reset_index(drop=True)
    vendor_issues = (results.loc[fit_mask].groupby(["vendor", "subcategory"]).size().reset_index(name="count")
                     .sort_values(["vendor", "count"], ascending=[True, False]).drop_duplicates("vendor")
                     .set_index("vendor")["subcategory"].to_dict())

    review = results[results["requires_human_review"].fillna(True)].copy()
    return {
        "total_returns": total,
        "fit_returns": fit_total,
        "fit_return_rate": (fit_total / total) if total else 0,
        "human_review_count": int(len(review)),
        "category_distribution": category_distribution,
        "fit_distribution": fit_distribution,
        "sku_table": sku,
        "vendor_table": vendor,
        "sku_dominant_issues": sku_issues,
        "vendor_dominant_issues": vendor_issues,
        "review_queue": review,
    }


def actionable_insights(analytics: dict, limit: int = 3) -> list[str]:
    insights: list[str] = []
    results_reviewed = analytics["total_returns"]
    for _, row in analytics["sku_table"].head(limit).iterrows():
        sku = row["sku"]
        sku_fit = row["fit_related_returns"]
        # We only recommend investigation; results do not establish causality.
        insights.append(
            f"{sku} ({row['product_name']}) has a {row['fit_return_rate']:.0%} fit-related return rate "
            f"({sku_fit}/{row['total_returns']} analyzed returns); the dominant issue is "
            f"{analytics['sku_dominant_issues'].get(sku, 'OTHER_FIT')}. Review its size mapping and product measurements."
        )
    for _, row in analytics["vendor_table"].head(min(2, limit)).iterrows():
        insights.append(
            f"Vendor {row['vendor']} has a {row['fit_return_rate']:.0%} fit-related return rate across "
            f"{row['total_returns']} analyzed returns; the dominant issue is "
            f"{analytics['vendor_dominant_issues'].get(row['vendor'], 'OTHER_FIT')}. Investigate whether its size chart aligns with actual measurements."
        )
    return insights if results_reviewed else ["No analyzed return records are available yet."]


def estimate_cost(summary: ProcessingSummary, config: ModelConfig) -> str | None:
    """Estimate cost only when the user provides both input/output provider rates.

    Token figures are disclosed assumptions for planning, not claimed provider prices.
    """
    classifier_rates = (config.classifier_input_cost_per_million, config.classifier_output_cost_per_million)
    evaluator_rates = (config.evaluator_input_cost_per_million, config.evaluator_output_cost_per_million)
    if any(rate is None for rate in (*classifier_rates, *evaluator_rates)):
        return None
    # Planning assumptions: 250 input / 80 output tokens for classifier+fit; 350/100 evaluator.
    c_cost = ((summary.classifier_calls + summary.fit_calls) * ((250 * classifier_rates[0] + 80 * classifier_rates[1]) / 1_000_000))
    e_cost = summary.evaluator_calls * ((350 * evaluator_rates[0] + 100 * evaluator_rates[1]) / 1_000_000)
    return f"Estimated USD ${c_cost + e_cost:.4f} using configured rates and disclosed planning token assumptions."
