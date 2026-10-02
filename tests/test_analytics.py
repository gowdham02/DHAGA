import pandas as pd

from services.analytics import calculate_analytics


def test_analytics_calculates_fit_rates_and_review_queue():
    records = pd.DataFrame([
        {"order_id": "1", "sku": "S1", "product_name": "Top", "vendor": "V1", "size": "M", "category": "FIT", "subcategory": "TOO_TIGHT", "requires_human_review": False},
        {"order_id": "2", "sku": "S1", "product_name": "Top", "vendor": "V1", "size": "L", "category": "QUALITY", "subcategory": "QUALITY_ISSUE", "requires_human_review": True},
        {"order_id": "3", "sku": "S2", "product_name": "Dress", "vendor": "V2", "size": "S", "category": "FIT", "subcategory": "TOO_LOOSE", "requires_human_review": False},
    ])
    result = calculate_analytics(records)
    assert result["fit_returns"] == 2
    assert result["fit_return_rate"] == 2 / 3
    assert result["human_review_count"] == 1
    assert result["sku_table"].iloc[0]["sku"] == "S2"
