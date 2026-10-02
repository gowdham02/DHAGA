"""Generate clearly labelled synthetic return data for DHAGA FITLENS demos."""

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd


RNG = random.Random(42)
PRODUCTS = [
    ("SKU101", "Aarna Straight Kurti", "V1 North Looms"),
    ("SKU102", "Mira Cotton Kurti", "V1 North Looms"),
    ("SKU103", "Nila Anarkali Kurti", "V2 Jaipur Textiles"),
    ("SKU104", "Riya Printed Shirt", "V2 Jaipur Textiles"),
    ("SKU105", "Tara Linen Shirt", "V3 Urban Weave"),
    ("SKU106", "Kavya Palazzo Set", "V3 Urban Weave"),
    ("SKU107", "Meher Festive Kurta", "V4 Saffron Studio"),
    ("SKU108", "Ira Everyday Top", "V4 Saffron Studio"),
    ("SKU109", "Zoya Flared Dress", "V5 Coastline Apparel"),
    ("SKU110", "Sana Ribbed Top", "V5 Coastline Apparel"),
    ("SKU111", "Avni Denim Jacket", "V6 Indigo Works"),
    ("SKU112", "Diya Summer Co-ord", "V6 Indigo Works"),
    ("SKU113", "Leela Rayon Kurti", "V7 Thread Theory"),
    ("SKU114", "Pihu Casual Tee", "V7 Thread Theory"),
    ("SKU115", "Anaya Office Trousers", "V8 Form & Fit"),
    ("SKU116", "Veda Oversized Shirt", "V8 Form & Fit"),
]

COMMENTS = {
    "FIT": [
        "kurti bahut tight hai", "size is smaller than expected", "shoulder romba tight", "not fitting properly",
        "shirt loose hai", "length is too short", "size chart didn't match", "sleeves are too short",
        "M size bhi chota laga", "fit is weird around the chest", "bahut bada ho gaya", "shoulder fitting off hai",
        "waist tight but rest okay", "actual measurement chart se alag hai", "armhole feels tight", "too long for me",
    ],
    "QUALITY": [
        "material quality not good", "stitching came undone after one wear", "fabric feels very cheap", "thread coming out",
        "colour faded on first wash", "received with a small tear", "cloth quality bilkul achi nahi", "zip is defective",
    ],
    "PRODUCT": [
        "colour different from image", "design looked different online", "wrong product received", "print is not as shown",
        "image wala shade nahi hai", "product looks different in real", "style did not suit me", "pattern is misleading",
    ],
    "DELIVERY": [
        "delivery came late", "parcel arrived after the event", "courier delayed too much", "delivery bahut late hui",
        "package was delivered to wrong place", "received after expected date", "tracking never updated", "late delivery so returning",
    ],
    "OTHER": [
        "not happy", "size feels weird", "didn't like it", "return please", "okay but no", "looks odd", "change my mind", "hmm not sure",
    ],
}


def generate_records(record_count: int = 650) -> pd.DataFrame:
    """Create diverse synthetic records; high-fit SKUs intentionally support analysis demos."""
    rows = []
    for index in range(record_count):
        sku, product, vendor = RNG.choice(PRODUCTS)
        # S101, S103, S115 have deliberately elevated fit issues for dashboard discovery.
        fit_bias = 0.62 if sku in {"SKU101", "SKU103", "SKU115"} else (0.38 if vendor == "V8 Form & Fit" else 0.24)
        roll = RNG.random()
        if roll < fit_bias:
            actual = "FIT"
        elif roll < fit_bias + 0.18:
            actual = "QUALITY"
        elif roll < fit_bias + 0.32:
            actual = "PRODUCT"
        elif roll < fit_bias + 0.42:
            actual = "DELIVERY"
        else:
            actual = "OTHER"
        detail = RNG.choice(COMMENTS[actual])
        if RNG.random() < 0.35:
            detail += RNG.choice([" pls help", " yaar", " :( ", "", " for my usual size", " totally unexpected"])
        if RNG.random() < 0.035:
            detail = ""  # realistic weak/missing free text
        # Roughly 44% retain the business-problem dropdown value of Other.
        dropdown = "Other" if RNG.random() < 0.44 else actual.title()
        rows.append(
            {
                "order_id": f"SYN-{100000 + index}",
                "sku": sku,
                "product_name": product,
                "vendor": vendor,
                "size": RNG.choice(["XS", "S", "M", "L", "XL", "XXL"]),
                "return_reason": dropdown,
                "return_reason_detail": detail,
                "data_source": "SYNTHETIC DEMO DATA",
                "synthetic_actual_theme": actual,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    path = Path(__file__).with_name("demo_returns.csv")
    data = generate_records()
    data.to_csv(path, index=False)
    print(f"Generated {len(data)} SYNTHETIC DEMO DATA records at {path}")


if __name__ == "__main__":
    main()
