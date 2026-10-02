from io import BytesIO

import pandas as pd
import pytest

from services.data_processor import CsvValidationError, read_and_validate_csv


REQUIRED = ["order_id", "sku", "product_name", "vendor", "size", "return_reason", "return_reason_detail"]


class Upload(BytesIO):
    def getvalue(self):
        return super().getvalue()


def test_csv_validation_and_duplicate_removal():
    content = (",".join(REQUIRED) + "\n" + "1,S1,Top,V1,M,Other,tight hai\n" + "1,S1,Top,V1,M,Other,tight hai\n").encode()
    df, report = read_and_validate_csv(Upload(content))
    assert len(df) == 1
    assert report.duplicate_rows_removed == 1
    assert df.loc[0, "original_return_text"] == "Other | tight hai"


def test_missing_columns_are_explained():
    with pytest.raises(CsvValidationError, match="Missing required columns"):
        read_and_validate_csv(Upload(b"order_id,sku\n1,S1\n"))
