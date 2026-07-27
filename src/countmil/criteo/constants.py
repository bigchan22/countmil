"""Constants for the Criteo_x1 strict-v1 feature-bag experiment."""

from __future__ import annotations

PROTOCOL_TAG = "criteo_x1_c4_c11_k128_strictv1"

EXPECTED_ZIP_SHA256 = "ad87602d1a0c855a234da10f4510ba058990247380e961083f7d60b8bff62608"
EXPECTED_CSV_MD5 = {
    "train": "30b89c1c7213013b92df52ec44f52dc5",
    "valid": "f73c71fb3c4f66b6ebdfa032646bea72",
    "test": "2c48b26e84c04a69b948082edae46f8c",
}
EXPECTED_ROWS = {"train": 33_003_326, "valid": 8_250_124, "test": 4_587_167}

NUMERIC_COLS = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
EXPECTED_COLUMNS = ["label", *ALL_FEATURE_COLS]

GROUP_COLS = ["C4", "C11"]
BAG_SIZE = 128
SPLIT_BAGS = {"train": 12_000, "valid": 3_000, "test": 5_000}
SEEDS = [0, 1, 2]

