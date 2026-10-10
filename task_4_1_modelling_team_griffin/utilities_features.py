# imports.
import numpy as np
import polars as pl

# Features list.
FEATURES = [
    "spread",
    "obi",
    "dist_to_mid",
    "hour",
    "vpin_5s",
    "vpin_30s",
    "vpin_120s",
    "rv_5s",
    "rv_30s",
    "rv_120s",
    "liq_binance_5s",
    "liq_binance_30s",
    "liq_binance_120s",
    "liq_bybit_5s",
    "liq_bybit_30s",
    "liq_bybit_120s",
    "sign",
    "log_notional",
    "net_liq_5s",
    "net_liq_30s",
    "net_liq_120s",
    "liq_agreement_30s"
]

# derived features.
def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add derived features that are cheap to compute from existing columns.
    Called once per enriched DataFrame before splitting into windows.
    """
    return df.with_columns([
        pl.col("notional").log1p().cast(pl.Float32).alias("log_notional"),

        (pl.col("liq_binance_5s") + pl.col("liq_bybit_5s")).cast(pl.Float32).alias("net_liq_5s"),

        (pl.col("liq_binance_30s") + pl.col("liq_bybit_30s")).cast(pl.Float32).alias("net_liq_30s"),

        (pl.col("liq_binance_120s") + pl.col("liq_bybit_120s")).cast(pl.Float32).alias("net_liq_120s"),

        (pl.col("liq_binance_30s").sign() * pl.col("liq_bybit_30s").sign()).cast(pl.Float32).alias("liq_agreement_30s")
    ])

def extract_Xyw(
    df: pl.DataFrame,
    tau: int,
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract X, y, w, ts from a feature-enriched DataFrame.

    Rows with null or NaN pnl are dropped.
    Feature NaNs are preserved.
    """
    if features is None:
        features = FEATURES

    pnl_col = f"pnl_{tau}s"

    required_cols = [pnl_col, "weight", "timestamp"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError("Missing required columns: " + ", ".join(missing_required))

    valid = df.filter(
        pl.col(pnl_col).is_not_null() & pl.col(pnl_col).is_not_nan()
    )

    X = valid.select(features).to_numpy().astype(np.float32)
    y = valid[pnl_col].to_numpy().astype(np.float32)
    w = valid["weight"].to_numpy().astype(np.float32)
    ts = valid["timestamp"].to_numpy().astype(np.int64)

    return X, y, w, ts

