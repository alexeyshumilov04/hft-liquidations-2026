# imports.
import datetime
import numpy as np
import polars as pl
from dataclasses import dataclass


# Contants.
HORIZONS_S           = [30, 120, 300]    # seconds.
REBATE_BPS          = 0.5               
CLIPPED_NOTIONAL    = 100_000           # USD cap for weight.
BYBIT_DELAY_US      = 200_000           # microseconds(200 ms).
TURNOVER_CONSTRAINT = 500_000           # USD per day minimum.
US_PER_SEC          = 1_000_000
US_PER_DAY          = 86_400 * US_PER_SEC


# Markout preparation.
def prepare_mid(
    lf_bbo : pl.LazyFrame
) -> pl.DataFrame:
    """
    Extract mid_price from BBO (best bid/offer) data.
 
    Returns DataFrame with columns: [timestamp, mid_price].
    """
    df = (
        lf_bbo.select(
            pl.col('timestamp'),
            ((pl.col('bid_price') + pl.col('ask_price')) / 2).cast(pl.Float32).alias('mid_price')
        )
        .sort("timestamp")
        .collect(engine='streaming')
    )

    return df

def prepare_trades(
    lf_trades: pl.LazyFrame
) -> pl.DataFrame:
    """
    Encode trade sign and compute notional / weight.
 
    Returns DataFrame with columns:
        [timestamp, price, sign, notional, weight].
    """
    df = (
        lf_trades
        .select(
            pl.col("timestamp"),
            pl.col("price").cast(pl.Float32),
 
            # taker side → sign (+1 taker buy / -1 taker sell).
            pl.when(pl.col("side") == "buy")
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(-1, dtype=pl.Int8))
            .alias("sign"),
 
            # notional = price × amount.
            (pl.col("price") * pl.col("amount")).cast(pl.Float32).alias("notional"),

            (pl.col("price") * pl.col("amount")).clip(upper_bound=CLIPPED_NOTIONAL).alias("weight")
        )
        .sort("timestamp")
        .collect(engine="streaming")
    )
    return df

def compute_markout(
    df_trades: pl.DataFrame,
    df_mid: pl.DataFrame,
    horizons_s: list[int] | None = None
) -> pl.DataFrame:
    """
    For each trade compute maker PnL at each horizon t.
 
    Logic per horizon:
      1. lookup_ts = trade.timestamp + t * 1e6.
      2. mid(t) = last BBO mid <= lookup_ts  (asof backward = forward-fill).
      3. pnl(t) = -sign * (mid(t) - price) / price * 10_000 + REBATE.
      4. If lookup_ts exceeds last BBO timestamp -> pnl = null (excluded).
 
    Returns df_trades with added columns: pnl_30s, pnl_120s, pnl_300s.
    """
    if horizons_s is None:
        horizons_s = HORIZONS_S
 
    max_bbo_ts  = df_mid["timestamp"].max()
    result      = df_trades.clone()
 
    for tau in horizons_s:
        tau_us      = tau * US_PER_SEC
        col_name    = f"pnl_{tau}s"
 
        # add lookup timestamp.
        df_lookup = result.with_columns(
            (pl.col("timestamp") + tau_us).alias("lookup_ts")
        )
 
        # asof join: find last mid <= lookup_ts.
        df_joined = df_lookup.join_asof(
            df_mid.rename({"timestamp": "mid_ts"}),
            left_on     = "lookup_ts",
            right_on    = "mid_ts",
            strategy    = "backward"
        )
 
        # compute PnL; null out trades where lookup_ts > max BBO.
        pnl = (
            -df_joined["sign"].cast(pl.Float32)
            * (df_joined["mid_price"] - df_joined["price"]) / df_joined["price"]
            * 10_000
            + REBATE_BPS
        )
 
        # mask trades beyond BBO range.
        mask_valid = df_lookup["lookup_ts"] <= max_bbo_ts
        pnl = pl.when(mask_valid).then(pnl).otherwise(None).alias(col_name)
 
        result = result.with_columns(pnl)
 
    return result

def build_markout_table(
    lf_trades: pl.LazyFrame,
    lf_bbo: pl.LazyFrame,
    horizons_s: list[int] | None = None
) -> pl.DataFrame:
    """
    End-to-end: raw trades + BBO -> markout table.
 
    Returns DataFrame with columns:
        [timestamp, price, sign, notional, weight, pnl_30s, pnl_120s, pnl_300s].
    """
    df_mid      = prepare_mid(lf_bbo)
    df_trades   = prepare_trades(lf_trades)
    df          = compute_markout(df_trades, df_mid, horizons_s)

    return df


# Score calculation.
@dataclass
class ScoreResult:
    """Container for all scoring outputs at one horizon."""
    tau: int
    pnl_all: float
    pnl_kept: float
    pnl_filtered: float
    score: float
    kept_turnover_total: float
    kept_turnover_per_day: float
    n_days: float
    n_total: int
    n_kept: int
    n_filtered: int
    filter_rate: float                 # fraction of trades filtered.
    turnover_ok: bool                  # constraint check.

def compute_score(
    df_markout: pl.DataFrame,
    f: np.ndarray | pl.Series,
    tau: int
) -> ScoreResult:
    """
    Compute Score(t) given a markout table and binary filter.
 
    Parameters
    ----------
    df_markout : DataFrame with columns [weight, pnl_{tau}s, timestamp].
    f          : array of 0/1 (same length as df_markout).
                 1 = filter (remove), 0 = keep.
    tau        : horizon in seconds (30, 120, or 300).
 
    Returns
    -------
    ScoreResult with all metrics.
    """
    pnl_col = f'pnl_{tau}s'

    if isinstance(f, pl.Series):
        f_np = f.to_numpy()
    else:
        f_np = np.asarray(f, dtype=np.float64)

    w   = df_markout['weight'].to_numpy().astype(np.float64)
    pnl = df_markout[pnl_col].to_numpy().astype(np.float64)

    # mask for valid check.
    valid = ~np.isnan(pnl)
    w       = w[valid]
    pnl     = pnl[valid]
    f_np    = f_np[valid]

    # filt/kept.
    filt = (f_np == 1)
    kept = (f_np == 0)

    # pnl_all.
    pnl_all = np.sum(w * pnl) / np.sum(w) if np.sum(w) > 0 else 0.0

    # pnl_kept.
    w_kept = w[kept]
    pnl_kept_val = (
        np.sum(w_kept * pnl[kept]) / np.sum(w_kept)
        if np.sum(w_kept) > 0 else 0.0
    )

    # pnL_filtered.
    w_filt = w[filt]
    pnl_filt_val = (
        np.sum(w_filt * pnl[filt]) / np.sum(w_filt)
        if np.sum(w_filt) > 0 else 0.0
    )

    # Score.
    score = pnl_kept_val - pnl_all

    # Turnover constraint.
    ts = df_markout["timestamp"].to_numpy()[valid]
    n_days = (ts.max() - ts.min()) / US_PER_DAY if len(ts) > 1 else 1.0
    # at least 1 day to avoid division by zero.
    n_days = max(n_days, 1.0)
    kept_turnover_total = float(np.sum(w_kept))
    kept_turnover_per_day = kept_turnover_total / n_days

    return ScoreResult(
        tau                     = tau,
        pnl_all                 = round(pnl_all, 6),
        pnl_kept                = round(pnl_kept_val, 6),
        pnl_filtered            = round(pnl_filt_val, 6),
        score                   = round(score, 6),
        kept_turnover_total     = round(kept_turnover_total, 2),
        kept_turnover_per_day   = round(kept_turnover_per_day, 2),
        n_days                  = round(n_days, 2),
        n_total                 = int(valid.sum()),
        n_kept                  = int(kept.sum()),
        n_filtered              = int(filt.sum()),
        filter_rate             = round(filt.sum() / valid.sum(), 4) if valid.sum() > 0 else 0.0,
        turnover_ok             = kept_turnover_per_day >= TURNOVER_CONSTRAINT
    )
        
def compute_score_all_horizons(
    df_markout: pl.DataFrame,
    f: np.ndarray | pl.Series,
    horizons_s: list[int] | None = None
) -> list[ScoreResult]:
    """Compute Score for all horizons at once."""

    if horizons_s is None:
        horizons_s = HORIZONS_S

    return [compute_score(df_markout, f, tau) for tau in horizons_s]
 
def score_summary_df(
    results: list[ScoreResult]
) -> pl.DataFrame:
    """Convert list of ScoreResult into a readable Polars DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "horizon":              f"{r.tau}s",
            "PnL_all (bps)":       r.pnl_all,
            "PnL_kept (bps)":      r.pnl_kept,
            "PnL_filtered (bps)":  r.pnl_filtered,
            "Score (bps)":         r.score,
            "filter_rate (%)":     round(r.filter_rate * 100, 2),
            "kept_usd/day":        r.kept_turnover_per_day,
            "turnover_ok":         r.turnover_ok
        })
    return pl.DataFrame(rows)


# Per-day score (for statistical tests).
def compute_daily_scores(
    df_markout: pl.DataFrame,
    f: np.ndarray | pl.Series,
    tau: int
) -> pl.DataFrame:
    """
    Compute Score(t) per calendar day.
 
    Needed for DM tests and block bootstrap - daily aggregation
    eliminates overlapping-markout autocorrelation (day >> τ).
 
    Returns DataFrame with columns: [day_ts, pnl_all, pnl_kept, score, kept_turnover].
    """

    pnl_col = f"pnl_{tau}s"
 
    if isinstance(f, pl.Series):
        f_arr = f
    else:
        f_arr = pl.Series("f", f, dtype=pl.Int8)

    df = df_markout.with_columns(
        f_arr.alias("_f"),
        (pl.col("timestamp") // US_PER_DAY).alias("_day")
    ).filter(pl.col(pnl_col).is_not_null())

    # per-day aggregation.
    df_daily = (
        df.group_by("_day")
        .agg(
            # PnL_all components.
            (pl.col("weight") * pl.col(pnl_col)).sum().alias("w_pnl_sum"),
            pl.col("weight").sum().alias("w_sum"),
 
            # PnL_kept components.
            ((1 - pl.col("_f")).cast(pl.Float64) * pl.col("weight") * pl.col(pnl_col)).sum().alias("w_pnl_kept_sum"),
            ((1 - pl.col("_f")).cast(pl.Float64) * pl.col("weight")).sum().alias("w_kept_sum")
        )
        .sort("_day")
        .with_columns(
            (pl.col("w_pnl_sum") / pl.col("w_sum")).alias("pnl_all"),
            (pl.col("w_pnl_kept_sum") / pl.col("w_kept_sum")).alias("pnl_kept")
        )
        .with_columns(
            (pl.col("pnl_kept") - pl.col("pnl_all")).alias("score"),
            pl.col("w_kept_sum").alias("kept_turnover")
        )
        .select("_day", "pnl_all", "pnl_kept", "score", "kept_turnover")
        .rename({"_day": "day_ts"})
    )
 
    return df_daily


# Bybit timestamp shift helper.
def shift_bybit_timestamps(lf_bybit: pl.LazyFrame) -> pl.LazyFrame:
    """
    Shift Bybit liquidation timestamps forward by +200 ms
    to account for cross-exchange network latency.
    """
    return lf_bybit.with_columns(
        (pl.col("timestamp") + BYBIT_DELAY_US).alias("timestamp")
    )


# Train / validation split helpers.
def date_to_us(
    year: int, 
    month: int, 
    day: int
) -> int:
    """Convert UTC date to microseconds since epoch."""
    
    dt = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * US_PER_SEC)
 
# pre-computed split boundaries.
TRAIN_START_US = date_to_us(2025, 12, 1)
TRAIN_END_US   = date_to_us(2026, 2, 1)     # exclusive: val starts here.
VAL_END_US     = date_to_us(2026, 3, 1)     # exclusive.
 
def split_train_val(
    df: pl.DataFrame,
    train_start: int    = TRAIN_START_US,
    val_start: int      = TRAIN_END_US,
    val_end: int        = VAL_END_US
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chronological train / validation split."""

    df_train = df.filter(
        (pl.col("timestamp") >= train_start) & (pl.col("timestamp") < val_start)
    )
    df_val = df.filter(
        (pl.col("timestamp") >= val_start) & (pl.col("timestamp") < val_end)
    )
    return df_train, df_val