# imports.
import numpy as np
import polars as pl
from pathlib import Path


# Constants.
US_PER_SEC      = 1_000_000
N_MIN_OBS       = 3              # minimum trades in rolling window for VPIN/RV.


# 1. Raw readers (one chunk).
def read_trades_for_chunk(
    path: Path,
    chunk_start_us: int,
    chunk_end_us:   int,
    max_window_us:  int,
    clipped_notional: float
) -> pl.DataFrame:
    """
    Read trades for [chunk_start - max_window, chunk_end).

    The left overlap (max_window_us) is needed so rolling features at the start
    of the chunk see enough history. The overlap region is dropped later.

    Adds columns: sign (Int8), notional (Float32), weight (Float32, clipped), hour (Int8, UTC).

    `hour` is computed via from_epoch — does not assume any particular epoch alignment.
    """
    read_start = chunk_start_us - max_window_us
    read_end   = chunk_end_us

    lf = (
        pl.scan_parquet(str(path))
        .filter(
            (pl.col("timestamp") >= read_start)
            & (pl.col("timestamp") < read_end)
        )
        .select([
            pl.col("timestamp").cast(pl.Int64),
            pl.col("price").cast(pl.Float64),
            pl.col("amount").cast(pl.Float64),
            pl.col("side")
        ])
        .with_columns([
            pl.when(pl.col("side") == "buy")
              .then(pl.lit(1))
              .otherwise(pl.lit(-1))
              .cast(pl.Int8)
              .alias("sign"),
            (pl.col("price") * pl.col("amount")).cast(pl.Float32).alias("notional"),
            (pl.col("price") * pl.col("amount"))
                .clip(upper_bound=clipped_notional)
                .cast(pl.Float32)
                .alias("weight"),
            # explicit UTC hour from microsecond epoch.
            pl.from_epoch(pl.col("timestamp"), time_unit="us")
                .dt.hour()
                .cast(pl.Int8)
                .alias("hour")
        ])
        .select(["timestamp", "price", "sign", "notional", "weight", "hour"])
        .sort("timestamp")
    )

    return _collect_streaming(lf)

def read_bbo_for_chunk(
    path: Path,
    chunk_start_us: int,
    chunk_end_us:   int,
    bbo_lookback_us: int,
    max_horizon_us:  int
) -> pl.DataFrame:
    """
    Read BBO for [chunk_start - bbo_lookback, chunk_end + max_horizon).

    - Left lookback ensures the trade-time ASOF backward at chunk start finds a quote.
    - Right lookahead ensures the markout ASOF at chunk_end + max_horizon finds a quote.

    Returns: timestamp, mid_price (Float64), spread (Float32), obi (Float32).
    """
    read_start = chunk_start_us - bbo_lookback_us
    read_end   = chunk_end_us   + max_horizon_us

    lf = (
        pl.scan_parquet(str(path))
        .filter(
            (pl.col("timestamp") >= read_start)
            & (pl.col("timestamp") < read_end)
        )
        .select([
            pl.col("timestamp").cast(pl.Int64),
            pl.col("bid_price").cast(pl.Float64),
            pl.col("ask_price").cast(pl.Float64),
            pl.col("bid_amount").cast(pl.Float64),
            pl.col("ask_amount").cast(pl.Float64)
        ])
        .with_columns([
            ((pl.col("bid_price") + pl.col("ask_price")) / 2.0)
                .cast(pl.Float64)
                .alias("mid_price"),
            (pl.col("ask_price") - pl.col("bid_price"))
                .cast(pl.Float32)
                .alias("spread"),
            (pl.col("bid_amount") / (pl.col("bid_amount") + pl.col("ask_amount")))
                .cast(pl.Float32)
                .alias("obi"),
        ])
        .select(["timestamp", "mid_price", "spread", "obi"])
        .sort("timestamp")
    )

    return _collect_streaming(lf)

def read_liquidations_for_chunk(
    path: Path,
    chunk_start_us: int,
    chunk_end_us:   int,
    max_window_us:  int,
    timestamp_shift_us: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read signed liquidation amounts for [chunk_start - max_window, chunk_end),
    after applying timestamp_shift_us (used for Bybit's +200ms cross-exchange lag).

    Returns:
        liq_ts      : sorted int64 timestamps after shift.
        cum_prefix  : float64 cumsum prefix of length len(liq_ts) + 1, cum_prefix[0] = 0.

    Empty data → (empty array, [0.0]).
    """
    shifted_read_start = chunk_start_us - max_window_us
    shifted_read_end   = chunk_end_us

    # Adjust read window to RAW timestamps (before shift).
    raw_read_start = shifted_read_start - timestamp_shift_us
    raw_read_end   = shifted_read_end   - timestamp_shift_us

    lf = (
        pl.scan_parquet(str(path))
        .filter(
            (pl.col("timestamp") >= raw_read_start)
            & (pl.col("timestamp") < raw_read_end)
        )
        .select([
            (pl.col("timestamp").cast(pl.Int64) + timestamp_shift_us).alias("timestamp"),
            pl.when(pl.col("side") == "buy")
              .then(pl.col("amount"))
              .otherwise(-pl.col("amount"))
              .cast(pl.Float64)
              .alias("signed_amount")
        ])
        .sort("timestamp")
    )

    df = _collect_streaming(lf)

    if df.height == 0:
        return np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.float64)

    liq_ts = df["timestamp"].to_numpy().astype(np.int64, copy=False)
    amt    = df["signed_amount"].to_numpy().astype(np.float64, copy=False)

    cum_prefix = np.empty(len(amt) + 1, dtype=np.float64)
    cum_prefix[0] = 0.0
    np.cumsum(amt, out=cum_prefix[1:])

    return liq_ts, cum_prefix


# 2. ASOF BBO + markout PnL.
def add_bbo_and_markout(
    trades: pl.DataFrame,
    bbo:    pl.DataFrame,
    horizons_s:  list[int],
    rebate_bps:  float,
    max_bbo_ts:  int,
) -> pl.DataFrame:
    """
    ASOF join BBO at trade-time (mid, spread, obi, dist_to_mid)
    and at t + tau for each markout horizon (pnl_{tau}s).

    Trades whose (t + tau) > max_bbo_ts get pnl_{tau}s = NaN.

    If a trade has no BBO before it (extremely rare — only when bbo_lookback
    didn't reach back far enough), trade-time BBO columns will be NaN.
    """
    if trades.height == 0:
        return trades

    # current BBO at trade-time (backward asof).
    df = trades.join_asof(
        bbo,
        on="timestamp",
        strategy="backward"
    )

    df = df.with_columns([
        ((pl.col("price") - pl.col("mid_price")) / pl.col("mid_price") * 10000.0)
            .cast(pl.Float32)
            .alias("dist_to_mid")
    ])

    # markouts at t + tau.
    for tau in horizons_s:
        ts_col     = f"_lookup_ts_{tau}"
        bbo_ts_col = f"_bbo_ts_{tau}"
        mid_col    = f"_mid_{tau}"
        pnl_col    = f"pnl_{tau}s"
        tau_us     = tau * US_PER_SEC

        right = bbo.select([
            pl.col("timestamp").alias(bbo_ts_col),
            pl.col("mid_price").alias(mid_col)
        ])

        df = (
            df
            .with_columns((pl.col("timestamp") + tau_us).alias(ts_col))
            .join_asof(
                right,
                left_on=ts_col,
                right_on=bbo_ts_col,
                strategy="backward"
            )
            .with_columns([
                # pnl = NaN if lookup beyond BBO range.
                pl.when(pl.col(ts_col) <= max_bbo_ts)
                  .then(
                      -1.0 * pl.col("sign")
                      * (pl.col(mid_col) - pl.col("price"))
                      / pl.col("price") * 10000.0
                      + rebate_bps
                  )
                  .otherwise(float("nan"))
                  .cast(pl.Float32)
                  .alias(pnl_col)
            ])
            .drop([ts_col, bbo_ts_col, mid_col])
        )

    # cast mid_price to Float32 for output (Float64 was needed only for join precision).
    df = df.with_columns(pl.col("mid_price").cast(pl.Float32))

    return df



# 3. Rolling features (VPIN, realized volatility).
def _rolling_count_and_sum(
    ts:        np.ndarray,
    values:    np.ndarray,
    window_us: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (count, sum) over [t - window_us, t) for each t in ts.
    Excludes the row at t itself (strict half-open interval).
    """
    left  = np.searchsorted(ts, ts - window_us, side="left")
    right = np.searchsorted(ts, ts,             side="left")

    count = right - left

    prefix = np.empty(len(values) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values.astype(np.float64, copy=False), out=prefix[1:])

    s = prefix[right] - prefix[left]
    return count, s

def _rolling_sum_sq(
    ts:        np.ndarray,
    values:    np.ndarray,
    window_us: int
) -> np.ndarray:
    """Returns sum of squares over [t - window_us, t)."""
    left  = np.searchsorted(ts, ts - window_us, side="left")
    right = np.searchsorted(ts, ts,             side="left")

    v64 = values.astype(np.float64, copy=False)
    prefix2 = np.empty(len(v64) + 1, dtype=np.float64)
    prefix2[0] = 0.0
    np.cumsum(v64 * v64, out=prefix2[1:])

    return prefix2[right] - prefix2[left]

def add_rolling_features(
    df:        pl.DataFrame,
    windows_s: list[int],
    n_min_obs: int = N_MIN_OBS
) -> pl.DataFrame:
    """
    Add VPIN and realized-volatility features for each window in windows_s.

    VPIN_W       = |sum(sign * notional)| / sum(notional)  over [t - W, t)
    RV_W         = stddev_sample(log(price))               over [t - W, t)

    Both are set to NaN when the window contains fewer than `n_min_obs` trades.
    """
    if df.height == 0:
        return df

    df = df.sort("timestamp").rechunk()

    ts       = df["timestamp"].to_numpy().astype(np.int64,   copy=False)
    sign     = df["sign"    ].to_numpy().astype(np.float64,  copy=False)
    notional = df["notional"].to_numpy().astype(np.float64,  copy=False)
    price    = df["price"   ].to_numpy().astype(np.float64,  copy=False)

    signed_notional = sign * notional
    log_price       = np.log(price)

    new_cols = []
    for W in windows_s:
        window_us = W * US_PER_SEC

        # VPIN: |sum_signed| / sum_abs, NaN if too few obs or sum_abs == 0.
        count_sn, sum_signed = _rolling_count_and_sum(ts, signed_notional, window_us)
        _,        sum_abs    = _rolling_count_and_sum(ts, notional,        window_us)

        vpin = np.full(len(ts), np.nan, dtype=np.float32)
        ok = (count_sn >= n_min_obs) & (sum_abs > 0.0)
        vpin[ok] = (np.abs(sum_signed[ok]) / sum_abs[ok]).astype(np.float32)

        # RV: sample stddev of log_price, NaN if count < n_min_obs.
        count_rv, sum_lp = _rolling_count_and_sum(ts, log_price, window_us)
        sum_lp2          = _rolling_sum_sq(ts, log_price, window_us)

        rv = np.full(len(ts), np.nan, dtype=np.float32)
        ok_rv = count_rv >= n_min_obs
        c = count_rv[ok_rv].astype(np.float64)
        var = (sum_lp2[ok_rv] - sum_lp[ok_rv] ** 2 / c) / (c - 1.0)
        var = np.maximum(var, 0.0)        # numerical floor.
        rv[ok_rv] = np.sqrt(var).astype(np.float32)

        new_cols.append(pl.Series(f"vpin_{W}s", vpin))
        new_cols.append(pl.Series(f"rv_{W}s",   rv))

    return df.with_columns(new_cols)



# 4. Liquidation flow features.
def _liq_window_sum(
    trade_ts:       np.ndarray,
    liq_ts:         np.ndarray,
    liq_cum_prefix: np.ndarray,
    window_us:      int
) -> np.ndarray:
    """
    Sum of signed liquidations over [t - window_us, t) for each trade.

    Uses cumulative-sum prefix table:
        sum_in_window(t) = cum_prefix[hi] - cum_prefix[lo]
        where lo = first liq with ts >= t - W,
              hi = first liq with ts >= t       (i.e. exclusive of t).
    Returns Float32.
    """
    if len(liq_ts) == 0:
        return np.zeros(len(trade_ts), dtype=np.float32)

    lo = np.searchsorted(liq_ts, trade_ts - window_us, side="left")
    hi = np.searchsorted(liq_ts, trade_ts,             side="left")

    out = liq_cum_prefix[hi] - liq_cum_prefix[lo]
    return out.astype(np.float32)

def add_liquidation_features(
    df:        pl.DataFrame,
    bin_ts:    np.ndarray,
    bin_cum:   np.ndarray,
    byb_ts:    np.ndarray,
    byb_cum:   np.ndarray,
    windows_s: list[int]
) -> pl.DataFrame:
    """
    For each window W in windows_s, add liq_binance_Ws and liq_bybit_Ws.
    Zero values are valid (no liquidations) and stay as 0.0 — NOT NaN.
    """
    if df.height == 0:
        return df

    ts = df["timestamp"].to_numpy().astype(np.int64, copy=False)

    new_cols = []
    for W in windows_s:
        window_us = W * US_PER_SEC
        new_cols.append(pl.Series(
            f"liq_binance_{W}s",
            _liq_window_sum(ts, bin_ts, bin_cum, window_us)
        ))
        new_cols.append(pl.Series(
            f"liq_bybit_{W}s",
            _liq_window_sum(ts, byb_ts, byb_cum, window_us)
        ))

    return df.with_columns(new_cols)


# 5. End-to-end chunk pipeline.
def compute_chunk(
    chunk_start_us:   int,
    chunk_end_us:     int,
    raw_trades_path:  Path,
    raw_bbo_path:     Path,
    raw_liq_bin_path: Path,
    raw_liq_byb_path: Path,
    horizons_s:       list[int],
    windows_s:        list[int],
    bbo_lookback_us:  int,
    bybit_delay_us:   int,
    clipped_notional: float,
    rebate_bps:       float,
    output_columns:   list[str],
    n_min_obs:        int = N_MIN_OBS
) -> pl.DataFrame:
    """
    Compute one enriched chunk and return only rows from [chunk_start_us, chunk_end_us).
    """
    max_horizon_us = max(horizons_s) * US_PER_SEC
    max_window_us  = max(windows_s)  * US_PER_SEC

    # read raw inputs
    trades = read_trades_for_chunk(
        raw_trades_path,
        chunk_start_us, chunk_end_us,
        max_window_us=max_window_us,
        clipped_notional=clipped_notional
    )
    if trades.height == 0:
        empty_schema = {c: pl.Float32 for c in output_columns}
        return pl.DataFrame(schema=empty_schema)

    bbo = read_bbo_for_chunk(
        raw_bbo_path,
        chunk_start_us, chunk_end_us,
        bbo_lookback_us=bbo_lookback_us,
        max_horizon_us=max_horizon_us
    )

    if bbo.height == 0:
        # cannot compute markout / mid features — chunk is unusable.
        empty_schema = {c: pl.Float32 for c in output_columns}
        return pl.DataFrame(schema=empty_schema)

    max_bbo_ts = int(bbo["timestamp"].max())

    bin_ts, bin_cum = read_liquidations_for_chunk(
        raw_liq_bin_path,
        chunk_start_us, chunk_end_us,
        max_window_us=max_window_us,
        timestamp_shift_us=0
    )
    byb_ts, byb_cum = read_liquidations_for_chunk(
        raw_liq_byb_path,
        chunk_start_us, chunk_end_us,
        max_window_us=max_window_us,
        timestamp_shift_us=bybit_delay_us
    )

    # pipeline
    df = add_bbo_and_markout(
        trades, bbo,
        horizons_s=horizons_s,
        rebate_bps=rebate_bps,
        max_bbo_ts=max_bbo_ts,
    )
    df = add_rolling_features(df, windows_s=windows_s, n_min_obs=n_min_obs)
    df = add_liquidation_features(df, bin_ts, bin_cum, byb_ts, byb_cum, windows_s=windows_s)

    # drop the left overlap region and order columns.
    df = (
        df
        .filter(
            (pl.col("timestamp") >= chunk_start_us)
            & (pl.col("timestamp") < chunk_end_us)
        )
        .select(output_columns)
        .sort("timestamp")
    )

    return df


# 6. Subsampling.
def make_subsample_for_chunk(
    df:                pl.DataFrame,
    global_row_offset: int,
    subsample_rate:    int
) -> pl.DataFrame:
    """
    Chronological deterministic subsample: keep rows where
    (global_row_offset + local_index) % subsample_rate == 0.

    This is stable across chunk boundaries because indexing is global.
    """
    if df.height == 0:
        return df

    idx = global_row_offset + np.arange(df.height)
    mask = (idx % subsample_rate) == 0
    return df.filter(pl.Series(mask))


# 7. Utility helpers.
def _collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect with streaming engine, fallback for older Polars."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)

def count_rows_parquet(path: Path) -> int:
    """Count rows in a parquet file without loading data."""
    return int(
        _collect_streaming(
            pl.scan_parquet(str(path)).select(pl.len().alias("n"))
        )[0, "n"]
    )

def write_parquet(df: pl.DataFrame, path: Path):
    """Write parquet with zstd compression and statistics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path), compression="zstd", statistics=True)

def safe_unlink(path: Path):
    """Delete file if it exists."""
    if path.exists():
        path.unlink()

def build_chunk_plan(
    raw_trades_path: Path,
    chunk_days:      int,
    align_to_utc:    bool
) -> tuple[list[tuple[int, int]], int, int]:
    """
    Read timestamp range from raw trades and split into [start, end) chunks.

    Returns: (chunks, raw_ts_min, raw_ts_max).
    """
    day_us = 86_400 * US_PER_SEC

    range_df = _collect_streaming(
        pl.scan_parquet(str(raw_trades_path)).select([
            pl.col("timestamp").min().alias("ts_min"),
            pl.col("timestamp").max().alias("ts_max")
        ])
    )
    raw_ts_min = int(range_df[0, "ts_min"])
    raw_ts_max = int(range_df[0, "ts_max"])

    if align_to_utc:
        first_chunk_start = (raw_ts_min // day_us) * day_us
    else:
        first_chunk_start = raw_ts_min

    chunk_width_us = chunk_days * day_us
    last_exclusive = raw_ts_max + 1

    chunks = []
    cs = first_chunk_start
    while cs < last_exclusive:
        ce = min(cs + chunk_width_us, last_exclusive)
        chunks.append((cs, ce))
        cs += chunk_width_us

    return chunks, raw_ts_min, raw_ts_max
