# import libraries.
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# ======================================
# Features.
# ======================================

# spread feature.
def add_spread_feature(
    lf_lob: pl.LazyFrame,
    lf_markout: pl.LazyFrame
) -> pl.DataFrame:
    
    """
    Function:
        - Calculate (ask - bid) / (ask + bid) feature based on LOB.
        - Prepare markout Dataframe for PNL calculation.

    Return:
        - pl.Dataframe: markout with spread feature column.
    """

    df_lob = (
        lf_lob
        .select(
            pl.col("timestamp"),
            ((pl.col("ask_price") - pl.col("bid_price")) / (pl.col("ask_price") + pl.col("bid_price"))).cast(pl.Float32).alias("half_spread")
        )
        .collect(engine="streaming")
    )

    df = lf_markout.collect(engine="streaming").join_asof(
        df_lob,
        on          = "timestamp",
        strategy    = "backward"
    )

    del df_lob

    return df


# order book imbalance(OBI) feature.
def add_obi_feature(
    lf_lob: pl.LazyFrame,
    lf_markout : pl.LazyFrame
) -> pl.DataFrame:
    
    """
    Function:
        - Calculate (bid_amount) / (ask_amount + bid_amount) feature based on LOB.
        - Prepared markout Dataframe for PNL calculation.

    Return:
        - pl.Dataframe: markout with OBI feature column.
    """

    df_lob = (
        lf_lob
        .select(
            pl.col("timestamp"),
            (pl.col("bid_amount") / (pl.col("ask_amount") + pl.col("bid_amount"))).cast(pl.Float32).alias("obi")
        )
        .collect(engine="streaming")
    )

    df = lf_markout.collect(engine="streaming").join_asof(
        df_lob,
        on          = "timestamp",
        strategy    = "backward"
    )

    del df_lob

    return df


# trade flow.
def build_trade_flow_bars(
    lf_trades: pl.LazyFrame,
    bar_size_s: int = 1
) -> pl.DataFrame:
    
    """
    Function: 
        - Agregate trades into bars (default size is 1s).
        - Calculate `signed_trade_money_flow` = sum(sign * amount * price).
        - Calculate `total_trade_money_flow` = sum(amount * price).
        - Calculate `normalized_trade_money_flow` = `signed_trade_money_flow` / `total_trade_money_flow`.
 
    Return:
        - pl.Dataframe: trades bars with trade flow based features columns.
    """

    # from second to microsecond.
    bar_size_us = bar_size_s * 1_000_000 

    df = (
        lf_trades
        .with_columns(
            (pl.col("timestamp") // bar_size_us * bar_size_us).alias("bar_ts"),
            (
                pl.when(pl.col("side") == "buy")
                .then(pl.col("price") * pl.col("amount"))
                .otherwise(-pl.col("price") * pl.col("amount"))
            ).cast(pl.Float32).alias("signed_dollar_volume")
        )
        .group_by("bar_ts")
        .agg(
            pl.col("signed_dollar_volume").sum().alias("signed_trade_money_flow"),
            pl.col("signed_dollar_volume").abs().sum().alias("total_trade_money_flow"),
            (pl.col("signed_dollar_volume").sum() / pl.col("signed_dollar_volume").abs().sum()).alias("normalized_trade_money_flow")
        )
        .sort("bar_ts")
        .collect(engine="streaming")
    )

    return df

def add_trade_flow_feature(
    lf_trades: pl.LazyFrame,
    lf_markout: pl.LazyFrame,
    windows_sec: list[int] = [1, 5, 20],
    name_feature: str = "normalized_trade_money_flow"
) -> pl.DataFrame:
    
    """
    Function:
        - calculate flow based feature.
        - you can choose: "signed_trade_money_flow", "total_trade_money_flow", "normalized_trade_money_flow" as `name_feature`.

    Return:
        - pl.Dataframe: markout with flow based feature column.
    """

    if (name_feature not in ["signed_trade_money_flow", "total_trade_money_flow", "normalized_trade_money_flow"]):
        raise ValueError("You can choose only: signed_trade_money_flow, total_trade_money_flow, normalized_trade_money_flow as name_feature")

    df = lf_markout.collect(engine="streaming")

    for w in windows_sec:
        df_bars = build_trade_flow_bars(lf_trades, bar_size_s=w)

        df_bars = df_bars.select(
            pl.col("bar_ts").alias("timestamp"),
            pl.col(name_feature).alias(f"{name_feature}_{w}s"),
        )

        df = df.join_asof(df_bars, on="timestamp", strategy="backward")
        del df_bars

    return df


# trade flow.
def build_liquidations_flow_bars(
    lf_liq: pl.LazyFrame,
    bar_size_s: int = 1
) -> pl.DataFrame:
    
    """
    Function: 
        - Agregate liquidations into bars (default size is 1s).
        - Calculate `signed_liquidations_money_flow` = sum(sign * amount * price).
        - Calculate `total_liquidations_money_flow` = sum(amount * price).
        - Calculate `normalized_liquidations_money_flow` = `signed_liquidations_money_flow` / `total_liquidations_money_flow`.
 
    Return:
        - pl.Dataframe: liquidations bars with liquidation flow based features columns.
    """

    # from second to microsecond.
    bar_size_us = bar_size_s * 1_000_000 

    df = (
        lf_liq
        .with_columns(
            (pl.col("timestamp") // bar_size_us * bar_size_us).alias("bar_ts"),
            (
                pl.when(pl.col("side") == "buy")
                .then(pl.col("price") * pl.col("amount"))
                .otherwise(-pl.col("price") * pl.col("amount"))
            ).cast(pl.Float32).alias("signed_dollar_volume")
        )
        .group_by("bar_ts")
        .agg(
            pl.col("signed_dollar_volume").sum().alias("signed_liquidations_money_flow"),
            pl.col("signed_dollar_volume").abs().sum().alias("total_liquidations_money_flow"),
            (pl.col("signed_dollar_volume").sum() / pl.col("signed_dollar_volume").abs().sum()).alias("normalized_liquidations_money_flow")
        )
        .sort("bar_ts")
        .collect(engine="streaming")
    )

    return df

def add_trade_liquidations_feature(
    lf_liq_bybit: pl.LazyFrame,
    lf_liq_binance: pl.LazyFrame,
    lf_markout: pl.LazyFrame,
    windows_sec: list[int] = [1, 5, 20],
    name_feature: str = "normalized_liquidations_money_flow"
) -> pl.DataFrame:
    
    """
    Function:
        - calculate flow based feature.
        - you can choose: "signed_liquidations_money_flow", "total_liquidations_money_flow", "normalized_liquidations_money_flow" as `name_feature`.

    Return:
        - pl.Dataframe: markout with flow based feature column.
    """

    if (name_feature not in ["signed_liquidations_money_flow", "total_liquidations_money_flow", "normalized_liquidations_money_flow"]):
        raise ValueError("You can choose only: signed_liquidations_money_flow, total_liquidations_money_flow, normalized_liquidations_money_flow as name_feature")

    df = lf_markout.collect(engine="streaming")

    # concatenate liquidations + sort by timestamps.
    lf_liq = pl.concat([
        lf_liq_binance,
        lf_liq_bybit.with_columns(
            (pl.col("timestamp") + 200_000).alias("timestamp")
        ),
    ]).sort("timestamp")

    for w in windows_sec:
        df_bars = build_liquidations_flow_bars(lf_liq, bar_size_s=w)

        df_bars = df_bars.select(
            pl.col("bar_ts").alias("timestamp"),
            pl.col(name_feature).alias(f"{name_feature}_{w}s"),
        )

        df = df.join_asof(df_bars, on="timestamp", strategy="backward")
        del df_bars

    return df


# recent volatility.
def build_volatility_bars(
    lf_lob: pl.LazyFrame,
    bar_size_s: int = 1
) -> pl.DataFrame:
    """
    Function: 
        - Agregate lob into bars (default size is 1s).
        - Calculate `rv` = `mid`.std().

    Return:
        - pl.Dataframe: lob bars with `rv`(realized volatility), `mid_last` features columns.
    """

    bar_size_us = bar_size_s * 1_000_000

    df = (
        lf_lob
        .with_columns(
            (pl.col("timestamp") // bar_size_us * bar_size_us).alias("bar_ts"),
            ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid"),
        )
        .group_by("bar_ts")
        .agg(
            pl.col("mid").std().cast(pl.Float32).alias("rv"),
            pl.col("mid").last().cast(pl.Float32).alias("mid_last")
        )
        .sort("bar_ts")
        .collect(engine="streaming")
    )

    return df

def add_volatility_feature(
    lf_markout: pl.LazyFrame,
    lf_lob: pl.LazyFrame,
    windows_sec: list[int] = [5, 20, 60]
) -> pl.DataFrame:
    
    """
    Function:
        - calculate realized volatility (std of mid_price) over different time windows.
        - each window produces a column: rv_5s, rv_20s, rv_60s etc.

    Return:
        - pl.DataFrame: markout with volatility feature columns.
    """

    df = lf_markout.collect(engine="streaming")

    for w in windows_sec:
        df_bars = build_volatility_bars(lf_lob, bar_size_s=w)

        df_bars = df_bars.select(
            pl.col("bar_ts").alias("timestamp"),
            pl.col("rv").alias(f"rv_{w}s")
        )

        df = df.join_asof(df_bars, on="timestamp", strategy="backward")
        del df_bars

    return df


# Order Flow Imbalance.
def build_ofi_bars(
    lf_lob: pl.LazyFrame,
    bar_size_s: int = 1
) -> pl.DataFrame:
    """
    Function:
        - Calculate per-tick OFI: delta bid_amount - delta ask_amount on best level.
        - Aggregate to bars of given size.

    Return:
        - pl.DataFrame: bars with `ofi` column.
    """

    bar_size_us = bar_size_s * 1_000_000

    df = (
        lf_lob
        .sort("timestamp")
        .with_columns(
            # delta bid.
            pl.when(pl.col("bid_price") > pl.col("bid_price").shift(1))
              .then(pl.col("bid_amount"))
              .when(pl.col("bid_price") < pl.col("bid_price").shift(1))
              .then(-pl.col("bid_amount").shift(1))
              .otherwise(pl.col("bid_amount") - pl.col("bid_amount").shift(1))
              .alias("d_bid"),

            # delta ask.
            pl.when(pl.col("ask_price") < pl.col("ask_price").shift(1))
              .then(pl.col("ask_amount"))
              .when(pl.col("ask_price") > pl.col("ask_price").shift(1))
              .then(-pl.col("ask_amount").shift(1))
              .otherwise(pl.col("ask_amount") - pl.col("ask_amount").shift(1))
              .alias("d_ask"),
        )
        .with_columns(
            (pl.col("d_bid") - pl.col("d_ask")).cast(pl.Float32).alias("ofi_tick"),
            (pl.col("timestamp") // bar_size_us * bar_size_us).alias("bar_ts"),
        )
        .group_by("bar_ts")
        .agg(pl.col("ofi_tick").sum().alias("ofi"))
        .sort("bar_ts")
        .collect(engine="streaming")
    )

    return df

def add_ofi_feature(
    lf_markout: pl.LazyFrame,
    lf_lob: pl.LazyFrame,
    windows_sec: list[int] = [5, 15, 60],
) -> pl.DataFrame:
    """
    Function:
        - Calculate OFI over different time windows.
        - Each window produces a column: ofi_5s, ofi_15s, ofi_60s.

    Return:
        - pl.DataFrame: markout with OFI feature columns.
    """

    df = lf_markout.collect(engine="streaming")

    for w in windows_sec:
        df_bars = build_ofi_bars(lf_lob, bar_size_s=w)

        df_bars = df_bars.select(
            pl.col("bar_ts").alias("timestamp"),
            pl.col("ofi").alias(f"ofi_{w}s"),
        )

        df = df.join_asof(df_bars, on="timestamp", strategy="backward")
        del df_bars

    return df


# VPIN.
def build_vpin_buckets(
    lf_trades: pl.LazyFrame,
    bucket_volume_usd: float = 500_000,
    bar_size_s: int = 1
) -> pl.DataFrame:
    """
    Function:
        - Aggregate trades into volume buckets of given dollar size.
        - For each bucket compute imbalance = |signed| / total.

    Return:
        - pl.DataFrame: volume buckets with `imbalance` column.
    """

    bar_size_us = bar_size_s * 1_000_000

    # First calculate minute bars for eficient.
    df_time_bars = (
        lf_trades
        .with_columns(
            (pl.col("timestamp") // bar_size_us * bar_size_us).alias("bar_ts"),
            (
                pl.when(pl.col("side") == "buy")
                .then(pl.col("price") * pl.col("amount"))
                .otherwise(-pl.col("price") * pl.col("amount"))
            ).cast(pl.Float32).alias("signed_dollar_volume")
        )
        .group_by("bar_ts")
        .agg(
            pl.col("signed_dollar_volume").sum().alias("signed_vol"),
            pl.col("signed_dollar_volume").abs().sum().alias("total_vol")
        )
        .sort("bar_ts")
        .collect(engine="streaming")
    )

    # Make buckets based on cummulitive volume.
    df_buckets = (
        df_time_bars
        .with_columns(
            (pl.col("total_vol").cum_sum() // bucket_volume_usd).alias("bucket_id")
        )
        .group_by("bucket_id")
        .agg(
            pl.col("bar_ts").last().alias("bucket_ts"),
            pl.col("signed_vol").sum().alias("b_signed"),
            pl.col("total_vol").sum().alias("b_total")
        )
        .sort("bucket_id")
        .with_columns(
            (pl.col("b_signed").abs() / pl.col("b_total")).cast(pl.Float32).alias("imbalance")
        )
        .select("bucket_ts", "imbalance")
    )

    return df_buckets

def add_vpin_feature(
    lf_markout: pl.LazyFrame,
    lf_trades: pl.LazyFrame,
    bucket_volume_usd: float = 500_000,
    smoothing_windows: list[int] = [10, 50],
) -> pl.DataFrame:
    """
    Function:
        - Calculate VPIN as rolling mean of bucket imbalances.
        - Each smoothing window produces a column: vpin_10, vpin_50.

    Return:
        - pl.DataFrame: markout with VPIN feature columns.
    """

    df = lf_markout.collect(engine="streaming")

    df_buckets = build_vpin_buckets(
        lf_trades,
        bucket_volume_usd=bucket_volume_usd,
    )

    for n in smoothing_windows:
        df_vpin = (
            df_buckets
            .with_columns(
                pl.col("imbalance")
                .rolling_mean(window_size=n, min_samples=1)
                .cast(pl.Float32)
                .alias(f"vpin_{n}")
            )
            .select(
                pl.col("bucket_ts").alias("timestamp"),
                pl.col(f"vpin_{n}")
            )
            .sort("timestamp")
        )

        df = df.join_asof(df_vpin, on="timestamp", strategy="backward")
        del df_vpin

    del df_buckets

    return df


# Amihud.
def build_amihud_bars(
    lf_lob: pl.LazyFrame,
    lf_trades: pl.LazyFrame,
    bar_size_s: int = 60
) -> pl.DataFrame:
    """
    Function:
        - Aggregate LOB and trades into bars.
        - Calculate Amihud = |bar_return| / bar_volume.

    Return:
        - pl.DataFrame: bars with `amihud` column.
    """

    bar_size_us = bar_size_s * 1_000_000

    # mid_price bars: first и last for return.
    df_mid_bars = (
        lf_lob
        .with_columns(
            (pl.col("timestamp") // bar_size_us * bar_size_us).alias("bar_ts"),
            ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid")
        )
        .group_by("bar_ts")
        .agg(
            pl.col("mid").first().alias("mid_first"),
            pl.col("mid").last().alias("mid_last")
        )
        .sort("bar_ts")
        .collect(engine="streaming")
    )

    # volume per bar.
    df_vol_bars = (
        lf_trades
        .with_columns(
            (pl.col("timestamp") // bar_size_us * bar_size_us).alias("bar_ts"),
            (pl.col("price") * pl.col("amount")).cast(pl.Float32).alias("notional")
        )
        .group_by("bar_ts")
        .agg(pl.col("notional").sum().alias("volume"))
        .sort("bar_ts")
        .collect(engine="streaming")
    )

    # join + amihud.
    df = (
        df_mid_bars
        .join(df_vol_bars, on="bar_ts", how="left")
        .fill_null(0)
        .with_columns(
            ((pl.col("mid_last") - pl.col("mid_first")).abs() / pl.col("mid_first"))
            .alias("abs_return"),
        )
        .with_columns(
            pl.when(pl.col("volume") > 0)
            .then(pl.col("abs_return") / pl.col("volume"))
            .otherwise(None)
            .cast(pl.Float32)
            .alias("amihud")
        )
        .select("bar_ts", "amihud")
        .sort("bar_ts")
    )

    return df

def add_amihud_feature(
    lf_markout: pl.LazyFrame,
    lf_lob: pl.LazyFrame,
    lf_trades: pl.LazyFrame,
    windows_sec: list[int] = [5, 20, 60]
) -> pl.DataFrame:
    """
    Function:
        - Calculate Amihud illiquidity over different time windows.
        - Each window produces a column: amihud_5s, amihud_20s, amihud_60s.

    Return:
        - pl.DataFrame: markout with Amihud feature columns.
    """

    df = lf_markout.collect(engine="streaming")

    for w in windows_sec:
        df_bars = build_amihud_bars(lf_lob, lf_trades, bar_size_s=w)

        df_bars = df_bars.select(
            pl.col("bar_ts").alias("timestamp"),
            pl.col("amihud").alias(f"amihud_{w}s"),
        )

        df = df.join_asof(df_bars, on="timestamp", strategy="backward")
        del df_bars

    return df


# microprice.
def add_microprice_deviation_feature(
    lf_lob: pl.LazyFrame,
    lf_markout: pl.LazyFrame
) -> pl.DataFrame:
    
    """
    microprice = ask * bid_vol/(bid_vol+ask_vol) + bid * ask_vol/(bid_vol+ask_vol)

    deviation = (microprice - mid) / half_spread

    Shows the direction of short-term price pressure.
    """

    df_lob = (
        lf_lob
        .select(
            pl.col("timestamp"),
            (((pl.col("ask_price") * pl.col("bid_amount") + pl.col("bid_price") * pl.col("ask_amount")) 
            / (pl.col("bid_amount") + pl.col("ask_amount")) 
            - ((pl.col("bid_price") + pl.col("ask_price")) / 2)) 
            / ((pl.col("ask_price") - pl.col("bid_price")) / 2))
            .cast(pl.Float32)
            .alias("microprice_dev")
        )
        .collect(engine="streaming")
    )

    df_trades = lf_markout.collect(engine="streaming")

    df = df_trades.join_asof(df_lob, on="timestamp", strategy="backward")

    return df


# distance to mid.
def add_distance_to_mid_feature(
    lf_lob: pl.LazyFrame,
    lf_markout: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Function:
        - Calculate distance from trade price to mid in bps.
        - Positive = trade above mid (taker buy), negative = below mid (taker sell).

    Return:
        - pl.DataFrame: markout with `dist_to_mid` feature column.
    """

    df_mid = (
        lf_lob
        .select(
            pl.col("timestamp"),
            ((pl.col("bid_price") + pl.col("ask_price")) / 2)
            .cast(pl.Float32)
            .alias("mid_at_trade"),
        )
        .collect(engine="streaming")
    )

    df = lf_markout.collect(engine="streaming")

    df = df.join_asof(df_mid, on="timestamp", strategy="backward")

    df = df.with_columns(
        ((pl.col("price") - pl.col("mid_at_trade")) / pl.col("mid_at_trade") * 10_000)
        .cast(pl.Float32)
        .alias("dist_to_mid")
    ).drop("mid_at_trade")

    del df_mid

    return df


# ======================================
# Utilities.
# ======================================

def add_quantile_buckets(
    df: pl.DataFrame,
    col: str,
    n_buckets: int = 5
) -> pl.DataFrame:
    
    """
    Splits col into n_buckets by quantiles.

    Adds the `{col}_bucket` column.
    """

    # quantiles.
    quantiles = [i / n_buckets for i in range(1, n_buckets)]
    boundaries = [df[col].quantile(q) for q in quantiles]

    bucket_expr = pl.lit(n_buckets, dtype=pl.UInt8)
    for i, b in reversed(list(enumerate(boundaries))):
        bucket_expr = (
            pl.when(pl.col(col) <= b)
            .then(pl.lit(i + 1, dtype=pl.UInt8))
            .otherwise(bucket_expr)
        )

    return df.with_columns(bucket_expr.alias(f"{col}_bucket"))


def evaluate_class(
    df: pl.DataFrame,
    class_col: str,
    horizons: list[int] = [30, 120, 300],
) -> pl.DataFrame:
    """
    For each bucket calculate weighted PnL, count, turnover share.
    """

    agg_exprs = [pl.col("weight").sum().alias("turnover")]

    for h in horizons:
        pnl_col = f"pnl_{h}s"
        agg_exprs.append(
            (pl.col("weight") * pl.col(pnl_col)).sum().alias(f"w_pnl_sum_{h}s")
        )

    df_agg = df.group_by(class_col).agg(agg_exprs).sort(class_col)

    # weighted mean pnl.
    for h in horizons:
        df_agg = df_agg.with_columns(
            (pl.col(f"w_pnl_sum_{h}s") / pl.col("turnover"))
            .round(4)
            .alias(f"w_pnl_{h}s")
        ).drop(f"w_pnl_sum_{h}s")

    # turnover share.
    total_turnover = df_agg["turnover"].sum()
    df_agg = df_agg.with_columns(
        (pl.col("turnover") / total_turnover * 100)
        .round(2)
        .alias("turnover_pct")
    )

    return df_agg


def plot_class(
    df_eval: pl.DataFrame,
    class_col: str,
    horizons: list[int] = [30, 120, 300],
):
    """
    Bar chart: weighted PnL by bucket for each horizon.
    """

    buckets = df_eval[class_col].to_list()
    x = range(len(buckets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, h in enumerate(horizons):
        vals = df_eval[f"w_pnl_{h}s"].to_list()
        ax.bar([xi + i * width for xi in x], vals, width, label=f"{h}s")

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([str(b) for b in buckets])
    ax.set_xlabel(class_col)
    ax.set_ylabel("Weighted PnL (bps)")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.legend()
    ax.set_title(f"Markout by {class_col}")

    plt.tight_layout()
    plt.show()


def evaluate_class_by_period(
    df: pl.DataFrame,
    class_col: str,
    train_end_us: int = 1769904000000000,
    horizons: list[int] = [30, 120, 300],
):
    df_train = df.filter(pl.col("timestamp") < train_end_us)
    df_val = df.filter(pl.col("timestamp") >= train_end_us)

    result_train = evaluate_class(df_train, class_col, horizons)
    result_val = evaluate_class(df_val, class_col, horizons)

    return result_train, result_val


def add_hour_feature(
    lf_markout: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Function:
        - Extract hour of day (UTC) from trade timestamp.

    Return:
        - pl.DataFrame: markout with `hour_utc` column.
    """

    df = (
        lf_markout
        .with_columns(
            ((pl.col("timestamp") // 3_600_000_000) % 24)
            .cast(pl.UInt8)
            .alias("hour_utc")
        )
        .collect(engine="streaming")
    )

    return df


def ks_test_class(
    df: pl.DataFrame,
    class_col: str,
    pnl_col: str = "pnl_30s",
    sample_size: int = 1_000_000,
    seed: int = 42
):

    df_sampled = df.sample(n=min(sample_size, len(df)), seed=seed)
    all_pnl = df_sampled[pnl_col].drop_nulls().to_numpy()
    buckets = df_sampled[class_col].drop_nulls().unique().sort().to_list()

    rows = []
    for b in buckets:
        bucket_pnl = df_sampled.filter(pl.col(class_col) == b)[pnl_col].drop_nulls().to_numpy()
        stat, pval = ks_2samp(bucket_pnl, all_pnl)
        rows.append({"bucket": b, "ks_stat": round(stat, 4), "p_value": pval})

    return pl.DataFrame(rows)