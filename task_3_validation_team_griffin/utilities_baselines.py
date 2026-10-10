# imports.
import numpy as np
import polars as pl


# Simple filters.
def no_filter(
    n:int
) -> np.ndarray:
    """
    Keep all trades.

    Score = 0 by defenition.
    """
    return np.zeros(n, dtype=np.int8)

def random_filter(
    n:           int,
    filter_rate: float,
    seed:        int = 42 
) -> np.ndarray:
    """
    Random "Coin - Flip" filter.

    Null hypothesis: a random filter should produce Score ~= 0.
    """
    rng = np.random.default_rng(seed)
    return (rng.random(n) < filter_rate).astype(np.int8)

def random_filter_ensemble(
    n:              int,
    filter_rate:    float,
    n_runs:         int = 100,
    base_seed:      int = 0
) -> list[np.ndarray]:
    """
    Generate multiple random filters for empirical distribution.
    """
    return [
        random_filter(
            n           = n,
            filter_rate = filter_rate,
            seed        = base_seed + i
        )
        for i in range(n_runs)
    ]


# Threshold filter.
def threshold_filter(
    values:     np.ndarray,
    threshold:  float,
    direction:  str = "above"
) -> np.ndarray:
    """
    Filter trades where feature is above or below threshold.

    - direction="above": filter where values > threshold  (remove high-toxicity).

    - direction="below": filter where values < threshold  (remove low-value).

    NaN values are never filtered (kept).
    """

    mask = np.isnan(values)

    if "above" == direction:
        f = (values > threshold).astype(np.int8)
    elif "below" == direction:
        f = (values < threshold).astype(np.int8)
    else:
        raise ValueError(f"direction must be 'above' or 'below', got '{direction}'")
    
    f[mask] = 0    # NaN -> keep (don't filter what we can't evaluate).
    return f

def scan_thresholds(
    values:     np.ndarray,
    quantiles:  list[float] | None = None,
    direction:  str = "above"
) -> list[tuple[float, float, np.ndarray]]:
    """
    Generate threshold filters at each quantile of `values`.

    Returns list of (quantile, threshold_value, filter_array).
    Useful for finding the best threshold per feature.

    Default quantiles: 10%, 20%, ..., 90%.
    """

    if quantiles is None:
        quantiles = [q / 100 for q in range(10, 100, 10)]

    valid = values[~np.isnan(values)]
    if 0 == valid.size : return []

    results = []
    for q in quantiles:
        thresh  = float(np.quantile(valid, q))
        f       = threshold_filter(values, thresh, direction)
        results.append((q, thresh, f))

    return results


# Hour filter.
def hour_filter(
    hours:       np.ndarray,
    toxic_hours: list[int]
) -> np.ndarray:
    """
    Filter trades at specific UTC hours.

    From Task 2 Executive Summary: hour 22 UTC is worst (-0.73 bps at 30s).
    """
    mask = np.isin(hours, toxic_hours)
    return mask.astype(np.int8)


# Oracle filter (look-ahead).
def oracle_filter(
    pnl: np.ndarray
) -> np.ndarray:
    """
    Filter trades with pnl < 0.

    This is the theoretical upper bound on Score.

    NaN pnl -> keep (not enough info to filter).
    """
    f           = np.zeros(len(pnl), dtype=np.int8)
    valid       = ~np.isnan(pnl)
    f[valid]    = (pnl[valid] < 0).astype(np.int8)

    return f


# Composite filter helpers.
def combine_filters_or(
    filters: list[np.ndarray]
) -> np.ndarray:
    
    """
    Filter if ANY filter says to filter (union).
    """

    result = np.zeros(len(filters[0]), dtype=np.int8)

    for f in filters:
        result = np.maximum(result, f)

    return result

def combine_filters_and(
    filters: list[np.ndarray]
) -> np.ndarray:
    
    """
    Filter only if ALL filters agree to filter (intersection).
    """

    result = np.ones(len(filters[0]), dtype=np.int8)

    for f in filters:
        result = np.minimum(result, f)

    return result


# Feature extraction helpers.
def get_feature_array(
    df:       pl.DataFrame,
    col_name: str
) -> np.ndarray:
    
    """
    Extract feature column as float64 numpy array.
    """
    return df[col_name].to_numpy().astype(np.float64)

def get_pnl_array(
    df:  pl.DataFrame,
    tau: int,
) -> np.ndarray:
    
    """
    Extract pnl column as float64 numpy array (NaN preserved).
    """

    col = f"pnl_{tau}s"
    
    return df[col].to_numpy().astype(np.float64)