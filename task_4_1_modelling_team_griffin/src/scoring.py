# scoring.py - compute PNL for windows / days / trades.


# imports.
from dataclasses import dataclass
import numpy as np


# constants.
US_PER_SEC = 1_000_000
US_PER_DAY = 86_400 * US_PER_SEC
DEFAULT_TURNOVER_CONSTRAINT = 500_000


# dataclasses.
@dataclass
class ScoreResult:
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
    filter_rate: float

    turnover_ok: bool

@dataclass
class ThresholdResult:
    threshold: float | None
    score: float
    pnl_all: float
    pnl_kept: float
    filter_rate: float
    kept_turnover_per_day: float

    turnover_ok: bool
    filter_rate_ok: bool
    valid: bool


# helper functions.
def _valid_mask(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray | None = None,
    f: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)

    if ts is not None:
        mask &= np.isfinite(ts)

    if f is not None:
        mask &= np.isfinite(f)

    return mask

def _weighted_mean(
    x: np.ndarray,
    w: np.ndarray,
) -> float:
    s = np.sum(w)
    if s <= 0:
        return 0.0
    return float(np.sum(w * x) / s)

def make_filter_from_threshold(
    toxicity: np.ndarray,
    threshold: float | None,
) -> np.ndarray:
    if threshold is None or not np.isfinite(threshold):
        return np.zeros(len(toxicity), dtype=np.int8)

    return (toxicity > threshold).astype(np.int8)


# main scoring.
def score_from_arrays(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    f: np.ndarray,
    turnover_constraint: float = DEFAULT_TURNOVER_CONSTRAINT
) -> ScoreResult:
    
    y   = np.asarray(y, dtype=np.float64)
    w   = np.asarray(w, dtype=np.float64)
    ts  = np.asarray(ts, dtype=np.int64)
    f   = np.asarray(f, dtype=np.float64)

    valid = _valid_mask(y, w, ts, f)

    y = y[valid]
    w = w[valid]
    ts = ts[valid]
    f = f[valid].astype(np.int8)

    kept = f == 0
    filt = f == 1

    if len(ts) > 1:
        n_days = max((int(ts.max()) - int(ts.min())) / US_PER_DAY, 1.0)
    else:
        n_days = 1.0

    pnl_all = _weighted_mean(y, w)

    if kept.any():
        pnl_kept = _weighted_mean(y[kept], w[kept])
        kept_turnover_total = float(np.sum(w[kept]))
    else:
        pnl_kept = 0.0
        kept_turnover_total = 0.0

    if filt.any():
        pnl_filtered = _weighted_mean(y[filt], w[filt])
    else:
        pnl_filtered = 0.0

    kept_turnover_per_day = kept_turnover_total / n_days

    return ScoreResult(
        pnl_all                 = round(pnl_all, 6),
        pnl_kept                = round(pnl_kept, 6),
        pnl_filtered            = round(pnl_filtered, 6),
        score                   = round(pnl_kept - pnl_all, 6),

        kept_turnover_total     = round(kept_turnover_total, 2),
        kept_turnover_per_day   = round(kept_turnover_per_day, 2),
        n_days                  = round(float(n_days), 2),

        n_total                 = int(len(y)),
        n_kept                  = int(kept.sum()),
        n_filtered              = int(filt.sum()),
        filter_rate             = round(float(filt.mean()), 4),

        turnover_ok=kept_turnover_per_day >= turnover_constraint
    )

def compute_daily_scores_from_arrays(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    f: np.ndarray
) -> np.ndarray:
    
    y   = np.asarray(y, dtype=np.float64)
    w   = np.asarray(w, dtype=np.float64)
    ts  = np.asarray(ts, dtype=np.int64)
    f   = np.asarray(f, dtype=np.float64)

    valid = _valid_mask(y, w, ts, f)

    y   = y[valid]
    w   = w[valid]
    ts  = ts[valid]
    f   = f[valid].astype(np.int8)


    day_ids = ts // US_PER_DAY
    unique_days = np.unique(day_ids)

    daily_scores = []

    for day in unique_days:
        mask = day_ids == day

        y_d = y[mask]
        w_d = w[mask]
        f_d = f[mask]

        pnl_all = _weighted_mean(y_d, w_d)

        kept = f_d == 0
        if kept.any():
            pnl_kept = _weighted_mean(y_d[kept], w_d[kept])
        else:
            pnl_kept = 0.0

        daily_scores.append(pnl_kept - pnl_all)

    return np.asarray(daily_scores, dtype=np.float64)

def compute_daily_pnl_kept_from_arrays(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    f: np.ndarray
) -> np.ndarray:
    
    y   = np.asarray(y, dtype=np.float64)
    w   = np.asarray(w, dtype=np.float64)
    ts  = np.asarray(ts, dtype=np.int64)
    f   = np.asarray(f, dtype=np.float64)

    valid = _valid_mask(y, w, ts, f)

    y   = y[valid]
    w   = w[valid]
    ts  = ts[valid]
    f   = f[valid].astype(np.int8)

    day_ids = ts // US_PER_DAY
    unique_days = np.unique(day_ids)

    daily_pnl_kept = []

    for day in unique_days:
        mask = day_ids == day

        y_d = y[mask]
        w_d = w[mask]
        f_d = f[mask]

        kept = f_d == 0

        if kept.any():
            pnl_kept = _weighted_mean(y_d[kept], w_d[kept])
        else:
            pnl_kept = 0.0

        daily_pnl_kept.append(pnl_kept)

    return np.asarray(daily_pnl_kept, dtype=np.float64)

def compute_trade_diagnostics_from_arrays(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    f: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute trade-level diagnostics for score decomposition.

    Parameters
    ----------
    y:
        Trade-level pnl, for example pnl_30s.

    w:
        Trade weights.

    ts:
        Timestamps in microseconds.

    f:
        Binary filter:
            1 = filtered / removed,
            0 = kept.

    Returns
    -------
    diagnostics:
        Dictionary with trade-level arrays.

    Notes
    -----
    The column `score_contribution` satisfies:

        score_contribution.sum() == pnl_kept - pnl_all

    up to floating point error.
    """

    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.int64)
    f = np.asarray(f, dtype=np.float64)

    valid = _valid_mask(y, w, ts, f)

    y = y[valid]
    w = w[valid]
    ts = ts[valid]
    f = f[valid].astype(np.int8)

    n = len(y)

    is_kept = f == 0
    is_filtered = f == 1

    total_weight = float(np.sum(w))
    kept_weight = float(np.sum(w[is_kept]))

    pnl_all_contribution = np.zeros(n, dtype=np.float64)
    pnl_kept_contribution = np.zeros(n, dtype=np.float64)

    if total_weight > 0:
        pnl_all_contribution = w * y / total_weight

    if kept_weight > 0:
        pnl_kept_contribution[is_kept] = w[is_kept] * y[is_kept] / kept_weight

    score_contribution = pnl_kept_contribution - pnl_all_contribution

    kept_pnl = np.full(n, np.nan, dtype=np.float64)
    kept_pnl[is_kept] = y[is_kept]

    filtered_pnl = np.full(n, np.nan, dtype=np.float64)
    filtered_pnl[is_filtered] = y[is_filtered]

    return {
        "timestamp": ts,
        "pnl": y,
        "weight": w,
        "filter": f,
        "is_kept": is_kept,
        "is_filtered": is_filtered,

        # Contributions to weighted aggregate values.
        "pnl_all_contribution": pnl_all_contribution,
        "pnl_kept_contribution": pnl_kept_contribution,
        "score_contribution": score_contribution,

        # Raw per-trade pnl separated by decision.
        "kept_pnl": kept_pnl,
        "filtered_pnl": filtered_pnl
    }


# prediction conversion.
def to_toxicity_score(
    pred: np.ndarray,
    prediction_type: str
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)

    if prediction_type == "pnl":
        return -pred

    if prediction_type in {"toxicity", "probability"}:
        return pred

    raise ValueError(
        "Unknown prediction_type. "
        "Expected one of: 'pnl', 'toxicity', 'probability'."
    )


# threshold tuning.
def tune_threshold_from_toxicity(
    toxicity: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    max_filter_rate: float,
    min_filter_rate: float | None = None,
    n_grid: int = 200,
    turnover_constraint: float = DEFAULT_TURNOVER_CONSTRAINT
) -> ThresholdResult:
    toxicity = np.asarray(toxicity, dtype=np.float64)

    valid = (
        np.isfinite(toxicity)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
        & np.isfinite(ts)
    )

    toxicity = toxicity[valid]
    y = np.asarray(y, dtype=np.float64)[valid]
    w = np.asarray(w, dtype=np.float64)[valid]
    ts = np.asarray(ts, dtype=np.int64)[valid]

    if min_filter_rate is None:
        min_filter_rate = 0.0

    thresholds = np.nanquantile(
        toxicity,
        np.linspace(0.0, 1.0, n_grid),
    )

    thresholds = np.unique(thresholds[np.isfinite(thresholds)])

    best_result: ThresholdResult | None = None

    for threshold in thresholds:
        f = make_filter_from_threshold(toxicity, threshold)

        score = score_from_arrays(
            y   = y,
            w   = w,
            ts  = ts,
            f   = f,
            turnover_constraint = turnover_constraint
        )

        filter_rate_ok = (
            score.filter_rate <= max_filter_rate
            and score.filter_rate >= min_filter_rate
        )

        valid_threshold = score.turnover_ok and filter_rate_ok

        if not valid_threshold:
            continue

        candidate = ThresholdResult(
            threshold               = float(threshold),
            score                   = score.score,
            pnl_all                 = score.pnl_all,
            pnl_kept                = score.pnl_kept,
            filter_rate             = score.filter_rate,
            kept_turnover_per_day   = score.kept_turnover_per_day,
            turnover_ok             = score.turnover_ok,
            filter_rate_ok          = filter_rate_ok,
            valid                   = True
        )

        if best_result is None or candidate.score > best_result.score:
            best_result = candidate

    if best_result is None:
        return ThresholdResult(
            threshold               = None,
            score                   = 0.0,
            pnl_all                 = 0.0,
            pnl_kept                = 0.0,
            filter_rate             = 0.0,
            kept_turnover_per_day   = 0.0,
            turnover_ok             = False,
            filter_rate_ok          = False,
            valid                   = False
        )

    return best_result