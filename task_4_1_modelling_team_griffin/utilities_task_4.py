# imports.
import datetime
import numpy as np
import polars as pl
from tqdm.auto import tqdm
from dataclasses import dataclass
from utilities_features import extract_Xyw, FEATURES


# costants.
US_PER_SEC = 1_000_000
US_PER_DAY = 86_400 * US_PER_SEC

HORIZONS_S = [30, 120, 300]
TURNOVER_CONSTRAINT = 500_000

def date_to_us(
    year: int, 
    month: int, 
    day: int
) -> int:
    
    """
    UTC date -> microseconds since epoch at start of day.
    """

    dt = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * US_PER_SEC)

WINDOWS = {
    "W1": {
        "train_start": date_to_us(2025, 11, 1),
        "train_end":   date_to_us(2025, 12, 1),
        "embargo_end": date_to_us(2025, 12, 8),
        "test_end":    date_to_us(2026,  1, 1)
    },
    "W2": {
        "train_start": date_to_us(2025, 11, 1),
        "train_end":   date_to_us(2026,  1, 1),
        "embargo_end": date_to_us(2026,  1, 8),
        "test_end":    date_to_us(2026,  2, 1)
    },
    "W3": {
        "train_start": date_to_us(2025, 11, 1),
        "train_end":   date_to_us(2026,  2, 1),
        "embargo_end": date_to_us(2026,  2, 8),
        "test_end":    date_to_us(2026,  3, 1)
    },
    "W4": {
        "train_start": date_to_us(2025, 11, 1),
        "train_end":   date_to_us(2026,  3, 1),
        "embargo_end": date_to_us(2026,  3, 8),
        "test_end":    date_to_us(2026,  4, 1)
    },
    "W5": {
        "train_start": date_to_us(2025, 11, 1),
        "train_end":   date_to_us(2026,  4, 1),
        "embargo_end": date_to_us(2026,  4, 8),
        "test_end":    date_to_us(2026,  4, 29)
    }
}


# window split.
def get_window_split(
    df: pl.DataFrame,
    window_id: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Return outer train and outer test for W1-W5.
    Embargo rows are dropped.
    """
    if window_id not in WINDOWS:
        raise ValueError(f"Unknown window_id={window_id}. Expected one of {list(WINDOWS)}.")

    w = WINDOWS[window_id]

    df_train = df.filter(
        (pl.col("timestamp") >= w["train_start"]) & (pl.col("timestamp") < w["train_end"])
    )

    df_test = df.filter(
        (pl.col("timestamp") >= w["embargo_end"]) & (pl.col("timestamp") < w["test_end"])
    )

    return df_train, df_test

def split_train_calib_by_time(
    df_train: pl.DataFrame,
    calib_frac: float = 0.20,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split outer train into model_train and calibration by time.

    model_train:
        earlier part of outer train, used for fitting model parameters.

    calib:
        latest part of outer train, used only for threshold tuning.
    """
    if not (0.0 < calib_frac < 1.0):
        raise ValueError("calib_frac must be between 0 and 1.")

    if df_train.height == 0:
        return df_train, df_train

    ts_min = int(df_train["timestamp"].min())
    ts_max = int(df_train["timestamp"].max())

    split_ts = int(ts_min + (1.0 - calib_frac) * (ts_max - ts_min))

    df_model_train = df_train.filter(pl.col("timestamp") < split_ts)
    df_calib = df_train.filter(pl.col("timestamp") >= split_ts)

    return df_model_train, df_calib


# Score result.
@dataclass
class ArrayScoreResult:
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

def score_from_arrays(
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    f: np.ndarray,
) -> ArrayScoreResult:
    """
    Compute official-style Score from arrays.

    f:
        binary filter, 1=filter/remove, 0=keep.
    """

    # Nan check.
    valid = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(f)

    y   = y[valid]
    w   = w[valid]
    ts  = ts[valid]
    f   = f[valid]


    kept = f == 0
    filt = f == 1

    n_days = (
        max((int(ts.max()) - int(ts.min())) / US_PER_DAY, 1.0)
        if len(ts) > 1 else 1.0
    )

    pnl_all = float(np.sum(w * y) / np.sum(w))

    if kept.any():
        w_kept = w[kept]
        pnl_kept = float(np.sum(w_kept * y[kept]) / np.sum(w_kept))
        kept_turnover_total = float(np.sum(w_kept))
    else:
        pnl_kept = 0.0
        kept_turnover_total = 0.0

    if filt.any():
        w_filt = w[filt]
        pnl_filtered = float(np.sum(w_filt * y[filt]) / np.sum(w_filt))
    else:
        pnl_filtered = 0.0

    kept_turnover_per_day = kept_turnover_total / n_days

    return ArrayScoreResult(
        pnl_all                 = round(pnl_all, 6),
        pnl_kept                = round(pnl_kept, 6),
        pnl_filtered            = round(pnl_filtered, 6),
        score                   = round(pnl_kept - pnl_all, 6),
        kept_turnover_total     = round(kept_turnover_total, 2),
        kept_turnover_per_day   = round(kept_turnover_per_day, 2),
        n_days                  = round(n_days, 2),
        n_total                 = int(len(y)),
        n_kept                  = int(kept.sum()),
        n_filtered              = int(filt.sum()),
        filter_rate             = round(float(filt.mean()), 4),
        turnover_ok             = kept_turnover_per_day >= TURNOVER_CONSTRAINT
    )


# Daily Scores.
def compute_daily_scores_from_arrays(
    y:  np.ndarray,
    w:  np.ndarray,
    ts: np.ndarray,
    f:  np.ndarray,
) -> np.ndarray:
    """
    Compute per-day Score from OOS arrays.

    Score_d = PnL_kept_d - PnL_all_d (weighted, per calendar day).

    Returns
    -------
    daily_scores : array of shape (n_days,), one Score per day.
    """
    day_ids = ts // US_PER_DAY
    unique_days = np.unique(day_ids)

    daily_scores = []

    for d in unique_days:
        mask = day_ids == d

        y_d = y[mask]
        w_d = w[mask]
        f_d = f[mask]

        sum_w = w_d.sum()
        if sum_w == 0:
            continue

        pnl_all = (w_d * y_d).sum() / sum_w

        kept = f_d == 0
        w_kept = w_d[kept]
        sum_w_kept = w_kept.sum()

        if sum_w_kept == 0:
            pnl_kept = 0.0
        else:
            pnl_kept = (w_kept * y_d[kept]).sum() / sum_w_kept

        daily_scores.append(pnl_kept - pnl_all)

    return np.array(daily_scores, dtype=np.float64)

def collect_oos_arrays(
    df: pl.DataFrame,
    tau: int,
    train_fn,
    predict_fn,
    prediction_type: str,
    max_filter_rate: float = 0.30,
    calib_frac: float = 0.20,
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run walk-forward and concatenate OOS (y, w, ts, f) across W1-W5.

    Returns
    -------
    y_oos, w_oos, ts_oos, f_oos — concatenated test arrays.
    """
    if features is None:
        features = FEATURES

    all_y, all_w, all_ts, all_f = [], [], [], []

    for wid in ["W1", "W2", "W3", "W4", "W5"]:
        df_outer_train, df_test = get_window_split(df, wid)
        df_model_train, df_calib = split_train_calib_by_time(df_outer_train, calib_frac)

        X_tr, y_tr, w_tr, ts_tr = extract_Xyw(df_model_train, tau, features)
        X_ca, y_ca, w_ca, ts_ca = extract_Xyw(df_calib, tau, features)
        X_te, y_te, w_te, ts_te = extract_Xyw(df_test, tau, features)

        model = train_fn(X_tr, y_tr, w_tr)

        pred_ca = predict_fn(model, X_ca)
        pred_te = predict_fn(model, X_te)

        tox_ca = to_toxicity_score(pred_ca, prediction_type)
        tox_te = to_toxicity_score(pred_te, prediction_type)

        thr = tune_threshold_from_toxicity(
            tox_ca, y_ca, w_ca, ts_ca,
            max_filter_rate=max_filter_rate,
        )

        if thr.valid and np.isfinite(thr.threshold):
            f_te = (tox_te > thr.threshold).astype(np.int8)
        else:
            f_te = np.zeros(len(tox_te), dtype=np.int8)

        all_y.append(y_te)
        all_w.append(w_te)
        all_ts.append(ts_te)
        all_f.append(f_te)

    return (
        np.concatenate(all_y),
        np.concatenate(all_w),
        np.concatenate(all_ts),
        np.concatenate(all_f)
    )

def compute_daily_pnl_kept_from_arrays(
    y:  np.ndarray,
    w:  np.ndarray,
    ts: np.ndarray,
    f:  np.ndarray,
) -> np.ndarray:
    """
    Compute per-day PnL_kept from OOS arrays.

    PnL_kept_d = weighted average pnl over kept trades for each calendar day.

    f:
        1 = filtered / removed
        0 = kept
    """

    valid = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.isfinite(ts) & np.isfinite(f)

    y  = y[valid]
    w  = w[valid]
    ts = ts[valid]
    f  = f[valid]

    day_ids = ts // US_PER_DAY
    unique_days = np.unique(day_ids)

    daily_pnl_kept = []

    for d in unique_days:
        mask = day_ids == d

        y_d = y[mask]
        w_d = w[mask]
        f_d = f[mask]

        kept = f_d == 0

        if not kept.any():
            continue

        w_kept = w_d[kept]
        sum_w_kept = w_kept.sum()

        if sum_w_kept <= 0:
            continue

        pnl_kept = (w_kept * y_d[kept]).sum() / sum_w_kept

        daily_pnl_kept.append(pnl_kept)

    return np.array(daily_pnl_kept, dtype=np.float64)

# Threshold Result.
@dataclass
class ThresholdResult:
    threshold: float
    score: float
    pnl_all: float
    pnl_kept: float
    filter_rate: float
    kept_turnover_per_day: float
    turnover_ok: bool
    filter_rate_ok: bool
    valid: bool

def tune_threshold_from_toxicity(
    toxicity: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    ts: np.ndarray,
    n_grid: int = 200,
    max_filter_rate: float | None = 0.30,
    min_filter_rate: float | None = None,
) -> ThresholdResult:
    """
    Select threshold on calibration data.

    Convention:
        higher toxicity -> more toxic.
        filter rule: f = 1{toxicity > threshold}.
    """
    toxicity    = np.asarray(toxicity, dtype=np.float64)
    y           = np.asarray(y, dtype=np.float64)
    w           = np.asarray(w, dtype=np.float64)
    ts          = np.asarray(ts, dtype=np.int64)

    valid = np.isfinite(toxicity) & np.isfinite(y) & np.isfinite(w) & (w > 0)

    toxicity    = toxicity[valid]
    y           = y[valid]
    w           = w[valid]
    ts          = ts[valid]

    qs = np.linspace(0.0, 1.0, n_grid)
    candidates = np.unique(np.quantile(toxicity, qs))

    best = None

    for c in candidates:
        f = (toxicity > c).astype(np.int8)
        score_res = score_from_arrays(y=y, w=w, ts=ts, f=f)

        filter_rate_ok = True

        if max_filter_rate is not None:
            filter_rate_ok = filter_rate_ok and (score_res.filter_rate <= max_filter_rate)

        if min_filter_rate is not None:
            filter_rate_ok = filter_rate_ok and (score_res.filter_rate >= min_filter_rate)

        if not score_res.turnover_ok or not filter_rate_ok:
            continue

        candidate = ThresholdResult(
            threshold               =float(c),
            score                   = score_res.score,
            pnl_all                 = score_res.pnl_all,
            pnl_kept                = score_res.pnl_kept,
            filter_rate             = score_res.filter_rate,
            kept_turnover_per_day   = score_res.kept_turnover_per_day,
            turnover_ok             = True,
            filter_rate_ok          = True,
            valid                   = True
        )

        if best is None or candidate.score > best.score:
            best = candidate

    if best is not None:
        return best

    f0 = np.zeros(len(y), dtype=np.int8)
    score_res = score_from_arrays(y=y, w=w, ts=ts, f=f0)

    return ThresholdResult(
        threshold               = np.inf,
        score                   = score_res.score,
        pnl_all                 = score_res.pnl_all,
        pnl_kept                = score_res.pnl_kept,
        filter_rate             = score_res.filter_rate,
        kept_turnover_per_day   = score_res.kept_turnover_per_day,
        turnover_ok             = score_res.turnover_ok,
        filter_rate_ok          = True,
        valid                   = False
    )

def to_toxicity_score(
    pred: np.ndarray,
    prediction_type: str,
) -> np.ndarray:
    """
    Convert model output to toxicity score.

    prediction_type='pnl':
        pred is predicted PnL. Low predicted PnL is toxic.
        toxicity = -pred.

    prediction_type='toxicity':
        pred is already toxicity score. High value is toxic.
    """
    pred = np.asarray(pred, dtype=np.float64)

    if prediction_type == "pnl": return -pred

    if prediction_type == "toxicity": return pred

    raise ValueError("prediction_type must be 'pnl' or 'toxicity'.")


@dataclass
class WindowResult:
    window_id: str
    tau: int

    n_model_train: int
    n_calib: int
    n_test: int

    threshold: float
    threshold_valid: bool

    calib_score: float
    test_score: float
    test_pnl_all: float
    test_pnl_kept: float
    test_pnl_filtered: float

    test_filter_rate: float
    test_kept_turnover_per_day: float
    test_turnover_ok: bool

    test_ts: np.ndarray
    test_toxicity: np.ndarray
    test_f: np.ndarray

def run_walk_forward(
    df: pl.DataFrame,
    tau: int,
    train_fn,
    predict_fn,
    prediction_type: str,
    max_filter_rates: list[float],
    features: list[str] | None = None,
    window_ids: list[str] | None = None,
    calib_frac: float = 0.20,
    min_filter_rate: float | None = None,
    n_threshold_grid: int = 200,
) -> tuple[pl.DataFrame, list[dict]]:
    """
    Walk-forward evaluation for many max_filter_rate values.

    Model is trained only once per window.
    Thresholds are tuned for different max_filter_rate values.

    Returns:

        result_df:
            Polars DataFrame with metrics.
        models:
            list of dictionaries with trained models and metadata.
    """

    if features is None:
        features = FEATURES

    if window_ids is None:
        window_ids = ["W1", "W2", "W3", "W4", "W5"]

    rows = []
    models = []

    for wid in tqdm(window_ids, desc="Walk-forward windows"):

        # train / test split.
        df_outer_train, df_test = get_window_split(df, wid)

        # model_train / calibration split.
        df_model_train, df_calib = split_train_calib_by_time(
            df_outer_train,
            calib_frac=calib_frac,
        )

        # extract arrays.
        X_tr, y_tr, w_tr, ts_tr = extract_Xyw(df_model_train, tau, features)
        X_ca, y_ca, w_ca, ts_ca = extract_Xyw(df_calib, tau, features)
        X_te, y_te, w_te, ts_te = extract_Xyw(df_test, tau, features)

        # train model once.
        model = train_fn(X_tr, y_tr, w_tr)

        # save trained model.
        models.append({
            "window": wid,
            "tau": tau,
            "model": model,
            "features": features,
            "n_model_train": int(X_tr.shape[0]),
            "n_calib": int(X_ca.shape[0]),
            "n_test": int(X_te.shape[0]),
        })

        # predict once.
        pred_calib = predict_fn(model, X_ca)
        pred_test = predict_fn(model, X_te)

        toxicity_calib = to_toxicity_score(pred_calib, prediction_type)
        toxicity_test = to_toxicity_score(pred_test, prediction_type)

        # now only threshold changes.
        for max_fr in max_filter_rates:

            thr = tune_threshold_from_toxicity(
                toxicity        = toxicity_calib,
                y               = y_ca,
                w               = w_ca,
                ts              = ts_ca,
                n_grid          = n_threshold_grid,
                max_filter_rate = max_fr,
                min_filter_rate = min_filter_rate,
            )

            if thr.valid and np.isfinite(thr.threshold):
                f_test = (toxicity_test > thr.threshold).astype(np.int8)
            else:
                f_test = np.zeros(len(toxicity_test), dtype=np.int8)

            test_score_obj = score_from_arrays(
                y  = y_te,
                w  = w_te,
                ts = ts_te,
                f  = f_test,
            )

            rows.append({
                "window": wid,
                "tau": f"{tau}s",
                "max_filter_rate": max_fr,

                "n_model_train": int(X_tr.shape[0]),
                "n_calib": int(X_ca.shape[0]),
                "n_test": int(X_te.shape[0]),

                "threshold": float(thr.threshold),
                "threshold_valid": bool(thr.valid),
                "calib_score": float(thr.score),

                "test_score": float(test_score_obj.score),
                "test_pnl_all": float(test_score_obj.pnl_all),
                "test_pnl_kept": float(test_score_obj.pnl_kept),
                "test_pnl_filtered": float(test_score_obj.pnl_filtered),
                "test_filter_rate (%)": round(test_score_obj.filter_rate * 100, 2),
                "test_kept_usd/day": float(test_score_obj.kept_turnover_per_day),
                "test_turnover_ok": bool(test_score_obj.turnover_ok),
            })

    result_df = pl.DataFrame(rows)

    return result_df, models

def results_summary_table(results: list[WindowResult]) -> pl.DataFrame:
    """Convert WindowResult list into readable Polars DataFrame."""
    rows = []

    for r in results:
        rows.append({
            "window": r.window_id,
            "tau": f"{r.tau}s",
            "n_model_train": r.n_model_train,
            "n_calib": r.n_calib,
            "n_test": r.n_test,
            "threshold": round(r.threshold, 6) if np.isfinite(r.threshold) else "inf",
            "threshold_valid": r.threshold_valid,
            "calib_score": r.calib_score,
            "test_score": r.test_score,
            "test_pnl_all": r.test_pnl_all,
            "test_pnl_kept": r.test_pnl_kept,
            "test_pnl_filtered": r.test_pnl_filtered,
            "test_filter_rate (%)": round(r.test_filter_rate * 100, 2),
            "test_kept_usd/day": r.test_kept_turnover_per_day,
            "test_turnover_ok": r.test_turnover_ok
        })

    return pl.DataFrame(rows)

def concatenate_oos_filters(
    results: list[WindowResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate OOS timestamps, toxicity scores and filters from all windows.

    Returns
    -------
    ts_all, toxicity_all, f_all
    """
    if not results:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int8)
        )

    ts_all = np.concatenate([r.test_ts for r in results])
    toxicity_all = np.concatenate([r.test_toxicity for r in results])
    f_all = np.concatenate([r.test_f for r in results])

    order = np.argsort(ts_all)

    return ts_all[order], toxicity_all[order], f_all[order]


