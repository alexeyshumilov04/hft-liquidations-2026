# imports.
import numpy as np
from dataclasses import dataclass


# PSI - Population Stability Index.
def compute_psi(
    expected:   np.ndarray,
    actual:     np.ndarray,
    n_bins:     int = 10
) -> float:
    """
    Population Stability Index between two distributions.

    PSI = sum( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

    Interpretation:
        - PSI < 0.10 - no significant change.
        - 0.10 - 0.25 - moderate change, investigate.
        - PSI > 0.25  - significant change, action required.

    Uses quantile-based binning from `expected` distribution.

    NaN values are dropped before computation.
    """

    expected    = expected[~np.isnan(expected)]
    actual      = actual[~np.isnan(actual)]

    if len(expected) < 20 or len(actual) < 20: return 0.0

    # quntile bin edges from expected distribution.
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges     = np.percentile(expected, quantiles)
    edges[0]  = -np.inf
    edges[-1] = np.inf

    # bin count -> proportions.
    exp_counts = np.histogram(expected, bins=edges)[0].astype(float)
    act_counts = np.histogram(actual, bins=edges)[0].astype(float)

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    # avoid log(0) and division by zero.
    eps     = 1e-8
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)

    # main formula.
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))

    return round(float(psi), 6)

def compute_psi_matrix(
    train_features: dict[str, np.ndarray],
    val_features:   dict[str, np.ndarray],
    n_bins:         int = 10,
) -> dict[str, float]:
    """
    Compute PSI for each feature: train vs validation distribution.

    Returns dict: feature_name -> PSI value.
    """
    result = {}
    for name in train_features:
        if name in val_features:
            result[name] = compute_psi(
                expected    = train_features[name],
                actual      = val_features[name],
                n_bins      = n_bins
            )

    return result

def psi_verdict(
    psi: float
) -> str:
    """
    Readable interpretation of PSI value.
    """
    if psi < 0.10: return "OK"
    elif psi < 0.25: return "INVESTIGATE"
    else: return "ACTION REQUIRED"


# CUSUM - Cumulative Sum Control Chart.
@dataclass
class CUSUMResult:
    """
    CUSUM chart data and alarm detection.
    """
    cusum_pos:      np.ndarray     # upper CUSUM.
    cusum_neg:      np.ndarray     # lower CUSUM.
    threshold:      float          # alarm threshold (H).
    alarms_pos:     np.ndarray     # indices where upper CUSUM crossed H.
    alarms_neg:     np.ndarray     # indices where lower CUSUM crossed -H.
    has_alarm:      bool

def cusum_chart(
    x:                  np.ndarray,
    target_mean:        float = 0.0,
    threshold_sigma:    float = 4.0,
    slack:              float | None = None
) -> CUSUMResult:
    """
    Two-sided CUSUM for detecting mean shift in a time series.

    Parameters:
        x:               observed values (daily Score).
        target_mean:     expected mean under "in-control" state.
        threshold_sigma: alarm threshold as multiple of std(x).
        slack:           allowance parameter K (default: 0.5 * std).

    CUSUM_pos detects upward shifts, CUSUM_neg detects downward shifts.
    Alarm when |CUSUM| > threshold.
    """

    # initial parametrs.
    T       = x.size
    sigma   = x.std() if x.std() > 0 else 1.0
    H       = threshold_sigma * sigma

    # slack.
    if slack is None: slack = 0.5 * sigma

    cusum_pos = np.zeros(T)
    cusum_neg = np.zeros(T)

    for t in range(1, T):
        cusum_pos[t] = max(0, cusum_pos[t-1] + (x[t] - target_mean) - slack)
        cusum_neg[t] = min(0, cusum_neg[t-1] + (x[t] - target_mean) + slack)

    alarms_pos = np.where(cusum_pos > H)[0]
    alarms_neg = np.where(cusum_neg < -H)[0]

    return CUSUMResult(
        cusum_pos   = cusum_pos,
        cusum_neg   = cusum_neg,
        threshold   = H,
        alarms_pos  = alarms_pos,
        alarms_neg  = alarms_neg,
        has_alarm   = len(alarms_pos) > 0 or len(alarms_neg) > 0
    )


# Rolling Score with control bands.
@dataclass
class RollingMonitorResult:
    """
    Rolling monitor chart data.
    """
    rolling_mean:   np.ndarray     # rolling mean of Score.
    upper_band:     float          # train mean + n_sigma * train std.
    lower_band:     float          # train mean - n_sigma * train std.
    center:         float          # train mean.
    breaches_above: np.ndarray     # indices where rolling > upper.
    breaches_below: np.ndarray     # indices where rolling < lower.
    has_breach:     bool

def rolling_score_monitor(
    daily_scores_train: np.ndarray,
    daily_scores_full:  np.ndarray,
    window:             int = 7,
    n_sigma:            float = 2.0
) -> RollingMonitorResult:
    """
    Rolling-mean Score with control bands from train period.

    Bands are set at train_mean +- n_sigma * train_std.
    Breach = rolling mean outside bands -> possible degradation or regime change.
    """
    # train statistics for bands.
    train_mean = daily_scores_train.mean()
    train_std  = daily_scores_train.std()

    upper = train_mean + n_sigma * train_std
    lower = train_mean - n_sigma * train_std

    # rolling mean on full series.
    T = daily_scores_full.size
    rolling = np.full(T, np.nan)
    for t in range(window - 1, T):
        rolling[t] = daily_scores_full[t - window + 1: t + 1].mean()

    valid       = ~np.isnan(rolling)
    above       = np.where(valid & (rolling > upper))[0]
    below       = np.where(valid & (rolling < lower))[0]

    return RollingMonitorResult(
        rolling_mean    = rolling,
        upper_band      = round(upper, 6),
        lower_band      = round(lower, 6),
        center          = round(train_mean, 6),
        breaches_above  = above,
        breaches_below  = below,
        has_breach      = len(above) > 0 or len(below) > 0
    )


# Killswitch evaluation.
@dataclass
class KillswitchStatus:
    """
    Killswitch evaluation result.
    """
    triggered:              bool
    reasons:                list[str]
    consecutive_neg_days:   int
    max_psi:                float
    max_psi_feature:        str
    turnover_ok:            bool

def evaluate_killswitch(
    daily_scores_recent:    np.ndarray,
    psi_values:             dict[str, float],
    kept_turnover_per_day:  float,
    turnover_constraint:    float = 500_000,
    max_consecutive_neg:    int = 7,
    psi_threshold:          float = 0.25
) -> KillswitchStatus:
    """
    Evaluate killswitch rules:
      1. Score < 0 for N consecutive days -> STOP.
      2. PSI > threshold on any core feature -> RECALIBRATE.
      3. Turnover below constraint -> STOP.
    """
    reasons = []

    # rule 1: consecutive negative days.
    consec = 0
    max_consec = 0
    for s in daily_scores_recent:
        if s < 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    if max_consec >= max_consecutive_neg:
        reasons.append(
            f"Score < 0 for {max_consec} consecutive days (threshold: {max_consecutive_neg})"
        )

    # rule 2: PSI breach.
    max_psi = 0.0
    max_psi_feat = ""
    for feat, psi in psi_values.items():
        if psi > max_psi:
            max_psi = psi
            max_psi_feat = feat
        if psi > psi_threshold:
            reasons.append(f"PSI({feat}) = {psi:.3f} > {psi_threshold}")

    # rule 3: turnover.
    turnover_ok = kept_turnover_per_day >= turnover_constraint
    if not turnover_ok:
        reasons.append(
            f"Turnover {kept_turnover_per_day:,.0f} < {turnover_constraint:,.0f}"
        )

    return KillswitchStatus(
        triggered               = len(reasons) > 0,
        reasons                 = reasons,
        consecutive_neg_days    = max_consec,
        max_psi                 = round(max_psi, 4),
        max_psi_feature         = max_psi_feat,
        turnover_ok             = turnover_ok
    )