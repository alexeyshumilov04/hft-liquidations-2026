# import.
import numpy as np
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr


# Weighted classification metrics.
@dataclass
class ClassificationMetrics:
    """Weighted classification metrics for filter as classifier."""
    name:               str
    precision_weighted: float     # of what we filtered, how much was truly toxic.
    recall_weighted:    float     # of all toxic trades, how much did we catch.
    f1_weighted:        float
    mcc_weighted:       float     # Matthews correlation coefficient.
    filter_rate:        float     # fraction of trades filtered.
    toxic_rate:         float     # fraction of trades that are actually toxic.

def compute_classification_metrics(
    f:       np.ndarray,
    pnl:     np.ndarray,
    w:       np.ndarray,
    name:    str = ""
) -> ClassificationMetrics:
    """
    Compute weighted precision/recall/F1/MCC for the filter.

    Ground truth: y = 1 if pnl < 0 (toxic), y = 0 otherwise.
    Prediction:   f = 1 (filter/remove), f = 0 (keep).

    NaN pnl rows are excluded from evaluation.
    """
    valid = ~np.isnan(pnl)
    f_v = f[valid].astype(np.float64)
    w_v = w[valid].astype(np.float64)
    y_v = (pnl[valid] < 0).astype(np.float64)   # ground truth: toxic.

    # weighted confusion matrix components.
    tp = np.sum(w_v * f_v * y_v)             # filtered AND toxic.
    fp = np.sum(w_v * f_v * (1 - y_v))       # filtered BUT profitable.
    fn = np.sum(w_v * (1 - f_v) * y_v)       # kept BUT toxic.
    tn = np.sum(w_v * (1 - f_v) * (1 - y_v)) # kept AND profitable.

    # precision: of filtered, fraction truly toxic.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # recall: of toxis, fraction we caught.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # f1 score.
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # MCC (weighted).
    numer = tp * tn - fp * fn
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numer / denom if denom > 0 else 0.0

    total_w = np.sum(w_v)
    filter_rate = np.sum(w_v * f_v) / total_w if total_w > 0 else 0.0
    toxic_rate  = np.sum(w_v * y_v) / total_w if total_w > 0 else 0.0

    return ClassificationMetrics(
        name                = name,
        precision_weighted  = round(precision, 6),
        recall_weighted     = round(recall, 6),
        f1_weighted         = round(f1, 6),
        mcc_weighted        = round(mcc, 6),
        filter_rate         = round(filter_rate, 6),
        toxic_rate          = round(toxic_rate, 6)
    )

def classification_metrics_table(
    metrics_list: list[ClassificationMetrics]
) -> dict[str, list]:
    """
    Convert list of ClassificationMetrics to dict for polars DataFrame.
    """
    return {
        "filter":    [m.name for m in metrics_list],
        "precision": [m.precision_weighted for m in metrics_list],
        "recall":    [m.recall_weighted for m in metrics_list],
        "F1":        [m.f1_weighted for m in metrics_list],
        "MCC":       [m.mcc_weighted for m in metrics_list],
        "filter_%":  [round(m.filter_rate * 100, 2) for m in metrics_list],
        "toxic_%":   [round(m.toxic_rate * 100, 2) for m in metrics_list]
    }


# Score-efficiency metrics.
def score_efficiency(
    score_bps:      float,
    filter_rate:    float
) -> float:
    """
    Score per percentage point of turnover lost.

    Higher = more efficient filter (gets more bps improvement per 1% filtered).
    """
    if filter_rate <= 0: return 0.0

    return score_bps / (filter_rate * 100)


# Correlation analysis: ML-quality vs Score.
def compute_quality_score_correlation(
    ml_metric:  np.ndarray,
    scores:     np.ndarray
) -> dict:
    """
    Compute Pearson and Spearman correlation between an ML metric
    and economic Score across multiple filters.

    Returns dict with pearson_r, spearman_rho, and n_filters.
    """

    valid   = ~(np.isnan(ml_metric) | np.isnan(scores))
    ml_v    = ml_metric[valid]
    sc_v    = scores[valid]

    n = ml_v.size
    if n < 3:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan"), "n_filters": n}

    pearson_r, pearson_p        = pearsonr(ml_v, sc_v)
    spearman_rho, spearman_p    = spearmanr(ml_v, sc_v)

    return {
        "pearson_r":    round(pearson_r, 4),
        "pearson_p":    round(pearson_p, 6),
        "spearman_rho": round(spearman_rho, 4),
        "spearman_p":   round(spearman_p, 6),
        "n_filters":    n
    }

