# imports.
import numpy as np
from dataclasses import dataclass
from itertools import combinations
from scipy.stats import norm


# Walk-forward and rolling - window backtesting.
@ dataclass
class FoldResult:
    """
    Result of one train/test fold.
    """
    fold_idx:       int
    train_days:     np.ndarray       # day indices used for training.
    test_days:      np.ndarray       # day indices used for testing.
    train_score:    float            # Score on train.
    test_score:     float            # Score on test.
    best_threshold: float | None     # threshold found on train.

def walk_forward_splits(
    n_days:     int,
    train_size: int,
    test_size:  int,
    step_size:  int | None = None    
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward (expanding or sliding) train/test splits.

    Returns list of (train_indices, test_indices) where indices are day positions [0, n_days).

    step_size defaults to test_size (non-overlapping test windows).
    """

    if step_size is None:
        step_size = test_size

    splits = []
    start = 0
    while (start + train_size + test_size) <= n_days:
        train_idx = np.arange(start, start + train_size)
        test_idx  = np.arange(start + train_size, start + train_size + test_size)
        splits.append((train_idx, test_idx))
        start += step_size

    return splits

def rolling_window_splits(
    n_days:     int,
    train_size: int,
    test_size:  int,
    step_size:  int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Rolling (fixed-size) window splits.
    Same as walk_forward but train window doesn't expand.
    """
    return walk_forward_splits(n_days, train_size, test_size, step_size)

def expanding_window_splits(
    n_days:        int,
    min_train_size: int,
    test_size:     int,
    step_size:     int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding window: train grows over time, test slides forward.
    """
    if step_size is None:
        step_size = test_size

    splits = []
    test_start = min_train_size
    while test_start + test_size <= n_days:
        train_idx = np.arange(0, test_start)
        test_idx  = np.arange(test_start, test_start + test_size)
        splits.append((train_idx, test_idx))
        test_start += step_size

    return splits


# Purged + embargoed CV / CPCV.
def purged_kfold_splits(
    n_days:         int,
    n_folds:        int = 5,
    purge_days:     int = 1,
    embargo_days:   int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    K-fold CV with purge and embargo on daily data.

    For each fold:
        - test = contiguous block of days.
        - train = all other days EXCEPT:
            - purge_days before test start (markout from train leaks into test).
            - embargo_days after test end (symmetric protection).
    
    Returns list of (train_indices, test_indices).
    """

    fold_size = n_days // n_folds
    all_days = np.arange(n_days)

    splits = []
    for k in range(n_folds):
        test_start  = k * fold_size
        test_end    = (k + 1) * fold_size if k < n_folds - 1 else n_days
        test_idx    = np.arange(test_start, test_end)

        # purge: remove days just before test.
        purge_start = max(0, test_start - purge_days)

        # embargo: remove days just after test.
        embargo_end = min(n_days, test_end + embargo_days)

        excluded = set(range(purge_start, embargo_end))
        train_idx = np.array([d for d in all_days if d not in excluded])

        splits.append((train_idx, test_idx))

    return splits

def cpcv_splits(
    n_days:       int,
    n_groups:     int = 6,
    n_test_groups: int = 1,
    purge_days:   int = 1,
    embargo_days: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Split days into n_groups contiguous blocks. For each combination of n_test_groups blocks as test, 
    the rest (minus purge/embargo) is train.

    Number of combinations: C(n_groups, n_test_groups).

    Returns list of (train_indices, test_indices).
    """

    group_size = n_days // n_groups
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end   = (g + 1) * group_size if g < n_groups - 1 else n_days
        groups.append(np.arange(start, end))

    splits = []
    for test_combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in test_combo])

        # purge + embargo around each test group.
        excluded = set()
        for g in test_combo:
            g_start = groups[g].min()
            g_end   = groups[g].max() + 1
            for d in range(max(0, g_start - purge_days), min(n_days, g_end + embargo_days)):
                excluded.add(d)

        train_idx = np.array([
            d for d in range(n_days)
            if d not in excluded
        ])

        splits.append((train_idx, test_idx))

    return splits


# Statistical tests.
@dataclass
class TestResult:
    """
    Result of a statistical test.
    """
    test_name:  str
    statistic:  float
    p_value:    float
    reject_h0:  bool       # at 5%.
    ci_low:     float | None = None
    ci_high:    float | None = None

def newey_west_variance(
    x:          np.ndarray,
    max_lag:    int | None = None
) -> float:
    """
    Newey-West HAC variance estimator for the mean of x.

    Accounts for autocorrelation up to max_lag.

    Default max_lag: floor(4 * (T/100)^(2/9)) - Andrews (1991).
    """

    T = x.size
    if (T < 2) : return 0.0

    if max_lag is None:
        max_lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
        max_lag = max(max_lag, 1)

    x_dm = x - x.mean()

    # gamma(0).
    gamma_0 = np.sum(x_dm ** 2) / T

    # weighted autocovariances.
    gamma_sum = 0.0
    for h in range(1, max_lag + 1):
        gamma_h = np.sum(x_dm[h:] * x_dm[:-h]) / T
        weight  = 1.0 - h / (max_lag + 1)     # Bartlett kernel.
        gamma_sum += 2 * weight * gamma_h

    # variance of mean.
    return (gamma_0 + gamma_sum) / T

def diebold_mariano_test(
    d:          np.ndarray,
    max_lag:    int | None = None,
    alpha:      float = 0.05
) -> TestResult: 
    """
    Diebold-Mariano test with Newey-West HAC variance.

    Input: d = array of daily differences (Score_candidate - Score_baseline).
    H0: E[d] = 0  (candidate is no better than baseline).
    H1: E[d] > 0  (one-sided: candidate is better).

    Returns TestResult with DM statistic, p-value, and rejection decision.
    """

    T = d.size
    if T < 3: return TestResult("DM-HAC", 0.0, 1.0, False)

    d_bar = d.mean()
    var_d = newey_west_variance(d, max_lag)

    if var_d <= 0: return TestResult("DM-HAC", 0.0, 1.0, False)

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 1.0 - norm.cdf(dm_stat)       # one-sided.

    # confidence interval for mean difference.
    z_crit  = norm.ppf(1 - alpha / 2)
    se      = np.sqrt(var_d)
    ci_low  = d_bar - z_crit * se
    ci_high = d_bar + z_crit * se

    return TestResult(
        test_name   = "DM-HAC",
        statistic   = round(dm_stat, 4),
        p_value     = round(p_value, 6),
        reject_h0   = (p_value < alpha),
        ci_low      = round(ci_low, 6),
        ci_high     = round(ci_high, 6)
    )

def paired_permutation_test(
    d:           np.ndarray,
    n_perms:     int = 10_000,
    alpha:       float = 0.05,
    seed:        int = 42
) -> TestResult:
    """
    Non-parametric permutation test for paired differences.

    Under H0, d has symmetric distribution around 0 ->
    randomly flipping signs should not change the mean.
    H1: E[d] > 0 (one-sided).
    """
    rng = np.random.default_rng(seed)
    T   = d.size

    if T < 3: return TestResult("Permutation", 0.0, 1.0, False)

    observed = d.mean()

    # permutation distribution.
    signs       = rng.choice([-1, 1], size=(n_perms, T))
    perm_means  = (signs * d).mean(axis=1)

    p_value = np.mean(perm_means >= observed)

    return TestResult(
        test_name   = "Permutation",
        statistic   = round(observed, 6),
        p_value     = round(p_value, 6),
        reject_h0   = (p_value < alpha)
    )


# Bootstrap Methods.
@dataclass
class BootstrapResult:
    """
    Bootstrap confidence interval result.
    """
    method:     str
    mean:       float
    ci_low:     float
    ci_high:    float
    ci_width:   float
    std:        float
    n_boot:     int

def iid_bootstrap(
    x:       np.ndarray,
    n_boot:  int = 5000,
    alpha:   float = 0.05,
    seed:    int = 42
) -> BootstrapResult:
    """
    Naive IID bootstrap. Resamples with replacement, ignoring temporal structure.
    Included to demonstrate underestimation of variance.
    """
    rng = np.random.default_rng(seed)
    T   = x.size

    means = np.array([
        x[rng.integers(0, T, T)].mean()
        for _ in range(n_boot)
    ])

    return _boot_result(means, "IID", n_boot, alpha)

def moving_block_bootstrap(
    x:           np.ndarray,
    block_len:   int,
    n_boot:      int = 5000,
    alpha:       float = 0.05,
    seed:        int = 42
) -> BootstrapResult:
    """
    Moving Block Bootstrap (Kunsch 1989, Liu & Singh 1992).

    Resamples contiguous blocks of length `block_len` with replacement.
    Preserves within-block dependence structure.
    """
    rng = np.random.default_rng(seed)
    T   = x.size
    n_blocks = int(np.ceil(T / block_len))

    means = np.empty(n_boot)
    for b in range(n_boot):
        # sample n_blocks block-start positions.
        starts = rng.integers(0, T - block_len + 1, n_blocks)
        # concatenate blocks.
        sample = np.concatenate([x[s:s + block_len] for s in starts])[:T]
        means[b] = sample.mean()

    return _boot_result(means, f"MBB(L={block_len})", n_boot, alpha)

def circular_block_bootstrap(
    x:           np.ndarray,
    block_len:   int,
    n_boot:      int = 5000,
    alpha:       float = 0.05,
    seed:        int = 42
) -> BootstrapResult:
    """
    Circular Block Bootstrap (Politis & Romano 1992).

    Like MBB but wraps around: x is treated as circular.
    Eliminates edge bias.
    """
    rng = np.random.default_rng(seed)
    T   = x.size
    x_circ = np.tile(x, 2)     # duplicate for wrap-around.
    n_blocks = int(np.ceil(T / block_len))

    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        sample = np.concatenate([x_circ[s:s + block_len] for s in starts])[:T]
        means[b] = sample.mean()

    return _boot_result(means, f"CBB(L={block_len})", n_boot, alpha)

def stationary_bootstrap(
    x:              np.ndarray,
    mean_block_len: int,
    n_boot:         int = 5000,
    alpha:          float = 0.05,
    seed:           int = 42
) -> BootstrapResult:
    """
    Stationary Bootstrap (Politis & Romano 1994).

    Block length is random (geometric distribution with given mean).
    Produces stationary bootstrap samples.
    """
    rng = np.random.default_rng(seed)
    T   = x.size
    p   = 1.0 / mean_block_len          # geometric parameter.

    means = np.empty(n_boot)
    for b in range(n_boot):
        sample = np.empty(T)
        pos    = rng.integers(0, T)      # random start.
        for i in range(T):
            sample[i] = x[pos % T]
            # with probability p, jump to a new random position.
            if rng.random() < p:
                pos = rng.integers(0, T)
            else:
                pos += 1
        means[b] = sample.mean()

    return _boot_result(means, f"Stationary(L={mean_block_len})", n_boot, alpha)

def bayesian_bootstrap(
    x:       np.ndarray,
    n_boot:  int = 5000,
    alpha:   float = 0.05,
    seed:    int = 42
) -> BootstrapResult:
    """
    Bayesian Bootstrap (Rubin 1981).

    Instead of resampling, draws Dirichlet weights.
    Smoother estimates, faster, no degenerate samples.
    """
    rng = np.random.default_rng(seed)
    T   = len(x)

    means = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.dirichlet(np.ones(T))
        means[b] = np.dot(w, x)

    return _boot_result(means, "Bayesian", n_boot, alpha) 
    
def bayesian_block_bootstrap(
    x:           np.ndarray,
    block_len:   int,
    n_boot:      int = 5000,
    alpha:       float = 0.05,
    seed:        int = 42,
) -> BootstrapResult:
    """
    Bayesian Block Bootstrap - Dirichlet weights on blocks, not observations.

    1. Split x into blocks of `block_len`.
    2. Draw Dirichlet weights over blocks.
    3. Compute weighted mean (block weight shared by all obs in block).

    Preserves temporal dependence (block structure) +
    smooth weights (Bayesian) + no degenerate samples.
    """
    rng = np.random.default_rng(seed)
    T   = x.size

    # split into blocks.
    n_blocks = int(np.ceil(T / block_len))
    block_sums   = np.empty(n_blocks)
    block_counts = np.empty(n_blocks)
    for i in range(n_blocks):
        s = i * block_len
        e = min(s + block_len, T)
        block_sums[i]   = x[s:e].sum()
        block_counts[i] = e - s

    means = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.dirichlet(np.ones(n_blocks))
        # weighted mean: each block contributes w_k * (sum_k / count_k) * count_k
        # = w_k * sum_k, normalized by total weighted count.
        means[b] = np.dot(w, block_sums) / np.dot(w, block_counts)

    return _boot_result(means, f"BayesBlock(L={block_len})", n_boot, alpha)

def _boot_result(
    means:  np.ndarray,
    method: str,
    n_boot: int,
    alpha:  float
) -> BootstrapResult:
    """
    Construct BootstrapResult from array of bootstrap means.
    """
    ci_low  = np.percentile(means, 100 * alpha / 2)
    ci_high = np.percentile(means, 100 * (1 - alpha / 2))

    return BootstrapResult(
        method      = method,
        mean        = round(float(means.mean()), 6),
        ci_low      = round(float(ci_low), 6),
        ci_high     = round(float(ci_high), 6),
        ci_width    = round(float(ci_high - ci_low), 6),
        std         = round(float(means.std()), 6),
        n_boot      = n_boot
    )

def compare_bootstraps(
    results: list[BootstrapResult]
) -> list[dict]:
    """
    Format bootstrap comparison as list of dicts (for DataFrame).
    """
    return [
        {
            "method":   r.method,
            "mean":     r.mean,
            "CI_low":   r.ci_low,
            "CI_high":  r.ci_high,
            "CI_width": r.ci_width,
            "std":      r.std
        }
        for r in results
    ]


# Autocorrelation helpers.
def acf(
    x:        np.ndarray,
    max_lag:  int = 20
) -> np.ndarray:
    """
    Sample autocorrelation function.

    Returns array of length max_lag + 1 (lag 0 = 1.0).
    """
    x_dm  = x - x.mean()
    var   = np.sum(x_dm ** 2)

    if var == 0: return np.ones(max_lag + 1)

    result = np.empty(max_lag + 1)
    for h in range(max_lag + 1):
        if h == 0:
            result[h] = 1.0
        else:
            result[h] = np.sum(x_dm[h:] * x_dm[:-h]) / var

    return result

def optimal_block_length(
    x: np.ndarray,
) -> int:
    """
    Heuristic for optimal block length.

    Uses the rule: L ~= ceil(T^(1/3) * (2 * rho / (1 - rho^2))^(2/3)) where rho = ACF(1). 
    Falls back to ceil(T^(1/3)) if rho ~= 0.

    Minimum: 1. Maximum: T // 3.
    """
    T = x.size
    if T < 5: return 1

    rho = acf(x, max_lag=1)[1]
    rho = np.clip(rho, -0.99, 0.99)

    if abs(rho) < 0.05:
        L = int(np.ceil(T ** (1 / 3)))
    else:
        factor = (2 * abs(rho) / (1 - rho ** 2)) ** (2 / 3)
        L = int(np.ceil(T ** (1 / 3) * factor))

    L = max(1, min(L, T // 3))
    return L

    
    