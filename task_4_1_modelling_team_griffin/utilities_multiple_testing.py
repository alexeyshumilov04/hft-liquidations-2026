# imports.
import numpy as np
from dataclasses import dataclass
from utilities_validation import newey_west_variance
from scipy.stats import norm


# Pairwise DM matrix.
def pairwise_dm_matrix(
    daily_scores: dict[str, np.ndarray]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Compute all-pairs DM-HAC test statistics and p-values.

    Returns:
        names:    list of filter names.
        dm_stats: (N, N) matrix of DM statistics. dm_stats[i,j] > 0 means i > j.
        p_values: (N, N) matrix of one-sided p-values (H1: i better than j).
    """

    names = list(daily_scores.keys())
    N     = len(names)

    dm_stats = np.zeros((N, N))
    p_values = np.ones((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d     = daily_scores[names[i]] - daily_scores[names[j]]
            d_bar = d.mean()
            var_d = newey_west_variance(d)

            if var_d > 0:
                stat = d_bar / np.sqrt(var_d)
                pval = 1.0 - norm.cdf(stat)
            else:
                stat = 0.0
                pval = 0.5

            dm_stats[i, j] = stat
            p_values[i, j] = pval

    return names, dm_stats, p_values


# Hansen SPA test (Superior Predictive Ability).
@dataclass
class SPAResult:
    """
    Result of Hansen's SPA test.
    """
    test_stat:      float
    p_value:        float        # consistent p-value.
    reject_h0:      bool         # at alpha level.
    best_filter:    str
    best_mean_diff: float

def hansen_spa_test(
    daily_scores:   dict[str, np.ndarray],
    benchmark_name: str,
    n_boot:         int = 5000,
    alpha:          float = 0.05,
    seed:           int = 42
) -> SPAResult:
    """
    Hansen (2005) Superior Predictive Ability test.

    H0: no filter is better than the benchmark.
    H1: at least one filter is genuinely better.

    Uses stationary bootstrap of the loss differential matrix.

    If p < alpha -> reject H0 -> the best filter is genuinely superior,
    not just artifact.
    """
    rng   = np.random.default_rng(seed)
    names = [n for n in daily_scores if n != benchmark_name]
    K     = len(names)
    T     = daily_scores[benchmark_name].size

    if K == 0: return SPAResult(0.0, 1.0, False, benchmark_name, 0.0)

    # loss differentials: d_k,t = Score_k,t - Score_bench,t.
    D = np.column_stack([
        daily_scores[n] - daily_scores[benchmark_name]
        for n in names
    ])  # (T, K)

    # observed test statistic: max of studentized means.
    d_bar = D.mean(axis=0)                   # (K,)
    d_var = np.array([newey_west_variance(D[:, k]) for k in range(K)])
    d_std = np.sqrt(np.maximum(d_var, 1e-20))

    t_obs = np.max(d_bar / d_std)

    best_idx  = int(np.argmax(d_bar / d_std))
    best_name = names[best_idx]
    best_mean = float(d_bar[best_idx])

    # bootstrap: stationary bootstrap with geometric block length.
    mean_block = max(int(T ** (1/3)), 2)
    p_geom     = 1.0 / mean_block

    boot_stats = np.empty(n_boot)

    for b in range(n_boot):
        # generate bootstrap indices (stationary bootstrap).
        idx = np.empty(T, dtype=np.int64)
        pos = rng.integers(0, T)
        for t in range(T):
            idx[t] = pos % T
            if rng.random() < p_geom:
                pos = rng.integers(0, T)
            else:
                pos += 1

        D_boot    = D[idx]
        d_bar_b   = D_boot.mean(axis=0)

        # center under H0: d_bar_b - d_bar (re-centered).
        d_bar_centered = d_bar_b - d_bar

        # studentize with original variance estimate.
        t_boot = np.max(d_bar_centered / d_std)
        boot_stats[b] = t_boot

    p_value = float(np.mean(boot_stats >= t_obs))

    return SPAResult(
        test_stat       = round(t_obs, 4),
        p_value         = round(p_value, 4),
        reject_h0       = (p_value < alpha),
        best_filter     = best_name,
        best_mean_diff  = round(best_mean, 6)
    )


# Model Confidence Set (MCS).
@dataclass
class MCSResult:
    """
    Result of Model Confidence Set procedure.
    """
    confidence_set:     list[str]           # filters in the MCS (indistinguishable from best).
    eliminated:         list[str]           # filters removed (significantly worse).
    elimination_pvals:  dict[str, float]    # p-value at which each was eliminated.
    alpha:              float

def model_confidence_set(
    daily_scores:   dict[str, np.ndarray],
    n_boot:         int = 5000,
    alpha:          float = 0.05,
    seed:           int = 42
) -> MCSResult:
    """
    Model Confidence Set (Hansen, Lunde, Nason 2011).

    Iteratively eliminates the worst model until remaining models
    are statistically indistinguishable at level alpha.

    Uses the T_max statistic (maximum of pairwise t-statistics)
    and stationary bootstrap for p-values.
    """
    rng         = np.random.default_rng(seed)
    remaining   = list(daily_scores.keys())
    eliminated  = []
    elim_pvals  = {}

    while len(remaining) > 1:
        K = len(remaining)
        T = daily_scores[remaining[0]].size

        # compute pairwise loss differentials.
        D = {}
        for i in range(K):
            for j in range(i + 1, K):
                d = daily_scores[remaining[i]] - daily_scores[remaining[j]]
                D[(i, j)] = d

        # observed T_max: find the model with worst relative performance.
        mean_scores = np.array([daily_scores[n].mean() for n in remaining])
        worst_idx   = int(np.argmin(mean_scores))
        worst_name  = remaining[worst_idx]

        # t-statistics of worst vs each other.
        t_stats = []
        for j in range(K):
            if j == worst_idx:
                continue
            d = daily_scores[remaining[j]] - daily_scores[worst_name]
            d_bar = d.mean()
            var_d = newey_west_variance(d)
            t = d_bar / np.sqrt(max(var_d, 1e-20))
            t_stats.append(t)

        t_max_obs = max(t_stats) if t_stats else 0.0

        # bootstrap p-value for elimination.
        mean_block = max(int(T ** (1/3)), 2)
        p_geom     = 1.0 / mean_block

        boot_t_max = np.empty(n_boot)
        for b in range(n_boot):
            idx = np.empty(T, dtype=np.int64)
            pos = rng.integers(0, T)
            for t_i in range(T):
                idx[t_i] = pos % T
                if rng.random() < p_geom:
                    pos = rng.integers(0, T)
                else:
                    pos += 1

            # bootstrap t-stats for worst vs each other.
            t_stats_b = []
            for j in range(K):
                if j == worst_idx:
                    continue
                d       = daily_scores[remaining[j]] - daily_scores[worst_name]
                d_boot  = d[idx]
                d_bar_b = d_boot.mean() - d.mean()     # centered under H0.
                var_d   = newey_west_variance(d)
                t_b     = d_bar_b / np.sqrt(max(var_d, 1e-20))
                t_stats_b.append(t_b)

            boot_t_max[b] = max(t_stats_b) if t_stats_b else 0.0

        p_elim = float(np.mean(boot_t_max >= t_max_obs))

        if p_elim < alpha:
            # eliminate worst.
            remaining.remove(worst_name)
            eliminated.append(worst_name)
            elim_pvals[worst_name] = round(p_elim, 4)
        else:
            # cannot eliminate -> remaining set is the MCS.
            break

    return MCSResult(
        confidence_set      = remaining,
        eliminated          = eliminated,
        elimination_pvals   = elim_pvals,
        alpha               = alpha
    )


# Romano-Wolf stepdown.
@dataclass
class RomanoWolfResult:
    """
    Result of Romano-Wolf stepdown procedure.
    """
    rejected:           list[str]       # hypotheses rejected (filter is better than bench).
    not_rejected:       list[str]
    adjusted_pvalues:   dict[str, float]

def romano_wolf_stepdown(
    daily_scores:   dict[str, np.ndarray],
    benchmark_name: str,
    n_boot:         int = 5000,
    alpha:          float = 0.05,
    seed:           int = 42
) -> RomanoWolfResult:
    """
    Romano-Wolf (2005, 2016) stepdown procedure for FWER control.

    Tests H0_k: E[Score_k - Score_bench] <= 0 for each candidate k.
    Controls FWER at level alpha by accounting for the joint distribution
    of test statistics (correlation between tests).

    Steps:
      1. Compute all t-stats.
      2. Bootstrap(stationary) the joint distribution under H0.
      3. Compare the largest observed t-stat against the bootstrap max.
      4. If rejected, remove that hypothesis and repeat.
    """
    rng   = np.random.default_rng(seed)
    names = [n for n in daily_scores if n != benchmark_name]
    K     = len(names)
    T     = daily_scores[benchmark_name].size

    if K == 0:
        return RomanoWolfResult([], [], {})

    # loss differentials.
    D = np.column_stack([
        daily_scores[n] - daily_scores[benchmark_name]
        for n in names
    ])

    d_bar = D.mean(axis=0)
    d_var = np.array([newey_west_variance(D[:, k]) for k in range(K)])
    d_std = np.sqrt(np.maximum(d_var, 1e-20))
    t_obs = d_bar / d_std

    # bootstrap joint distribution (stationary bootstrap).
    mean_block = max(int(T ** (1/3)), 2)
    p_geom     = 1.0 / mean_block

    boot_t = np.empty((n_boot, K))
    for b in range(n_boot):
        idx = np.empty(T, dtype=np.int64)
        pos = rng.integers(0, T)
        for t_i in range(T):
            idx[t_i] = pos % T
            if rng.random() < p_geom:
                pos = rng.integers(0, T)
            else:
                pos += 1

        D_boot    = D[idx]
        d_bar_b   = D_boot.mean(axis=0) - d_bar    # centered.
        boot_t[b] = d_bar_b / d_std

    # stepdown.
    active    = set(range(K))
    rejected  = []
    adj_pvals = {}

    while active:
        active_list = sorted(active)

        # max observed t among active.
        max_t_obs = max(t_obs[k] for k in active_list)
        max_k     = max(active_list, key=lambda k: t_obs[k])

        # bootstrap max among active.
        boot_max = np.array([
            max(boot_t[b, k] for k in active_list)
            for b in range(n_boot)
        ])

        p_adj = float(np.mean(boot_max >= max_t_obs))

        if p_adj < alpha:
            rejected.append(names[max_k])
            adj_pvals[names[max_k]] = round(p_adj, 4)
            active.remove(max_k)
        else:
            # cannot reject any more.
            for k in active:
                adj_pvals[names[k]] = round(p_adj, 4)
            break

    not_rejected = [names[k] for k in range(K) if names[k] not in rejected]

    return RomanoWolfResult(
        rejected            = rejected,
        not_rejected        = not_rejected,
        adjusted_pvalues    = adj_pvals
    )


