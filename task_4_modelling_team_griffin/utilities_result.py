import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# pnl metrics.
def plot_test_metrics_boxplots(
    result_table,
    model_name: str = "Model",
    rate_col: str = "max_filter_rate",
    instrument: str = "BTC",
    tau: int = 30,
    turnover_constraint: float = 500_000
):
    """
    Plot 2x2 boxplots across walk-forward windows for key test metrics.

    Expected columns:
        rate_col
        test_score
        test_pnl_kept
        test_pnl_filtered
        test_kept_usd/day
        test_filter_rate (%)
    """

    # Polars -> pandas if needed.
    if hasattr(result_table, "to_pandas"):
        pdf = result_table.to_pandas()
    else:
        pdf = result_table.copy()

    pdf = pdf.copy()

    # If nothing was filtered, test_pnl_filtered=0 is only a technical placeholder.
    # Do not use it in distribution plots.
    if "test_filter_rate (%)" in pdf.columns:
        pdf["test_pnl_filtered_clean"] = np.where(
            pdf["test_filter_rate (%)"] > 0,
            pdf["test_pnl_filtered"],
            np.nan
        )
    else:
        pdf["test_pnl_filtered_clean"] = pdf["test_pnl_filtered"]

    # Convert kept USD/day to millions for readability.
    pdf["test_kept_usd/day_mln"] = pdf["test_kept_usd/day"] / 1_000_000

    rates = sorted(pdf[rate_col].unique())

    metrics = [
        ("test_score", "test_score", 0.0),
        ("test_pnl_kept", "test_pnl_kept", 0.0),
        ("test_pnl_filtered_clean", "test_pnl_filtered", 0.0),
        ("test_kept_usd/day_mln", "test_kept_usd/day, mln", turnover_constraint / 1_000_000)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()

    for ax, (col, ylabel, ref_line) in zip(axes, metrics):
        box_data = [
            pdf.loc[pdf[rate_col] == rate, col].dropna().values
            for rate in rates
        ]

        ax.boxplot(box_data, tick_labels=[str(rate) for rate in rates], showmeans=True)

        ax.axhline(ref_line, linestyle="--", linewidth=1)

        ax.set_xlabel(rate_col)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} across W1-W5")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{instrument}. {model_name}: horizon = {tau}", fontsize=14)
    plt.tight_layout()
    plt.show()


# feature importance for stack of lgbm models.
def plot_lgbm_feature_importance(
    models: list,
    top_n: int = 30,
    importance_type: str = "gain",
    interval: str = "std",
    normalize_per_model: bool = True,
    figsize: tuple = (9, 7),
    title: str = "LightGBM feature importance across WF windows",
):
    """
    Plot averaged LightGBM feature importance across saved WF models.

    Parameters:
        models:
            List of saved models from run_walk_forward.
            Expected format:
                [{"window": ..., "model": ..., "features": [...]}, ...]

        top_n:
            Number of top features to plot.

        importance_type:
            "gain"  - total gain from splits.
            "split" - number of splits.

        interval:
            "std"  - mean +/- std across WF windows.
            "sem"  - mean +/- standard error.
            "ci95" - mean +/- 1.96 * standard error.

        normalize_per_model:
            If True, feature importance inside each WF model is normalized to sum=1.
            This makes importances comparable across windows.

        figsize:
            Figure size.

    Returns:
        pandas.DataFrame with aggregated feature importance.
    """

    rows = []

    for k, item in enumerate(models):

        # Case 1: item is dict from run_walk_forward.
        if isinstance(item, dict):
            model = item["model"]
            window = item.get("window", f"model_{k}")
            features = item.get("features", None)
        else:
            model = item
            window = f"model_{k}"
            features = None

        # Case 1: sklearn-style LGBMRegressor.
        if hasattr(model, "booster_"):
            booster = model.booster_

        # Case 2: native lightgbm Booster.
        elif hasattr(model, "feature_importance"):
            booster = model

        # Case 3: model bundle/dict.
        elif isinstance(model, dict):
            inner_model = model["model"]

            if hasattr(inner_model, "booster_"):
                booster = inner_model.booster_
            elif hasattr(inner_model, "feature_importance"):
                booster = inner_model
            else:
                raise ValueError("Cannot find LightGBM booster inside model dict.")

        else:
            raise ValueError("Unknown model format.")

        importance = booster.feature_importance(
            importance_type=importance_type,
        ).astype(float)

        if features is None:
            features = booster.feature_name()

        if normalize_per_model:
            total = importance.sum()
            if total > 0:
                importance = importance / total

        for feature, value in zip(features, importance):
            rows.append({
                "window": window,
                "feature": feature,
                "importance": float(value),
            })

    fi = pd.DataFrame(rows)

    summary = (
        fi
        .groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            n_models=("importance", "count"),
        )
    )

    summary["std_importance"] = summary["std_importance"].fillna(0.0)
    summary["sem_importance"] = summary["std_importance"] / np.sqrt(summary["n_models"])

    if interval == "std":
        summary["interval"] = summary["std_importance"]
        interval_label = "mean ± std"
    elif interval == "sem":
        summary["interval"] = summary["sem_importance"]
        interval_label = "mean ± SEM"
    elif interval == "ci95":
        summary["interval"] = 1.96 * summary["sem_importance"]
        interval_label = "mean ± 95% CI"
    else:
        raise ValueError("interval must be one of: 'std', 'sem', 'ci95'.")

    summary = summary.sort_values("mean_importance", ascending=False)

    plot_df = (
        summary
        .head(top_n)
        .sort_values("mean_importance", ascending=True)
    )

    plt.figure(figsize=figsize)

    plt.barh(
        plot_df["feature"],
        plot_df["mean_importance"],
        xerr=plot_df["interval"],
        capsize=3,
    )

    plt.xlabel(f"Feature importance, {importance_type} ({interval_label})")
    plt.ylabel("Feature")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return summary


# feature importance for stack of linear models.
def plot_linear_coefficients(
    models: list,
    top_n: int = 30,
    interval: str = "std",
    figsize: tuple = (9, 7),
    title: str = "Linear model coefficients across WF windows",
):

    rows = []

    for k, item in enumerate(models):

        if isinstance(item, dict):
            model = item["model"]
            window = item.get("window", f"model_{k}")
            features = item.get("features", None)
        else:
            model = item
            window = f"model_{k}"
            features = None

        # Case 1: model bundle with scaler/model.
        if isinstance(model, dict):
            inner_model = model["model"]
        else:
            inner_model = model

        if not hasattr(inner_model, "coef_"):
            raise ValueError("Model does not have coef_. This function is only for linear models.")

        coef = np.asarray(inner_model.coef_).ravel()

        if features is None:
            features = [f"feature_{i}" for i in range(len(coef))]

        for feature, value in zip(features, coef):
            rows.append({
                "window": window,
                "feature": feature,
                "coef": float(value),
            })

    coef_df = pd.DataFrame(rows)

    summary = (
        coef_df
        .groupby("feature", as_index=False)
        .agg(
            mean_coef=("coef", "mean"),
            std_coef=("coef", "std"),
            n_models=("coef", "count"),
        )
    )

    summary["std_coef"] = summary["std_coef"].fillna(0.0)
    summary["sem_coef"] = summary["std_coef"] / np.sqrt(summary["n_models"])

    if interval == "std":
        summary["interval"] = summary["std_coef"]
        interval_label = "mean ± std"
    elif interval == "sem":
        summary["interval"] = summary["sem_coef"]
        interval_label = "mean ± SEM"
    elif interval == "ci95":
        summary["interval"] = 1.96 * summary["sem_coef"]
        interval_label = "mean ± 95% CI"
    else:
        raise ValueError("interval must be one of: 'std', 'sem', 'ci95'.")

    summary["abs_mean_coef"] = summary["mean_coef"].abs()

    plot_df = (
        summary
        .sort_values("abs_mean_coef", ascending=False)
        .head(top_n)
        .sort_values("mean_coef", ascending=True)
    )

    plt.figure(figsize=figsize)

    plt.barh(
        plot_df["feature"],
        plot_df["mean_coef"],
        xerr=plot_df["interval"],
        capsize=3,
    )

    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel(f"Coefficient ({interval_label})")
    plt.ylabel("Feature")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return summary.sort_values("abs_mean_coef", ascending=False)


