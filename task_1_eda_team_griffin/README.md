# Task 1: EDA for Liquidations Data

This folder contains the EDA part of the Griffin team work.

The analysis is based on BTC and ETH perpetual futures data from Binance and Bybit. The data covers December 2025 to February 2026.

The goal is to understand the structure of trades, order book data, and liquidations, then check whether these variables can be useful for a liquidation risk signal.

## Files

### `Foundational EDA. Part 1..ipynb`

Data quality checks.

Main parts:

* missing values
* duplicate rows
* timestamp monotonicity
* time coverage
* basic consistency checks

The main result is that the data is clean enough for further analysis. There are no critical missing value or timestamp problems.

### `Foundational_EDA. Part 2..ipynb`

Basic univariate analysis.

Main parts:

* trades analysis
* trade size distributions
* buy and sell balance
* LOB price analysis
* spread analysis
* liquidation activity by exchange

This notebook gives the first overview of market activity and shows that liquidation volume is highly clustered.

### `Intermediate EDA..ipynb`

Feature analysis.

Main parts:

* feature construction
* stationarity checks
* autocorrelation
* outlier structure
* non-linearities
* market regimes

The main result is that returns have weak autocorrelation, while volatility, volume, spread, and liquidation variables have heavy tails and regime-dependent behavior.

### `Advanced EDA. Part 1..ipynb`

Heatmap and first ML baseline.

Main parts:

* liquidation heatmap
* feature table for ML
* XGBoost regression for future liquidation volume
* XGBoost classification for future large liquidation events
* PR-AUC and threshold analysis

The heatmap shows that liquidations cluster in time and price. The classifier is useful mainly as an early-warning ranking model.

### `Advanced EDA. Part 2..ipynb`

Model stability and interpretation.

Main parts:

* Monte Carlo subsampling
* walk-forward Monte Carlo validation
* calibration curves
* permutation importance

The goal is to check that the model result is stable, not just a lucky train/test split.

### `Executive Summary.ipynb`

Short final summary of the whole EDA task.

It contains:

* objective
* key findings by EDA tier
* practical recommendations
* limitations
* references

## Main findings

The data is mostly clean and can be used for modeling.

Bybit has more liquidation events than Binance, but dollar liquidation volume is comparable. For exchange comparison, dollar volume is more useful than event count.

Liquidation volume is highly clustered. Large liquidation events tend to appear in bursts, which supports the idea of liquidation cascades.

Returns have near-zero autocorrelation at the minute level. This is expected for a liquid market.

Volatility, spread, and trading volume are strongly related. Liquidation features form a separate block and add information that is not fully captured by price volatility.

Most raw features have heavy tails. For production use, online normalization such as EMA z-score is needed.

The relationship between market features and future stress is non-linear. Tree-based models are more suitable than linear models.

The ML classifier for large future liquidations has useful predictive power. For BTC, PR-AUC is around 0.35. For ETH, PR-AUC is around 0.27.

Monte Carlo subsampling shows low PR-AUC variance across random train subsamples. The signal is stable with respect to train sampling.

Walk-forward validation shows that model quality changes across time. The signal is stronger in stressed regimes and weaker in calm periods.

Calibration curves are close to the diagonal, so ensemble probabilities can be used as approximate risk scores.

Permutation importance shows that the main predictors are trading volume and liquidation features.

## Practical recommendations

The core signal should use:

* liquidation volume
* liquidation imbalance
* rolling liquidation sums over 15 to 60 minutes
* trading volume
* spread-based liquidity features
* Amihud illiquidity

The model output should be used as a continuous liquidation risk score, not only as a binary prediction.

Thresholds should depend on the cost of false positives and missed liquidation events.

Walk-forward validation is important because the signal is regime-dependent.

## Limitations

The analysis uses minute-level aggregation. Some liquidation cascades can happen faster than one minute.

There is no full order book depth. Because of that, slippage and price impact features are only approximate.

The sample contains one major stress period, but may not cover all possible market regimes.

The analysis covers BTC and ETH only. Generalization to smaller altcoins was not tested.

## References

* Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
* Sepper, O. (2026). Slippage-at-Risk: A Forward-Looking Liquidity Risk Framework for Perpetual Futures Exchanges.
