# Task 2.1: Market Classes for BTC Maker Markout

This folder contains the solution for Task 2.1.

The goal is to find interpretable market classes that separate Binance BTC perpetual futures maker trades by expected markout and turnover.

The analysis uses approximately 6 months of data:

* train period: November 2025 to January 2026
* validation period: February 2026 to April 2026

The main markout horizons are:

* 30 seconds
* 120 seconds
* 300 seconds

## Files

### `Markout_Preparation_BTC.ipynb`

Prepares the BTC maker trade dataset for markout analysis.

Main steps:

* load and clean raw trade data
* calculate maker markouts at different horizons
* add rebate-adjusted PnL
* prepare features used in later notebooks
* save the prepared dataset for analysis

### `Baseline_Analysis_BTC.ipynb`

Computes the unconditional baseline for BTC maker trades.

Main steps:

* calculate weighted average maker PnL
* compare markout at 30s, 120s, and 300s
* estimate the baseline adverse selection level
* check turnover and basic sample statistics

Main baseline result:

* 30s: -0.105 bps
* 120s: -0.037 bps
* 300s: +0.009 bps

The maker loses on short horizons and approximately breaks even at 300 seconds including the rebate.

### `Market_Classes_BTC.ipynb`

Builds and evaluates interpretable market classes.

Main steps:

* create feature buckets
* compare bucket markouts
* measure turnover by bucket
* run KS-tests
* analyze non-linear and monotonic feature effects
* compare train and validation behavior

The strongest features are:

* distance to mid
* VPIN
* order book imbalance
* liquidation flow
* trade flow
* hour of day

The results show that several features separate profitable and toxic maker states.

### `Executive Summary.ipynb`

Contains the final summary of the task.

It includes:

* objective
* unconditional baseline
* main findings
* stability checks
* practical implications
* limitations
* references

## Main findings

The strongest single feature is distance to mid. Trades far below mid have positive maker markout, while most other buckets are negative or close to zero.

VPIN gives the cleanest monotonic toxicity signal. Low-VPIN buckets have positive markout, while high-VPIN buckets are toxic for the maker.

Order book imbalance is also useful. Bid-heavy states tend to have better maker markout, especially at longer horizons.

Liquidation flow adds information that is not fully duplicated by price-based features. High liquidation activity creates a clear distributional shift in markout.

Spread and volatility have non-linear effects. Moderate levels are often the most toxic, while extreme levels can become positive due to post-shock reversion.

Hour of day matters. Maker adverse selection is time-dependent and should be included as a model feature.

## Practical notes

The final filter should use tree-based models, because several important relationships are non-linear.

Recommended core features:

* distance to mid
* VPIN
* liquidation flow
* trade flow
* order book imbalance
* hour of day

Microprice deviation was not useful as a separate feature in this dataset, because with only best bid and best ask it becomes mathematically equivalent to order book imbalance.

## Limitations

ETH was not tested in this task.

Only best bid and best ask data were available. Deeper order book features could not be used.

Some quantile boundaries were computed on the full sample. In production, these boundaries should be estimated online.

The validation period contains one major stress episode, but it may not represent all possible market regimes.

## References

* Cont, Kukanov, Stoikov, 2014. The Price Impact of Order Book Events.
* Easley, Lopez de Prado, O'Hara, 2012. Flow Toxicity and Liquidity in a High-Frequency World.
* Stoikov, 2018. The Micro-Price: A High-Frequency Estimator of Future Prices.
* Amihud, 2002. Illiquidity and Stock Returns.
* Lopez de Prado, 2018. Advances in Financial Machine Learning.
