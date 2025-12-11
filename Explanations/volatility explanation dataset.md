# Volatility Metrics Explanation

## Overview
The dataset now includes **comprehensive daily and short-term volatility metrics** to identify stocks with the highest intraday volatility and explosive short-term price movements.

## Selection Methodology

Stocks are ranked by a **composite score** that weights four key metrics:

```
Composite Score =
  30% × Annualized Mean Absolute Daily Return +
  20% × Overall Annualized Volatility +
  25% × Average 10-day Rolling Volatility +
  25% × Peak 20-day Rolling Volatility
```

This approach captures both:
- **Consistent daily volatility** (stocks that move a lot every day)
- **Explosive short-term spikes** (stocks that have extreme 10-20 day periods)

## Volatility Metrics in the Dataset

### 1. Daily Volatility Metrics

#### `mean_abs_daily_return`
- **What it measures:** Average absolute daily price change
- **Formula:** mean(|log_return_t|)
- **Interpretation:** Average volatility per trading day
- **Example:** 0.0472 = Stock moves 4.72% on average per day

#### `std_daily_return`
- **What it measures:** Standard deviation of daily returns
- **Formula:** std(log_return_t)
- **Interpretation:** Typical daily return variability
- **Example:** 0.0826 = Daily returns typically vary ±8.26%

### 2. Rolling Volatility Metrics (Time-Varying)

#### `rolling_vol_10d`
- **What it measures:** 10-day rolling window volatility (annualized)
- **Formula:** sqrt(252) × std(returns over last 10 days)
- **Updates:** Daily as new data arrives
- **Use:** Detect recent short-term volatility spikes

#### `rolling_vol_20d`
- **What it measures:** 20-day rolling window volatility (annualized)
- **Formula:** sqrt(252) × std(returns over last 20 days)
- **Updates:** Daily as new data arrives
- **Use:** Smoother measure of recent volatility trends

### 3. Summary Metrics (Fixed per Stock)

#### `avg_10d_vol`
- **What it measures:** Average of all 10-day rolling volatilities
- **Interpretation:** Typical short-term volatility level
- **Example:** 1.09 = Stock averages 109% annualized volatility over 10-day periods

#### `avg_20d_vol`
- **What it measures:** Average of all 20-day rolling volatilities
- **Interpretation:** Typical monthly volatility level
- **Example:** 1.17 = Stock averages 117% annualized volatility over 20-day periods

#### `peak_10d_vol`
- **What it measures:** Maximum 10-day rolling volatility observed
- **Interpretation:** Most extreme short-term volatility spike
- **Example:** 3.68 = Highest 10-day period reached 368% annualized volatility

#### `peak_20d_vol`
- **What it measures:** Maximum 20-day rolling volatility observed
- **Interpretation:** Most extreme monthly volatility spike
- **Example:** 2.79 = Highest 20-day period reached 279% annualized volatility

#### `annualized_volatility`
- **What it measures:** Overall realized volatility for entire period
- **Formula:** sqrt(252) × std(all_daily_returns)
- **Interpretation:** Traditional volatility measure
- **Example:** 1.31 = 131% annualized volatility

#### `composite_score`
- **What it measures:** Weighted combination of all metrics
- **Interpretation:** Overall meme stock volatility ranking
- **Higher = More volatile** (better for meme stock analysis)

## Top 10 Most Volatile Stocks (Daily Metrics)

| Ticker | Daily Vol (%) | Peak 10d Vol | Peak 20d Vol | Composite Score |
|--------|--------------|--------------|--------------|-----------------|
| SRNE   | 24.74%       | 3104.93%     | 2147.20%     | 27.34           |
| MAXN   | 7.26%        | 640.25%      | 461.37%      | 7.38            |
| STEM   | 6.35%        | 414.10%      | 316.70%      | 6.21            |
| KULR   | 6.33%        | 397.07%      | 320.46%      | 6.25            |
| LIDR   | 5.88%        | 507.77%      | 380.55%      | 6.01            |
| GOEV   | 5.79%        | 709.37%      | 497.02%      | 6.29            |
| SPWR   | 5.67%        | 473.62%      | 433.14%      | 5.98            |
| WAVE   | 5.64%        | 616.26%      | 451.59%      | 6.00            |
| CIFR   | 5.55%        | 186.71%      | 172.37%      | 5.14            |
| CAN    | 5.42%        | 242.43%      | 200.77%      | 5.09            |

## Key Statistics

### Daily Volatility (Mean Absolute Return)
- **Mean:** 4.11% per day
- **Median:** 3.96% per day
- **Range:** 2.35% - 24.74% per day

**Interpretation:** On average, these stocks move ±4% daily. SRNE moves an astounding ±24.7% per day!

### Peak 20-Day Rolling Volatility
- **Mean:** 202.20% annualized
- **Median:** 154.90% annualized
- **Range:** 90.76% - 2147.20% annualized

**Interpretation:** During their most volatile 20-day periods, stocks averaged 202% annualized volatility. SRNE peaked at 2147%!

## How to Use These Metrics

### For Identifying Meme Stocks
- High `mean_abs_daily_return` → Consistent daily action
- High `peak_20d_vol` → Explosive squeeze potential
- High `composite_score` → Overall meme stock candidate

### For Trading Analysis
- `rolling_vol_10d` > `rolling_vol_20d` → Recent volatility spike
- `peak_10d_vol` >> `avg_10d_vol` → Had extreme volatility event
- High `mean_abs_daily_return` → Good for day trading

### For Risk Management
- `std_daily_return` → Size positions accordingly
- `peak_20d_vol` → Worst-case volatility scenario
- `rolling_vol_20d` → Current risk level

## Comparison: Old vs New Method

### Old Method (Annualized Only)
```python
volatility = sqrt(252) × std(all_returns)
```
- ✗ Only looked at overall period
- ✗ Missed short-term explosive moves
- ✗ Treated all volatility equally

### New Method (Daily + Rolling + Peaks)
```python
composite_score =
  30% × daily_volatility +
  20% × annualized_vol +
  25% × avg_10d_vol +
  25% × peak_20d_vol
```
- ✓ Captures daily volatility
- ✓ Identifies short-term spikes
- ✓ Weights explosive moves heavily
- ✓ Better represents meme stock behavior

## Example: SRNE (Sorrento Therapeutics)

```
Daily Volatility:     24.74% per day (moves ±25% daily!)
Peak 10-day Vol:      3104.93% annualized
Peak 20-day Vol:      2147.20% annualized
Average 10-day Vol:   619.75% annualized
Average 20-day Vol:   673.64% annualized
Composite Score:      27.34 (highest in dataset)
```

This stock is the **ultimate meme stock** - extreme daily volatility combined with absolutely explosive short-term spikes.

## Dataset Columns Reference

| Column | Type | Description |
|--------|------|-------------|
| `date` | Date | Trading date |
| `ticker` | String | Stock symbol |
| `adj_close` | Float | Adjusted closing price |
| `volume` | Integer | Trading volume |
| `return_log_daily` | Float | Daily log return |
| `rolling_vol_10d` | Float | 10-day rolling volatility (varies daily) |
| `rolling_vol_20d` | Float | 20-day rolling volatility (varies daily) |
| `mean_abs_daily_return` | Float | Average absolute daily return (fixed) |
| `std_daily_return` | Float | Std dev of daily returns (fixed) |
| `avg_10d_vol` | Float | Average 10-day rolling vol (fixed) |
| `avg_20d_vol` | Float | Average 20-day rolling vol (fixed) |
| `peak_10d_vol` | Float | Maximum 10-day rolling vol (fixed) |
| `peak_20d_vol` | Float | Maximum 20-day rolling vol (fixed) |
| `annualized_volatility` | Float | Overall annualized volatility (fixed) |
| `composite_score` | Float | Selection score (fixed) |

**Note:** "varies daily" = changes for each date row; "fixed" = same value for all dates of a given ticker
