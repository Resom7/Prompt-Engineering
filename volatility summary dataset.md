# Meme Stock Dataset Summary

## Overview
This dataset contains daily price and volatility data for the 92 most volatile US-listed stocks over the period from January 1, 2024 to May 20, 2025.

## Dataset File
**Filename:** `dataset_yahoo_top100_most_volatile_us_stocks.csv`
**Size:** 3.4 MB
**Rows:** 31,832 observations
**Date Range:** 2024-01-02 to 2025-05-19 (503 days)

## Data Columns
1. **date** - Trading date (YYYY-MM-DD)
2. **ticker** - Stock symbol
3. **adj_close** - Adjusted closing price
4. **volume** - Trading volume
5. **return_log_daily** - Daily log returns: ln(price_t / price_{t-1})
6. **rolling_vol_20d** - 20-day rolling annualized volatility
7. **overall_volatility** - Overall realized volatility for the full period (used for ranking)

## Volatility Calculation
- **Formula:** volatility = sqrt(252) × std(daily_log_returns)
- **Period:** Entire date range (505 days)
- **Minimum observations required:** 100 daily returns per stock

## Stock Universe
The dataset focuses on **high-volatility, meme-stock candidates** including:

### Top 10 Most Volatile Stocks
1. **SRNE** (Sorrento) - 862.51% annualized volatility
2. **SAVA** (Cassava Sciences) - 189.12%
3. **MAXN** (Maxeon Solar) - 181.19%
4. **GOEV** (Canoo) - 179.43%
5. **LIDR** (AEye) - 154.42%
6. **SPWR** (SunPower) - 151.31%
7. **FUBO** (fuboTV) - 133.38%
8. **APPS** (Digital Turbine) - 131.38%
9. **GME** (GameStop) - 122.05%
10. **LAZR** (Luminar) - 118.81%

### Classic Meme Stocks Included
- **GME** (GameStop): 122.05% volatility
- **AMC** (AMC Entertainment): 87.77%
- **BBBY** (Bed Bath & Beyond): 92.45%
- **BB** (BlackBerry): 61.14%
- **PLTR** (Palantir): 69.78%
- **TSLA** (Tesla): 67.52%
- **COIN** (Coinbase): 84.13%
- **HOOD** (Robinhood): 70.60%
- **SOFI** (SoFi): 62.89%

### Stock Categories
The universe includes:
- Classic meme stocks (GME, AMC, BBBY)
- Crypto-related stocks (MARA, RIOT, BTBT, MSTR, CAN, HUT)
- EV companies (RIVN, LCID, NIO, XPEV, LI, GOEV)
- High-volatility biotech (SRNE, SAVA, NVAX, VXRT, OCGN, MRNA, BNTX)
- Small-cap tech (APPS, FUBO, LAZR, LIDR, MVIS, CRSR)
- Recent IPOs and SPACs

## Volatility Statistics
- **Total stocks:** 92
- **Mean volatility:** 89.64%
- **Median volatility:** 75.06%
- **Min volatility:** 32.19% (NOK)
- **Max volatility:** 862.51% (SRNE)
- **Stocks with >100% volatility:** 19 (extreme meme territory)
- **Stocks with >50% volatility:** 77 (84% of dataset)

## Data Quality
- **Missing adj_close:** 0 (100% complete)
- **Missing volume:** 0 (100% complete)
- **Missing return_log_daily:** 92 (one per ticker on first day - expected)
- **Missing rolling_vol_20d:** 920 (first 10 days per ticker - expected)

## Usage
Load the dataset in Python:
```python
import pandas as pd

df = pd.read_csv('dataset_yahoo_top100_most_volatile_us_stocks.csv')
df['date'] = pd.to_datetime(df['date'])
```

## Source & Methodology
- **Data Source:** Yahoo Finance via yfinance library
- **Selection Method:** Ranked all stocks by realized volatility, selected top 100
- **Initial Universe:** 109 tickers (known meme stocks + volatile small/mid-caps)
- **Successfully Downloaded:** 92 tickers (17 failed due to delisting/data issues)
- **Date Range:** Matched to Reddit dataset (dataset_meme_reddit_historical_1.csv)

## Script
The dataset was generated using: `build_meme_stock_dataset.py`

To regenerate:
```bash
python build_meme_stock_dataset.py
```

## Notes
- Some tickers failed to download (TWTR, WISH, RIDE, etc.) likely due to delisting or merger/bankruptcy
- Wikipedia scraping was blocked (403 Forbidden), so the dataset relies on the curated meme stock list
- The dataset is much more representative of actual meme stock behavior than traditional indices like S&P 500
- Compare this to S&P 500: mean volatility would be ~28% vs. our 89.64%

## Related Files
- `build_meme_stock_dataset.py` - Main generation script
- `verify_dataset.py` - Verification and analysis script
- `dataset_meme_reddit_historical_1.csv` - Source Reddit data for date range
