"""
Build Dataset: Top 100 Most Volatile US Stocks from Yahoo Finance

This script:
1. Loads the Reddit meme stock dataset to infer the date range
2. Constructs a universe of US-listed stocks (S&P 500)
3. Downloads daily price data from Yahoo Finance for that period
4. Computes realized volatility for each stock
5. Selects the 100 most volatile stocks
6. Creates a comprehensive dataset with returns and rolling volatility
7. Saves the final dataset as CSV

Author: Claude Code
Date: 2025-12-11
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries, install if not available
try:
    import yfinance as yf
except ImportError:
    print("yfinance not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

# Install lxml for Wikipedia scraping (needed for pd.read_html)
try:
    import lxml
except ImportError:
    print("lxml not found. Installing for Wikipedia data access...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lxml"])
    import lxml

from datetime import datetime
import time

print("="*80)
print("BUILDING TOP 100 MOST VOLATILE US STOCKS DATASET")
print("="*80)
print()

# ============================================================================
# STEP 1: Load Reddit dataset and extract date range
# ============================================================================
print("STEP 1: Loading Reddit dataset and extracting date range...")
print("-"*80)

try:
    # Try to load the dataset
    reddit_df = pd.read_csv('dataset_meme_reddit_historical_1.csv')
    print(f"[OK] Loaded Reddit dataset: {len(reddit_df)} rows")

    # Check for timestamp column
    if 'created_dt' in reddit_df.columns:
        timestamp_col = 'created_dt'
    elif 'timestamp' in reddit_df.columns:
        timestamp_col = 'timestamp'
    elif 'date' in reddit_df.columns:
        timestamp_col = 'date'
    else:
        raise ValueError(f"Could not find timestamp column. Available columns: {reddit_df.columns.tolist()}")

    print(f"[OK] Using timestamp column: '{timestamp_col}'")

    # Convert to datetime
    reddit_df[timestamp_col] = pd.to_datetime(reddit_df[timestamp_col])

    # Extract date range (date only, no time)
    start_date = reddit_df[timestamp_col].min().date()
    end_date = reddit_df[timestamp_col].max().date()

    print(f"[OK] Date range extracted:")
    print(f"  Start date: {start_date}")
    print(f"  End date:   {end_date}")
    print(f"  Duration:   {(pd.Timestamp(end_date) - pd.Timestamp(start_date)).days} days")
    print()

except FileNotFoundError:
    raise FileNotFoundError(
        "Could not find 'dataset_meme_reddit_historical_1.csv' in the current directory. "
        "Please ensure the file exists."
    )
except Exception as e:
    raise Exception(f"Error loading Reddit dataset: {str(e)}")

# ============================================================================
# STEP 2: Construct US stock ticker universe
# ============================================================================
print("STEP 2: Constructing US stock ticker universe...")
print("-"*80)

# We'll build a comprehensive universe from multiple sources to capture smaller-cap,
# more volatile stocks that are typical meme stock candidates
print("Building broad US stock universe from multiple indices...")

tickers = []

# Method 1: Try to get NASDAQ 100 (includes more tech/growth stocks)
try:
    print("  Fetching NASDAQ 100 tickers...")
    nasdaq100_table = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')
    nasdaq100_df = nasdaq100_table[4]  # The main table with tickers
    nasdaq100_tickers = nasdaq100_df['Ticker'].tolist()
    nasdaq100_tickers = [t.replace('.', '-') for t in nasdaq100_tickers]
    tickers.extend(nasdaq100_tickers)
    print(f"    [OK] Added {len(nasdaq100_tickers)} NASDAQ 100 tickers")
except Exception as e:
    print(f"    Warning: Could not fetch NASDAQ 100: {e}")

# Method 2: Try to get Russell 2000 small-cap stocks (more volatile, meme-stock candidates)
try:
    print("  Fetching Russell 2000 tickers (sample)...")
    russell_table = pd.read_html('https://en.wikipedia.org/wiki/Russell_2000_Index')
    russell_df = russell_table[2]  # Component list
    russell_tickers = russell_df['Ticker'].head(2000).tolist()  # Get all 2000 tickers
    russell_tickers = [t.replace('.', '-') for t in russell_tickers]
    tickers.extend(russell_tickers)
    print(f"    [OK] Added {len(russell_tickers)} Russell 2000 tickers")
except Exception as e:
    print(f"    Warning: Could not fetch Russell 2000: {e}")

# Method 3: Add known meme stocks and volatile small-caps
print("  Adding known meme stocks and volatile small/mid-caps...")
meme_and_volatile_stocks = [
    # Classic meme stocks
    'GME', 'AMC', 'BBBY', 'BB', 'NOK', 'PLTR', 'CLOV', 'SOFI', 'HOOD',
    'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'BYND', 'DKNG', 'COIN', 'RBLX', 'UPST',

    # High volatility tech/growth
    'TSLA', 'NVDA', 'AMD', 'MARA', 'RIOT', 'SKLZ', 'SPCE', 'OPEN', 'ROOT', 'FUBO',
    'PLUG', 'FCEL', 'BLNK', 'CHPT', 'QS', 'WKHS', 'GOEV',

    # Biotech/pharma (volatile)
    'MRNA', 'BNTX', 'NVAX', 'VXRT', 'OCGN', 'SAVA', 'ARKG', 'SRNE', 'INO', 'ATOS',

    # Small-cap tech
    'MVIS', 'LAZR', 'OUST', 'LIDR', 'CRSR', 'MGNI', 'APPS', 'UWMC', 'RKT',
    'PATH', 'DDOG', 'SNOW', 'CRWD', 'ZS', 'NET', 'OKTA',

    # SPACs and recent IPOs
    'OPAD', 'ATER', 'KPLT', 'CLNE', 'WOOF', 'IONQ', 'HIMS', 'DOCS', 'FOUR',

    # Crypto-related
    'MSTR', 'PYPL', 'CAN', 'HUT', 'BTBT', 'SOS', 'CLSK', 'CIFR', 'BITF',

    # Other volatile small/mid caps
    'SPWR', 'ENPH', 'SEDG', 'RUN', 'MAXN', 'CSIQ', 'JKS', 'DQ',
    'PINS', 'SNAP', 'MTCH', 'BMBL', 'UBER', 'LYFT', 'DASH', 'ABNB', 'AFRM',

    # Additional volatile stocks to reach 100+
    'W', 'CVNA', 'CARVANA', 'SHOP', 'SQ', 'HOOD', 'ROKU', 'DOCU', 'ZM', 'PTON',
    'TDOC', 'SPOT', 'DKNG', 'PENN', 'DRAFT', 'SKLZ', 'GNOG',
    'BILI', 'GRPN', 'YELP', 'SNAP', 'TWLO', 'DBX', 'BOX',
    'ESTC', 'MDB', 'TEAM', 'WDAY', 'NOW', 'HUBS', 'ZI', 'BILL',
    'SMAR', 'FRSH', 'GTLB', 'S', 'NEGG', 'EXPR', 'KIRK', 'APRN',
    'PUBM', 'TTD', 'MGNI', 'APPS', 'IAS', 'ACHR', 'JOBY', 'LILM',
    'EVTL', 'ARVL', 'MULN', 'GEV', 'HYLN', 'PROTERRA', 'LEV',
    'FFIE', 'RIDE', 'WKHS', 'NKLA', 'FSR', 'GOEV', 'CANOO',
    'AYRO', 'SOLO', 'NIU', 'KNDI', 'BLNK', 'CHPT', 'EVGO',
    'STEM', 'NUVVE', 'WAVE', 'KULR', 'FLUX', 'PTRA'
]
tickers.extend(meme_and_volatile_stocks)
print(f"    [OK] Added {len(meme_and_volatile_stocks)} known meme/volatile stocks")

# Remove duplicates and clean
tickers = list(set(tickers))
tickers = [t.strip().upper() for t in tickers if t and isinstance(t, str)]

# If we still don't have enough tickers, add S&P 500 as backup
if len(tickers) < 200:
    print("  Adding S&P 500 for additional coverage...")
    try:
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_df = sp500_table[0]
        sp500_tickers = sp500_df['Symbol'].tolist()
        sp500_tickers = [t.replace('.', '-') for t in sp500_tickers]
        tickers.extend(sp500_tickers)
        tickers = list(set(tickers))
        print(f"    [OK] Added S&P 500 tickers")
    except:
        pass

# Final fallback if everything failed
if len(tickers) < 50:
    print("  Warning: Using comprehensive fallback ticker list...")
    tickers = [
        # Meme stocks
        'GME', 'AMC', 'BBBY', 'BB', 'NOK', 'PLTR', 'WISH', 'CLOV', 'SOFI', 'HOOD',
        'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'BYND', 'DKNG', 'COIN', 'RBLX', 'UPST',
        # High vol tech
        'TSLA', 'NVDA', 'AMD', 'MARA', 'RIOT', 'SPCE', 'PLUG', 'FCEL', 'BLNK', 'QS',
        # Large caps for diversity
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'INTC', 'MU', 'QCOM', 'AVGO',
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PYPL',
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY'
    ]

print(f"\n[OK] Total ticker universe: {len(tickers)} unique tickers")
print(f"  Sample tickers: {tickers[:15]}")
print()

# ============================================================================
# STEP 3: Download price data from Yahoo Finance
# ============================================================================
print("STEP 3: Downloading price data from Yahoo Finance...")
print("-"*80)
print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
print("This may take several minutes...")
print()

# Download data in batches to avoid rate limits
batch_size = 100
all_data = []
failed_tickers = []

for i in range(0, len(tickers), batch_size):
    batch_tickers = tickers[i:i+batch_size]
    print(f"  Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1} ({len(batch_tickers)} tickers)...")

    try:
        # Download data for batch
        data = yf.download(
            batch_tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='ticker',
            auto_adjust=False  # We'll use Adj Close explicitly
        )

        # If only one ticker, reshape the data
        if len(batch_tickers) == 1:
            ticker = batch_tickers[0]
            if not data.empty:
                temp_df = data[['Adj Close', 'Volume']].copy()
                temp_df['ticker'] = ticker
                temp_df['date'] = temp_df.index
                temp_df.columns = ['adj_close', 'volume', 'ticker', 'date']
                all_data.append(temp_df)
        else:
            # Process multi-ticker data
            for ticker in batch_tickers:
                try:
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker][['Adj Close', 'Volume']].copy()
                        ticker_data = ticker_data.dropna(subset=['Adj Close'])

                        if len(ticker_data) > 0:
                            ticker_data['ticker'] = ticker
                            ticker_data['date'] = ticker_data.index
                            ticker_data.columns = ['adj_close', 'volume', 'ticker', 'date']
                            all_data.append(ticker_data)
                        else:
                            failed_tickers.append(ticker)
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    failed_tickers.append(ticker)
                    continue

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    except Exception as e:
        print(f"    Warning: Batch download failed: {e}")
        failed_tickers.extend(batch_tickers)
        continue

# Combine all data
if not all_data:
    raise Exception("No data was downloaded successfully. Please check your internet connection and try again.")

yahoo_df = pd.concat(all_data, ignore_index=True)
yahoo_df['date'] = pd.to_datetime(yahoo_df['date']).dt.date

print(f"[OK] Downloaded data for {yahoo_df['ticker'].nunique()} tickers")
print(f"  Total rows: {len(yahoo_df)}")
if failed_tickers:
    print(f"  Failed tickers ({len(failed_tickers)}): {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
print()

# ============================================================================
# STEP 4: Compute daily log returns and multiple volatility metrics
# ============================================================================
print("STEP 4: Computing daily log returns and volatility metrics...")
print("-"*80)

# Sort by ticker and date
yahoo_df = yahoo_df.sort_values(['ticker', 'date'])

# Compute log returns for each ticker
yahoo_df['return_log_daily'] = yahoo_df.groupby('ticker')['adj_close'].transform(
    lambda x: np.log(x / x.shift(1))
)

# Compute rolling volatilities (10-day and 20-day)
yahoo_df['rolling_vol_10d'] = yahoo_df.groupby('ticker')['return_log_daily'].transform(
    lambda x: np.sqrt(252) * x.rolling(window=10, min_periods=5).std()
)
yahoo_df['rolling_vol_20d'] = yahoo_df.groupby('ticker')['return_log_daily'].transform(
    lambda x: np.sqrt(252) * x.rolling(window=20, min_periods=10).std()
)

# Compute multiple volatility metrics for each ticker
volatility_results = []

for ticker in yahoo_df['ticker'].unique():
    ticker_data = yahoo_df[yahoo_df['ticker'] == ticker].copy()

    # Get returns (excluding NaN from first return)
    returns = ticker_data['return_log_daily'].dropna()

    # Only include stocks with sufficient data
    if len(returns) >= 100:
        # 1. Mean absolute daily return (average daily volatility)
        mean_abs_daily_return = returns.abs().mean()

        # 2. Standard deviation of daily returns (daily volatility)
        std_daily_return = returns.std()

        # 3. Annualized realized volatility (for reference)
        annualized_vol = np.sqrt(252) * std_daily_return

        # 4. Average 10-day rolling volatility
        avg_10d_vol = ticker_data['rolling_vol_10d'].mean()

        # 5. Average 20-day rolling volatility
        avg_20d_vol = ticker_data['rolling_vol_20d'].mean()

        # 6. Peak (max) 10-day rolling volatility
        peak_10d_vol = ticker_data['rolling_vol_10d'].max()

        # 7. Peak (max) 20-day rolling volatility
        peak_20d_vol = ticker_data['rolling_vol_20d'].max()

        # Create composite score: weighted average of daily vol and peak short-term vol
        # This captures both consistent daily volatility and explosive short-term moves
        composite_score = (
            0.3 * mean_abs_daily_return * 252 +  # Annualized mean absolute return
            0.2 * annualized_vol +                # Overall annualized volatility
            0.25 * avg_10d_vol +                  # Average 10-day volatility
            0.25 * peak_20d_vol                   # Peak 20-day volatility
        )

        volatility_results.append({
            'ticker': ticker,
            'mean_abs_daily_return': mean_abs_daily_return,
            'std_daily_return': std_daily_return,
            'annualized_volatility': annualized_vol,
            'avg_10d_vol': avg_10d_vol,
            'avg_20d_vol': avg_20d_vol,
            'peak_10d_vol': peak_10d_vol,
            'peak_20d_vol': peak_20d_vol,
            'composite_score': composite_score,
            'num_observations': len(returns)
        })

volatility_df = pd.DataFrame(volatility_results)
print(f"[OK] Computed volatility metrics for {len(volatility_df)} tickers")
print(f"  (Excluded {yahoo_df['ticker'].nunique() - len(volatility_df)} tickers with < 100 observations)")
print(f"\nVolatility metrics calculated:")
print(f"  - Mean absolute daily return (daily volatility)")
print(f"  - Average 10-day rolling volatility")
print(f"  - Average 20-day rolling volatility")
print(f"  - Peak (max) 10-day and 20-day rolling volatility")
print()

# ============================================================================
# STEP 5: Select top 100 most volatile stocks
# ============================================================================
print("STEP 5: Selecting top 100 most volatile stocks...")
print("-"*80)

# Sort by composite score (daily vol + peak rolling vol) descending and take top 100
volatility_df = volatility_df.sort_values('composite_score', ascending=False)
top_100_df = volatility_df.head(100).copy()

top_100_tickers = top_100_df['ticker'].tolist()

print(f"[OK] Selected top 100 most volatile tickers (by composite score)")
print(f"\nSelection criteria:")
print(f"  - 30% weight: Mean absolute daily return")
print(f"  - 20% weight: Overall annualized volatility")
print(f"  - 25% weight: Average 10-day rolling volatility")
print(f"  - 25% weight: Peak 20-day rolling volatility")
print(f"\nTop 10 most volatile stocks:")
top_10_display = top_100_df.head(10)[['ticker', 'mean_abs_daily_return', 'avg_10d_vol', 'avg_20d_vol', 'peak_20d_vol', 'composite_score']].copy()
top_10_display.columns = ['Ticker', 'Daily Vol (%)', 'Avg 10d Vol', 'Avg 20d Vol', 'Peak 20d Vol', 'Score']
top_10_display['Daily Vol (%)'] = top_10_display['Daily Vol (%)'] * 100
print(top_10_display.to_string(index=False))

print(f"\nDaily volatility statistics (mean absolute return):")
print(f"  Mean:   {top_100_df['mean_abs_daily_return'].mean():.2%} per day")
print(f"  Median: {top_100_df['mean_abs_daily_return'].median():.2%} per day")
print(f"  Min:    {top_100_df['mean_abs_daily_return'].min():.2%} per day")
print(f"  Max:    {top_100_df['mean_abs_daily_return'].max():.2%} per day")

print(f"\nPeak 20-day rolling volatility statistics:")
print(f"  Mean:   {top_100_df['peak_20d_vol'].mean():.2%}")
print(f"  Median: {top_100_df['peak_20d_vol'].median():.2%}")
print(f"  Min:    {top_100_df['peak_20d_vol'].min():.2%}")
print(f"  Max:    {top_100_df['peak_20d_vol'].max():.2%}")
print()

# ============================================================================
# STEP 6: Create final dataset with all volatility metrics
# ============================================================================
print("STEP 6: Creating final dataset with all volatility metrics...")
print("-"*80)

# Filter to top 100 tickers only
final_df = yahoo_df[yahoo_df['ticker'].isin(top_100_tickers)].copy()

# Add all volatility metrics (same value for all rows of a given ticker)
metrics_to_add = top_100_df[[
    'ticker', 'mean_abs_daily_return', 'std_daily_return', 'annualized_volatility',
    'avg_10d_vol', 'avg_20d_vol', 'peak_10d_vol', 'peak_20d_vol', 'composite_score'
]]

final_df = final_df.merge(metrics_to_add, on='ticker', how='left')

# Sort by date and ticker for clean output
final_df = final_df.sort_values(['date', 'ticker'])

# Reorder columns for clarity
final_df = final_df[[
    'date', 'ticker', 'adj_close', 'volume',
    'return_log_daily',
    'rolling_vol_10d', 'rolling_vol_20d',
    'mean_abs_daily_return', 'std_daily_return',
    'avg_10d_vol', 'avg_20d_vol',
    'peak_10d_vol', 'peak_20d_vol',
    'annualized_volatility', 'composite_score'
]]

print(f"[OK] Final dataset created")
print(f"  Rows: {len(final_df)}")
print(f"  Tickers: {final_df['ticker'].nunique()}")
print(f"  Date range: {final_df['date'].min()} to {final_df['date'].max()}")
print(f"  Columns: {len(final_df.columns)}")
print(f"\nColumns included:")
print(f"  Basic: date, ticker, adj_close, volume, return_log_daily")
print(f"  Rolling: rolling_vol_10d, rolling_vol_20d")
print(f"  Daily metrics: mean_abs_daily_return, std_daily_return")
print(f"  Average rolling: avg_10d_vol, avg_20d_vol")
print(f"  Peak rolling: peak_10d_vol, peak_20d_vol")
print(f"  Summary: annualized_volatility, composite_score")
print()

# ============================================================================
# STEP 7: Save dataset and display summary
# ============================================================================
print("STEP 7: Saving dataset and displaying summary...")
print("-"*80)

output_filename = 'dataset_yahoo_top100_most_volatile_us_stocks.csv'
final_df.to_csv(output_filename, index=False)

print(f"[OK] Dataset saved to: {output_filename}")
print()

# Display summary
print("="*80)
print("DATASET SUMMARY")
print("="*80)
print(f"\nDate Range:")
print(f"  Start: {final_df['date'].min()}")
print(f"  End:   {final_df['date'].max()}")
print(f"  Days:  {(pd.Timestamp(final_df['date'].max()) - pd.Timestamp(final_df['date'].min())).days}")

print(f"\nDataset Size:")
print(f"  Number of tickers: {final_df['ticker'].nunique()}")
print(f"  Total rows: {len(final_df)}")
print(f"  Average rows per ticker: {len(final_df) / final_df['ticker'].nunique():.0f}")

print(f"\nFirst 10 rows:")
print(final_df.head(10).to_string(index=False))

print(f"\nLast 10 rows:")
print(final_df.tail(10).to_string(index=False))

print(f"\nData quality:")
print(f"  Missing adj_close: {final_df['adj_close'].isna().sum()}")
print(f"  Missing volume: {final_df['volume'].isna().sum()}")
print(f"  Missing return_log_daily: {final_df['return_log_daily'].isna().sum()}")
print(f"  Missing rolling_vol_20d: {final_df['rolling_vol_20d'].isna().sum()}")

print("\n" + "="*80)
print("DONE! Dataset successfully created.")
print("="*80)
print(f"\nYou can now load this dataset with:")
print(f"  df = pd.read_csv('{output_filename}')")
print()
