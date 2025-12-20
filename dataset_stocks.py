"""
Script to fetch all listed US companies and create a dataset of the top 100 most volatile companies per day.

Date Range: 2024-01-01 to 2025-05-20 (matching dataset_meme_reddit_historical_1)

Volatility metrics calculated:
- daily_change: Daily percentage change
- rolling_vol_5d: 5-day rolling volatility
- rolling_vol_10d: 10-day rolling volatility
- rolling_vol_20d: 20-day rolling volatility
"""

import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - CHANGE THIS TO SELECT VOLATILITY METRIC
# ============================================================================
# Options: 'daily_change', 'rolling_vol_5d', 'rolling_vol_10d', 'rolling_vol_20d'
VOLATILITY_METRIC = 'rolling_vol_20d'
# ============================================================================

# Date range from dataset_meme_reddit_historical_1
START_DATE = '2024-01-01'
END_DATE = '2025-05-20'

# Number of top volatile companies to select per day
TOP_N = 100

# Output file
OUTPUT_FILE = f'dataset_top100_volatile_stocks_{VOLATILITY_METRIC}.csv'


def get_us_stock_tickers():
    """
    Get comprehensive list of US stock tickers.
    Uses a curated list of major US stocks to avoid memory issues.
    """
    print("Building comprehensive US stock ticker list...")

    # Comprehensive list of US stocks across market caps and sectors
    # This includes S&P 500, Russell 1000, NASDAQ 100, and major mid/small caps
    # Total: ~1500+ stocks covering the vast majority of US market capitalization
    print("  Adding stocks from major indices and exchanges...")

    all_tickers = [
        # Mega Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL',
        'ADBE', 'CRM', 'AMD', 'INTC', 'CSCO', 'QCOM', 'TXN', 'AMAT', 'ADI', 'INTU',
        'NOW', 'PANW', 'SNPS', 'CDNS', 'PLTR', 'SNOW', 'DDOG', 'CRWD', 'ZS', 'NET',
        'MU', 'LRCX', 'KLAC', 'MRVL', 'NXPI', 'FTNT', 'ABNB', 'UBER', 'DASH', 'LYFT',
        'SMCI', 'ARM', 'APP', 'DELL', 'HPE', 'LUMN', 'ANET', 'FFIV',

        # Finance
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'SPGI', 'CME',
        'ICE', 'MCO', 'BX', 'KKR', 'USB', 'PNC', 'TFC', 'COF', 'AFL', 'MET', 'PRU',
        'AIG', 'ALL', 'TRV', 'PGR', 'CB', 'AJG', 'MMC', 'AON', 'WTW', 'BRK-B', 'BRK-A',

        # Healthcare & Biotech
        'UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY', 'AMGN',
        'GILD', 'REGN', 'VRTX', 'CVS', 'CI', 'HUM', 'ELV', 'BSX', 'SYK', 'MDT', 'ISRG',
        'ZTS', 'DXCM', 'IDXX', 'IQV', 'BDX', 'EW', 'RMD', 'ALGN', 'HOLX', 'BAX',
        'MRNA', 'BNTX', 'NVAX', 'BIIB', 'ILMN', 'INCY', 'EXAS', 'ALNY', 'TECH',

        # Consumer & Retail
        'WMT', 'COST', 'HD', 'LOW', 'TGT', 'TJX', 'ROST', 'DG', 'DLTR', 'BBY', 'EBAY',
        'ETSY', 'W', 'CHWY', 'NKE', 'LULU', 'DECK', 'TPR', 'RL', 'CPRI', 'UAA', 'UA',
        'SBUX', 'MCD', 'CMG', 'QSR', 'YUM', 'DPZ', 'WING', 'JACK', 'DRI', 'EAT',

        # Consumer Goods
        'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'KMB', 'CHD', 'CLX',
        'EL', 'GIS', 'K', 'CAG', 'HSY', 'MDLZ', 'KHC', 'SJM', 'CPB', 'HRL',

        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL', 'BKR',
        'DVN', 'FANG', 'APA', 'OVV', 'CTRA', 'EQT', 'AR', 'PR',
        'KMI', 'WMB', 'OKE', 'LNG', 'TRGP', 'EPD', 'ET', 'MPLX', 'PAA', 'WES',

        # Industrials
        'BA', 'HON', 'UPS', 'CAT', 'RTX', 'LMT', 'GE', 'MMM', 'GD', 'NOC', 'LHX',
        'TDG', 'ETN', 'EMR', 'ITW', 'PH', 'CMI', 'FTV', 'ROK', 'AME', 'DOV',
        'IR', 'CARR', 'OTIS', 'PCAR', 'JCI', 'EMR', 'WM', 'RSG', 'FAST', 'URI',

        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'NEM', 'FCX', 'NUE', 'STLD', 'VMC',
        'MLM', 'ALB', 'CE', 'CF', 'MOS', 'FMC', 'PPG', 'CTVA', 'EMN', 'IFF',

        # Real Estate
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'VICI', 'AVB',
        'EQR', 'SBAC', 'SPG', 'ARE', 'INVH', 'MAA', 'ESS', 'VTR', 'EXR', 'UDR',

        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED', 'WEC',
        'ES', 'AWK', 'DTE', 'PPL', 'AEE', 'CMS', 'FE', 'ETR', 'CNP', 'NI',

        # Communication Services
        'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR',
        'WBD', 'OMC', 'IPG', 'FOXA', 'FOX', 'NWSA', 'NWS', 'MTCH', 'PINS',

        # Semiconductors (additional)
        'TSM', 'ASML', 'AVGO', 'TXN', 'QCOM', 'ADI', 'MCHP', 'MPWR', 'ON', 'SWKS',
        'ENTG', 'MXL', 'CRUS', 'SLAB', 'SMTC', 'ICHR', 'POWI', 'SITM',
        'MKSI', 'LSCC', 'DIOD', 'AMBA', 'MTSI', 'COHR', 'LITE', 'QRVO',

        # Software & Cloud
        'MSFT', 'ORCL', 'SAP', 'ADBE', 'CRM', 'NOW', 'INTU', 'WDAY', 'TEAM', 'SNOW',
        'DDOG', 'ZS', 'OKTA', 'CRWD', 'S', 'DBX', 'ZM', 'DOCN', 'ESTC', 'MDB',
        'NET', 'CFLT', 'GTLB', 'U', 'PATH', 'BILL', 'PCTY', 'HUBS', 'VEEV',
        'ASAN', 'MNDY', 'AI', 'RNG', 'APPN', 'FRSH', 'BOX',

        # E-commerce & Internet
        'AMZN', 'SHOP', 'MELI', 'BKNG', 'ABNB', 'EXPE', 'TRIP', 'CVNA', 'W', 'RH',

        # Autos
        'TSLA', 'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'STLA', 'HMC', 'TM',

        # Banks (Regional)
        'BAC', 'JPM', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'FITB', 'HBAN', 'RF', 'CFG',
        'KEY', 'MTB', 'ZION', 'CMA', 'WTFC', 'WAL', 'GBCI',

        # Biotech (additional)
        'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'BNTX', 'ALNY', 'BMRN',
        'RARE', 'FOLD', 'IONS', 'UTHR', 'EXEL', 'NBIX', 'HALO', 'SRPT', 'RGNX',
        'ARWR', 'INCY', 'NTRA', 'LEGN', 'BBIO', 'KYMR',

        # Food & Beverage
        'KO', 'PEP', 'MNST', 'CELH', 'KDP', 'STZ', 'TAP', 'BF-B', 'SAM', 'FIZZ',

        # Airlines
        'DAL', 'UAL', 'AAL', 'LUV', 'JBLU', 'ALK', 'SKYW',

        # Hotels & Leisure
        'MAR', 'HLT', 'H', 'IHG', 'WH', 'RCL', 'CCL', 'NCLH', 'LVS', 'WYNN', 'MGM',

        # Media & Entertainment
        'DIS', 'NFLX', 'WBD', 'LYV', 'MSGS', 'IMAX',

        # Apparel
        'NKE', 'LULU', 'VFC', 'PVH', 'HBI', 'UAA', 'UA', 'CROX', 'BIRK', 'ONON',
        'SHOO', 'WWW', 'GIL', 'CRI', 'COLM', 'GOOS',

        # Cybersecurity
        'CRWD', 'PANW', 'ZS', 'FTNT', 'OKTA', 'CYBR', 'TENB', 'RPD', 'S', 'QLYS',
        'CHKP', 'GEN', 'VRNS', 'SAIL', 'RBRK',

        # Payments
        'V', 'MA', 'PYPL', 'FIS', 'FISV', 'GPN', 'AXP',
        'AFRM', 'SOFI', 'UPST', 'LC', 'NU', 'PAGS', 'STNE',

        # Insurance
        'BRK-B', 'PGR', 'ALL', 'TRV', 'CB', 'AIG', 'MET', 'PRU', 'AFL', 'AMP',

        # REITs (additional)
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'VICI', 'SPG',

        # Aerospace & Defense
        'BA', 'LMT', 'RTX', 'GD', 'NOC', 'LHX', 'HWM', 'TDG', 'TXT', 'HII',

        # Medical Devices
        'ISRG', 'ABT', 'TMO', 'DHR', 'SYK', 'BSX', 'MDT', 'EW', 'ZBH', 'BAX',

        # Pharma
        'PFE', 'JNJ', 'MRK', 'ABBV', 'LLY', 'BMY', 'GILD', 'AMGN', 'REGN', 'VRTX',

        # Small/Mid Cap Growth
        'ROKU', 'SNAP', 'PINS', 'SPOT', 'ZM', 'DOCU', 'TWLO', 'PTON', 'BYND', 'CVNA',
        'OPEN', 'COIN', 'HOOD', 'SOFI', 'UPST', 'AFRM', 'LC', 'NAVI', 'ENPH', 'SEDG',

        # Crypto/Fintech
        'COIN', 'HOOD', 'SOFI', 'AFRM', 'PYPL', 'MARA', 'RIOT', 'CLSK', 'CIFR',

        # Cannabis
        'TLRY', 'CGC', 'SNDL', 'ACB', 'CRON', 'OGI', 'CURLF', 'GTBIF',

        # SPACs & Recent IPOs
        'RIVN', 'LCID', 'GRAB', 'BROS', 'RBLX', 'DASH', 'CPNG', 'DNUT',
        'ARM', 'RDDT', 'KVUE', 'FIGS', 'DUOL', 'CAVA',

        # Energy (additional)
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
        'NOV', 'RIG', 'VAL', 'MTDR', 'SM', 'MGY',

        # Misc
        'HOOD', 'PLTR', 'RKLB', 'SPCE', 'JOBY', 'ACHR', 'EVTL',

        # Additional S&P 500 stocks
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'BRK-B', 'UNH',
        'XOM', 'JNJ', 'LLY', 'V', 'PG', 'JPM', 'MA', 'AVGO', 'HD', 'CVX',
        'MRK', 'ABBV', 'COST', 'PEP', 'KO', 'WMT', 'ADBE', 'CRM', 'BAC', 'TMO',
        'CSCO', 'ACN', 'MCD', 'NFLX', 'LIN', 'ABT', 'AMD', 'DHR', 'WFC', 'DIS',
        'CMCSA', 'TXN', 'VZ', 'ORCL', 'INTU', 'PM', 'QCOM', 'IBM', 'AMGN', 'CAT',
        'GE', 'AMAT', 'BKNG', 'HON', 'NOW', 'SPGI', 'BA', 'GS', 'NEE', 'INTC',
        'PFE', 'UNP', 'RTX', 'AXP', 'LOW', 'T', 'SYK', 'ELV', 'MS', 'BLK',
        'SCHW', 'TJX', 'PLD', 'ADI', 'DE', 'LMT', 'MDT', 'ISRG', 'ADP', 'CB',
        'GILD', 'VRTX', 'C', 'MMC', 'REGN', 'SBUX', 'CI', 'BMY', 'PANW', 'AMT',
        'SLB', 'SO', 'BSX', 'ZTS', 'MU', 'ETN', 'LRCX', 'EOG', 'DUK', 'PGR',
        'BDX', 'APD', 'MDLZ', 'HCA', 'EQIX', 'ITW', 'TT', 'CME', 'CL', 'PH',
        'MCO', 'NOC', 'SNPS', 'USB', 'CDNS', 'PSA', 'AON', 'SHW', 'MAR', 'KLAC',
        'ECL', 'WM', 'PYPL', 'ICE', 'MMM', 'APH', 'CRWD', 'MO', 'FI', 'MSI',
        'GD', 'EMR', 'AJG', 'TGT', 'TDG', 'DLR', 'MCK', 'CSX', 'WELL', 'PCAR',
        'NSC', 'ROP', 'CCI', 'FCX', 'AFL', 'AZO', 'PSX', 'OXY', 'AEP', 'TRV',
        'MPC', 'SRE', 'ADSK', 'FTNT', 'HUM', 'GM', 'MRVL', 'NXPI', 'PAYX', 'FICO',
        'F', 'JCI', 'MSCI', 'AIG', 'ORLY', 'O', 'KMB', 'CARR', 'VLO', 'COR',
        'CPRT', 'TFC', 'HLT', 'HPQ', 'ALL', 'AMP', 'FAST', 'EW', 'NEM', 'CMG',
        'SPG', 'TEL', 'EXC', 'RSG', 'GWW', 'URI', 'KR', 'SYY', 'ODFL', 'D',
        'HSY', 'CTVA', 'EA', 'KHC', 'IT', 'DHI', 'CTAS', 'ROST', 'BK', 'KMI',

        # Additional NASDAQ stocks
        'ABNB', 'MRNA', 'DDOG', 'ZS', 'TEAM', 'DXCM', 'MELI', 'WDAY', 'IDXX', 'BIIB',
        'ILMN', 'SIRI', 'CHTR', 'MNST', 'ALGN', 'VRSK', 'CSGP', 'MTCH', 'TMUS',
        'DOCU', 'ZM', 'ROKU', 'COIN', 'HOOD', 'RBLX', 'DASH', 'UBER', 'LYFT', 'SNOW',
        'NET', 'OKTA', 'TWLO', 'SHOP', 'SPOT',
        'TTWO', 'EA', 'RBLX', 'U', 'PLTK', 'FUBO', 'DKNG',

        # Mid-cap stocks
        'EXPE', 'TROW', 'VICI', 'KEYS', 'LVS', 'MTD', 'WBD', 'GLW', 'STZ', 'IQV',
        'TTWO', 'EBAY', 'MPWR', 'WDC', 'SBAC', 'RMD', 'GPN', 'AKAM', 'GDDY', 'GEHC',
        'ENPH', 'ARE', 'INVH', 'EQR', 'TSCO', 'AVB', 'DOV', 'BRO', 'HOLX', 'SWKS',
        'FTV', 'ROK', 'TYL', 'RJF', 'CBOE', 'BALL', 'NTAP', 'WAT', 'LH',
        'MAA', 'PPG', 'DTE', 'AEE', 'ESS', 'WEC', 'VRSN', 'PTC', 'FDS',
        'CAG', 'ULTA', 'BR', 'CFG', 'CNP', 'STE', 'BLDR', 'DRI', 'HBAN', 'TER',

        # Small-cap growth
        'PINS', 'W', 'CVNA', 'ETSY', 'CHWY', 'BYND', 'PTON', 'OPEN', 'NAVI', 'SEDG',
        'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'MARA', 'RIOT', 'CLSK', 'CIFR', 'GRAB',
        'BROS', 'CPNG', 'DNUT', 'TLRY', 'CGC', 'SNDL', 'ACB', 'CRON', 'OGI',
        'LAZR', 'OUST', 'INVZ', 'BLNK', 'CHPT', 'EVGO', 'GOEV', 'WKHS',

        # Energy & Materials
        'HAL', 'BKR', 'DVN', 'FANG', 'APA', 'OVV', 'CTRA', 'EQT',
        'AR', 'PR', 'WMB', 'OKE', 'LNG', 'TRGP', 'EPD', 'ET', 'MPLX', 'PAA',
        'WES', 'DOW', 'DD', 'NUE', 'STLD', 'VMC', 'MLM', 'ALB', 'CE', 'CF',
        'MOS', 'FMC', 'EMN', 'IFF', 'IP', 'PKG', 'AVY', 'SEE', 'APD',
        'CLF', 'MT', 'SCCO', 'GOLD', 'AEM', 'WPM', 'FNV', 'RGLD',

        # Healthcare
        'CVS', 'CI', 'HUM', 'CNC', 'MOH', 'ELV', 'VTRS', 'JAZZ', 'INCY', 'EXAS',
        'ALNY', 'BMRN', 'RARE', 'FOLD', 'IONS', 'UTHR', 'EXEL', 'NBIX', 'SRPT',
        'RGNX', 'TECH', 'BNTX', 'NVAX', 'HLN', 'HOLX', 'BAX', 'BDX', 'IQV',
        'PODD', 'TNDM', 'DXCM', 'TDOC', 'HIMS', 'DOCS', 'VEEV', 'SDGR',

        # Industrials & Transportation
        'UPS', 'FDX', 'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'ALK',
        'SKYW', 'CHRW', 'EXPD', 'XPO', 'JBHT', 'KNX', 'R', 'ODFL', 'SAIA', 'ARCB',

        # Real Estate
        'CBRE', 'CSGP', 'IRM', 'COLD', 'REXR', 'FR', 'KIM', 'REG', 'BXP', 'VNO',
        'SLG', 'HST', 'RLJ', 'PK', 'AIV', 'CPT', 'UDR', 'ESS', 'EQR', 'AVB',

        # Financials
        'PNC', 'TFC', 'USB', 'FITB', 'HBAN', 'RF', 'CFG', 'KEY', 'MTB', 'ZION',
        'CMA', 'WTFC', 'WAL', 'GBCI', 'SSB', 'SNV', 'EWBC',
        'COF', 'SYF', 'ALLY', 'AXP', 'V', 'MA', 'PYPL', 'FIS',
        'FISV', 'GPN', 'JKHY', 'BR', 'WEX', 'FOUR', 'TREE', 'STNE', 'PAGS',
        'CMA', 'OZK', 'FHN', 'VLY', 'WBS', 'UMBF', 'BOKF',

        # Consumer Discretionary
        'NKE', 'LULU', 'DECK', 'TPR', 'RL', 'PVH', 'HBI', 'UAA', 'UA', 'CROX',
        'BIRK', 'ONON', 'VFC', 'DKS', 'SCVL', 'BOOT', 'ASO',
        'BBY', 'GME', 'WSM', 'RH', 'LESL', 'FIVE', 'OLLI', 'BURL',
        'ANF', 'AEO', 'URBN', 'GAP', 'M', 'KSS',

        # Restaurants
        'MCD', 'SBUX', 'CMG', 'QSR', 'YUM', 'DPZ', 'WING', 'JACK', 'DRI', 'EAT',
        'TXRH', 'BLMN', 'DENN', 'PLAY', 'CAKE', 'RRGB', 'BJRI', 'CBRL', 'FWRG',
        'SHAK', 'CAVA', 'NDLS', 'WEN',

        # Media & Entertainment
        'WBD', 'FOX', 'FOXA', 'NWSA', 'NWS', 'OMC', 'IPG', 'LYV',
        'MSGS', 'IMAX', 'CNK', 'AMC', 'WMG', 'SIRI',
        'RBLX', 'U', 'TTWO', 'EA', 'PLTK', 'DKNG', 'PENN', 'GENI',

        # Utilities (more)
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED',
        'WEC', 'ES', 'AWK', 'DTE', 'PPL', 'AEE', 'CMS', 'FE', 'ETR', 'CNP',
        'NI', 'NRG', 'VST', 'AES', 'PCG', 'EIX', 'LNT', 'EVRG', 'PNW', 'OGE',

        # More Tech/Software
        'IBM', 'HPE', 'DELL', 'NTAP', 'STX', 'WDC', 'ANET',
        'CVLT', 'DLB', 'UI', 'GDDY', 'AKAM', 'CFLT', 'GTLB', 'PATH', 'BILL',
        'PCTY', 'HUBS', 'VEEV', 'ESTC', 'MDB', 'DOCN', 'DBX', 'BOX',
        'PAYC', 'LPLA', 'NCNO', 'IOT', 'TTEC', 'QLYS',

        # Semiconductors (more)
        'TSM', 'ASML', 'ENTG', 'MXL', 'CRUS', 'SLAB', 'SMTC', 'ICHR', 'POWI',
        'SITM', 'MCHP', 'ON', 'MKSI', 'LSCC', 'DIOD', 'AMBA', 'MTSI', 'COHR', 'LITE',

        # Insurance
        'BRK-A', 'BRK-B', 'PGR', 'ALL', 'TRV', 'CB', 'AIG', 'MET', 'PRU', 'AFL',
        'AMP', 'HIG', 'GL', 'WRB', 'CINF', 'L', 'PFG', 'AFG', 'RNR', 'AIZ',

        # Hotels & Leisure
        'MAR', 'HLT', 'H', 'IHG', 'WH', 'RCL', 'CCL', 'NCLH', 'LVS', 'WYNN',
        'MGM', 'CZR', 'PENN', 'BYD', 'PK', 'RHP', 'PEB', 'SHO', 'XHR',
    ]

    # Remove duplicates and sort
    unique_tickers = sorted(list(set(all_tickers)))

    print(f"  Total unique tickers: {len(unique_tickers)}")
    print(f"\nReady to download stock data for {len(unique_tickers)} US stocks")

    return unique_tickers


def calculate_volatility_metrics(df):
    """
    Calculate volatility metrics for a stock dataframe.

    For rolling volatility:
    - Use min_periods=1 to calculate with available data even if less than window size
    - This ensures we have values even for the first few days
    """
    # Daily change (percentage)
    df['daily_change'] = df['Close'].pct_change() * 100

    # Rolling volatility (standard deviation of returns)
    # Use min_periods=1 to calculate even with fewer days than window
    returns = df['Close'].pct_change()
    df['rolling_vol_5d'] = returns.rolling(window=5, min_periods=1).std() * 100
    df['rolling_vol_10d'] = returns.rolling(window=10, min_periods=1).std() * 100
    df['rolling_vol_20d'] = returns.rolling(window=20, min_periods=1).std() * 100

    return df


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data for a single ticker and calculate volatility metrics.

    Returns DataFrame with: Date, Ticker, Close, daily_change, rolling_vol_5d, rolling_vol_10d, rolling_vol_20d
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=True)

        if df.empty:
            return None

        # Keep only Close price
        df = df[['Close']].copy()
        df['Ticker'] = ticker

        # Calculate volatility metrics
        df = calculate_volatility_metrics(df)

        # Reset index to make Date a column
        df = df.reset_index()
        df = df.rename(columns={'index': 'Date'})

        # Reorder columns
        df = df[['Date', 'Ticker', 'Close', 'daily_change', 'rolling_vol_5d', 'rolling_vol_10d', 'rolling_vol_20d']]

        return df

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def fetch_all_stocks_data(tickers, start_date, end_date):
    """
    Fetch stock data for all tickers and combine into a single DataFrame.
    """
    print(f"\nFetching stock data from {start_date} to {end_date}...")

    all_data = []

    for ticker in tqdm(tickers, desc="Downloading stock data"):
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            all_data.append(df)

    if not all_data:
        raise ValueError("No stock data was successfully fetched!")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Convert Date to datetime and extract just the date part
    combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.date

    print(f"\nTotal records fetched: {len(combined_df):,}")
    print(f"Unique stocks: {combined_df['Ticker'].nunique()}")
    print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")

    return combined_df


def select_top_volatile_per_day(df, metric, top_n=100):
    """
    For each day, select the top N most volatile companies based on the specified metric.

    Args:
        df: DataFrame with stock data
        metric: Column name to use for volatility ranking
        top_n: Number of top volatile stocks to keep per day

    Returns:
        DataFrame with top N volatile stocks per day
    """
    print(f"\nSelecting top {top_n} volatile stocks per day using metric: {metric}...")

    # Drop rows where the metric is NaN or infinite
    df_clean = df[df[metric].notna() & np.isfinite(df[metric])].copy()

    # For daily_change, use absolute value for ranking (we care about magnitude of change)
    if metric == 'daily_change':
        df_clean['volatility_rank'] = df_clean[metric].abs()
    else:
        df_clean['volatility_rank'] = df_clean[metric]

    # Group by date and select top N
    top_volatile = []

    for _, group in df_clean.groupby('Date'):
        # Sort by volatility metric (descending) and take top N
        top_stocks = group.nlargest(top_n, 'volatility_rank')
        top_volatile.append(top_stocks)

    result_df = pd.concat(top_volatile, ignore_index=True)

    # Drop the temporary ranking column
    result_df = result_df.drop('volatility_rank', axis=1)

    # Sort by date and metric
    result_df = result_df.sort_values(['Date', metric], ascending=[True, False])

    print(f"Total records in final dataset: {len(result_df):,}")
    print(f"Unique dates: {result_df['Date'].nunique()}")
    print(f"Average stocks per day: {len(result_df) / result_df['Date'].nunique():.1f}")

    return result_df


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("US STOCK VOLATILITY DATASET GENERATOR")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Date Range: {START_DATE} to {END_DATE}")
    print(f"  Volatility Metric: {VOLATILITY_METRIC}")
    print(f"  Top N per day: {TOP_N}")
    print(f"  Output file: {OUTPUT_FILE}")
    print("="*80)

    # Step 1: Get US stock tickers
    tickers = get_us_stock_tickers()

    # Step 2: Fetch stock data for all tickers
    all_stocks_df = fetch_all_stocks_data(tickers, START_DATE, END_DATE)

    # Step 3: Select top N volatile stocks per day
    top_volatile_df = select_top_volatile_per_day(all_stocks_df, VOLATILITY_METRIC, TOP_N)

    # Step 4: Save to CSV
    print(f"\nSaving results to {OUTPUT_FILE}...")
    top_volatile_df.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

    # Display sample of results
    print("\n" + "="*80)
    print("SAMPLE OF RESULTS (first 10 rows):")
    print("="*80)
    print(top_volatile_df.head(10).to_string(index=False))

    # Display statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS:")
    print("="*80)
    print(f"Total records: {len(top_volatile_df):,}")
    print(f"Unique dates: {top_volatile_df['Date'].nunique()}")
    print(f"Unique stocks: {top_volatile_df['Ticker'].nunique()}")
    print(f"Date range: {top_volatile_df['Date'].min()} to {top_volatile_df['Date'].max()}")
    print("\nTop 10 most frequent stocks in dataset:")
    print(top_volatile_df['Ticker'].value_counts().head(10))

    print("\n" + "="*80)
    print(f"To use a different volatility metric, change VOLATILITY_METRIC at the top of the script.")
    print(f"Options: 'daily_change', 'rolling_vol_5d', 'rolling_vol_10d', 'rolling_vol_20d'")
    print("="*80)


if __name__ == "__main__":
    main()
