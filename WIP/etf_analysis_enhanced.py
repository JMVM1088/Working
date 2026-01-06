"""
ETF Drop Buy Strategy - Fully Robust Version
-------------------------------------------
Features:
- Multi-threshold backtesting
- Multi-target return analysis
- Market condition analysis
- 100% robust yfinance handling (no more missing columns / alignment errors)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
TICKERS = ['SPY', 'QQQ','SOXX','XLF','nvda']
DROP_THRESHOLDS = [-0.015, -0.02, -0.025,-0.03]
TARGET_RETURNS = [0.05, 0.075, 0.1]
MA_PERIOD = 200
LOOKBACK_YEARS = 1
MAX_HOLDING_DAYS = 252

# ============================================================================
# DEBUG PRINT
# ============================================================================
def debug(msg):
    print(msg)

# ============================================================================
# COLUMN REPAIR FUNCTIONS
# ============================================================================
def flatten_columns(df):
    """Fix yfinance MultiIndex → flatten into single layer"""
    if isinstance(df.columns, pd.MultiIndex):
        debug("  🔧 Detected MultiIndex columns, flattening...")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def normalize_columns(df):
    """Normalize, fix aliases, patch missing columns"""
    df = df.copy()
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    rename_alias = {
        "adjclose": "adj_close",
        "adjusted_close": "adj_close",
        "close": "close",
    }
    df = df.rename(columns=rename_alias)

    # Missing close → emergency fallback
    if "close" not in df.columns:
        raise ValueError("Downloaded data contains no 'Close' column.")

    # Missing adj_close → fallback to close
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # Ensure essential OHLCV exist (fill forward where possible)
    essential = ["open", "high", "low", "close", "adj_close"]
    for col in essential:
        if col not in df.columns:
            debug(f"  ⚠️ Missing column {col}, filling using close.")
            df[col] = df["close"]

    return df

# ============================================================================
# DATA DOWNLOAD + PREPARATION
# ============================================================================
def download_and_prepare(ticker, start_date, end_date):
    debug(f"\n📥 Downloading: {ticker}")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        debug(f"  ❌ Download failed: {e}")
        return None

    if df is None or df.empty:
        debug("  ⚠️ Empty DataFrame returned.")
        return None

    # Apply robust fixes
    df = flatten_columns(df)
    df = normalize_columns(df)

    # Compute indicators
    df["daily_return"] = df["close"].pct_change()
    df["ma_200"] = df["close"].rolling(window=MA_PERIOD).mean()

    # Align before comparing
    close_aligned, ma_aligned = df["close"].align(df["ma_200"])
    df["above_ma"] = close_aligned > ma_aligned

    df["volatility_20d"] = df["daily_return"].rolling(20).std()
    df = df.dropna(subset=["ma_200"])

    debug(f"  ✓ Prepared {len(df)} rows")
    return df

# ============================================================================
# BACKTEST FUNCTION
# ============================================================================
def backtest_multi_threshold(df, ticker):
    all_scenarios = []

    for drop_threshold in DROP_THRESHOLDS:
        for target_return in TARGET_RETURNS:

            daily_ret, above_ma = df["daily_return"].align(df["above_ma"], axis=0)
            signals = df[(daily_ret <= drop_threshold) & (above_ma == True)]

            if len(signals) == 0:
                continue

            trades = []

            for buy_date, row in signals.iterrows():
                entry = row["close"]
                target_price = entry * (1 + target_return)

                future = df.loc[buy_date:].iloc[1:]

                days_to_target = None
                exit_date = None
                max_return = -999

                for d, (d2, r2) in enumerate(future.iterrows(), 1):
                    if r2["high"] >= target_price and days_to_target is None:
                        days_to_target = d
                        exit_date = d2

                    ret = (r2["close"] - entry) / entry
                    max_return = max(max_return, ret)

                    if d >= MAX_HOLDING_DAYS:
                        break

                trades.append({
                    "Drop_Date": buy_date,
                    "Buy_Date": buy_date,
                    "Entry_Price": entry,
                    "Target_Price": target_price,
                    "Days_to_Target": days_to_target,
                    "Exit_Date": exit_date,
                    "Target_Reached": days_to_target is not None,
                    "Max_Return": max_return,
                    "Daily_Drop": row["daily_return"],
                    "Volatility": row["volatility_20d"],
                })

            trades_df = pd.DataFrame(trades)

            scenario = {
                "Ticker": ticker,
                "Drop_Threshold_%": drop_threshold * 100,
                "Target_Return_%": target_return * 100,
                "Total_Signals": len(trades_df),
                "Successful_Trades": trades_df["Target_Reached"].sum(),
                "Success_Rate_%": trades_df["Target_Reached"].mean() * 100,
                "Avg_Days_to_Target": trades_df[trades_df["Target_Reached"]]["Days_to_Target"].mean(),
                "Avg_Max_Return_Failed_%": trades_df[~trades_df["Target_Reached"]]["Max_Return"].mean() * 100,
                "Drop_Dates": "; ".join(trades_df["Drop_Date"].astype(str).tolist()),
            }

            all_scenarios.append(scenario)

    return pd.DataFrame(all_scenarios)

# ============================================================================
# MARKET CONDITION ANALYSIS
# ============================================================================
def analyze_market_conditions(df, ticker):
    df["Market_Trend"] = "Neutral"
    df.loc[df["close"] > df["close"].shift(20), "Market_Trend"] = "Uptrend"
    df.loc[df["close"] < df["close"].shift(20), "Market_Trend"] = "Downtrend"

    signals = df[(df["daily_return"] <= -0.015) & (df["above_ma"] == True)]

    results = []
    for cond in ["Uptrend", "Neutral", "Downtrend"]:
        subset = signals[signals["Market_Trend"] == cond]
        if len(subset) == 0:
            continue

        success = 0
        for date, row in subset.iterrows():
            future = df.loc[date:].iloc[1:31]
            if future["high"].max() >= row["close"] * 1.30:
                success += 1

        results.append({
            "Market_Condition": cond,
            "Signals": len(subset),
            "Approx_Success_Rate_%": success / len(subset) * 100
        })

    return pd.DataFrame(results)

# ============================================================================
# MAIN
# ============================================================================
def main():
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_YEARS * 365 + 400)

    all_results = []

    for ticker in TICKERS:
        debug("\n" + "█" * 80)
        debug(f"  ANALYZING {ticker}")
        debug("█" * 80)

        df = download_and_prepare(ticker, start, end)
        if df is None:
            continue

        result = backtest_multi_threshold(df, ticker)
        all_results.append(result)

        mc = analyze_market_conditions(df, ticker)
        debug("\nMarket condition impact:")
        print(mc)

    if len(all_results) > 0:
        final = pd.concat(all_results, ignore_index=True)
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"Result/{current_date}_ETF_Drop_backtest.csv"
        final.to_csv(filename, index=False)
        debug(f"\n💾 Saved: {filename}")
        print(final)

if __name__ == "__main__":
    main()
