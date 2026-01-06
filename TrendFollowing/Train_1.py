import pandas as pd
import numpy as np
import urllib
from sqlalchemy import create_engine, text
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & MODE TOGGLE
# ==========================================
CONFIG = {
    "MODE": "BACKTEST",         # Switch to 'SCANNER' for today's hits
    "START_DATE": "2024-01-01", # Range for Backtest
    "END_DATE": "2025-12-31",
    "TICKERS": ["NVDA", "TSLA", "AAPL", "AMD", "MSFT"],
    "BENCHMARK": "SPY",
    "SQL": {
        "DRIVER": "ODBC Driver 17 for SQL Server",
        "SERVER": "BEELINK",
        "DB": "Stock",
        "trusted_connection": "Yes"
        
    },
    "ATR_MULT": 3.0,  # 3x ATR Trailing Stop
    "VOL_MULT": 1.5   # 1.5x Relative Volume
}

# ==========================================
# 2. DATABASE ENGINE (FIXES USERWARNING)
# ==========================================
def get_engine():
    params = urllib.parse.quote_plus(
        f"DRIVER={{{CONFIG['SQL']['DRIVER']}}};"
        f"SERVER={CONFIG['SQL']['SERVER']};"
        f"DATABASE={CONFIG['SQL']['DB']};"
        f"trusted_connection={CONFIG['SQL']['trusted_connection']};"
       
    )
    # SQLAlchemy engine for MSSQL + PyODBC
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# ==========================================
# 3. BREITSTEIN CORE LOGIC
# ==========================================
def apply_strategy(df, spy_df):
    # SMAs & VWAP
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # ATR (Recommendation #2)
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # Volume Confirmation (Recommendation #3)
    df['VolRatio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Relative Strength vs SPY (Recommendation #1)
    df['RS_Score'] = df['Close'].pct_change(5) - spy_df['Close'].pct_change(5)
    
    # The "A+" Entry Signal
    df['Entry_Signal'] = (
        (df['High'] > df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1)) & # Structure
        (df['Close'] > df['VWAP']) & (df['Close'] > df['SMA20']) &             # Momentum
        (df['Close'] > df['SMA200']) &                                        # Trend Filter
        (df['VolRatio'] >= CONFIG['VOL_MULT']) &                             # Volume Fuel
        (df['RS_Score'] > 0)                                                 # Market Leadership
    )
    return df

# ==========================================
# 4. MODES: SCANNER & BACKTEST
# ==========================================
def run_backtest(df, ticker, engine):
    trades = []
    in_pos, stop_price, entry_price, entry_date = False, 0, 0, None
    
    # Filter for the user-specified date range
    mask = (df['Date'] >= CONFIG['START_DATE']) & (df['Date'] <= CONFIG['END_DATE'])
    bt_df = df.loc[mask]

    for i, row in bt_df.iterrows():
        if not in_pos:
            if row['Entry_Signal']:
                in_pos, entry_price, entry_date = True, row['Close'], row['Date']
                stop_price = row['Low'] - (CONFIG['ATR_MULT'] * row['ATR'])
        else:
            # Recommendation #2: Update ATR Trailing Stop (High-Water Mark)
            current_stop = row['Low'] - (CONFIG['ATR_MULT'] * row['ATR'])
            stop_price = max(stop_price, current_stop)
            
            if row['Close'] < stop_price:
                pnl = (row['Close'] / entry_price) - 1
                trades.append({
                    "Ticker": ticker, "EntryDate": entry_date, "ExitDate": row['Date'],
                    "EntryPrice": entry_price, "ExitPrice": row['Close'], "PnL_Pct": pnl
                })
                in_pos = False
    
    if trades:
        pd.DataFrame(trades).to_sql("BacktestTrades", engine, if_exists="append", index=False)
    print(f"Backtest Complete: {len(trades)} trades recorded for {ticker}.")

def run_scanner(df, ticker, engine):
    latest = df.iloc[-1]
    if latest['Entry_Signal']:
        # Recommendation #4: The Daily Report Card (Saving to SQL)
        report_card = pd.DataFrame([{
            "ScanDate": latest['Date'],
            "Ticker": ticker,
            "Price": latest['Close'],
            "VolRatio": latest['VolRatio'],
            "RS_Score": latest['RS_Score'],
            "Grade": "A",
            "MarketTheme": "Automated Breitstein A+ Setup"
        }])
        report_card.to_sql("TradingHotList", engine, if_exists="append", index=False)
        print(f"🔥 SCANNER ALERT: {ticker} passed all filters today!")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    engine = get_engine()
    
    # Fetch SPY first for RS Calculation
    spy_query = f"SELECT TradeDate as Date, [Close] FROM AI_ETF_Prices WHERE Symbol='{CONFIG['BENCHMARK']}' ORDER BY Date"
    spy_df = pd.read_sql(spy_query, engine)
    
    for ticker in CONFIG['TICKERS']:
        print(f"Processing {ticker}...")
        query = f"SELECT [Time] as Date, [High], [Low], [Close], [Volume] FROM AI_Stock_prices WHERE Symbol='{ticker}' ORDER BY Date"
        df = pd.read_sql(query, engine)
        
        if len(df) < 200: continue
        processed_df = apply_strategy(df, spy_df)
        
        if CONFIG["MODE"] == "BACKTEST":
            run_backtest(processed_df, ticker, engine)
        else:
            run_scanner(processed_df, ticker, engine)

if __name__ == "__main__":
    main()