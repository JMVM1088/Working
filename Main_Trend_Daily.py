import pandas as pd
import numpy as np
import urllib
from sqlalchemy import create_engine, text
from datetime import datetime
import Util as Util 

# --- CONFIGURATION (Added Target Logic) ---
CONFIG = {
    "MODE": "SCANNER", 
    "START_DATE": "2025-01-01",
    "END_DATE": "2025-12-31",
    "BENCHMARK": "SPY",
    "SQL": {
        "SERVER": "BEELINK",
        "DB": "Stock",
        "DRIVER": "ODBC Driver 17 for SQL Server"
    },
    "ATR_MULT": 3.0,
    "VOL_MULT": 1.5,
    "R_TARGET": 3.0  # Sell at 3x the initial risk
}

def get_engine():
    conn_str = (
        f"DRIVER={{{CONFIG['SQL']['DRIVER']}}};"
        f"SERVER={CONFIG['SQL']['SERVER']};"
        f"DATABASE={CONFIG['SQL']['DB']};"
        f"Trusted_Connection=yes;"
    )
    quoted_conn = urllib.parse.quote_plus(conn_str)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quoted_conn}")

engine = get_engine()

# --- HELPER FUNCTIONS ---

def check_if_open(ticker, engine):
    """Checks if ticker is currently in the OpenPositions table."""
    query = text("SELECT COUNT(*) FROM OpenPositions WHERE Ticker = :t")
    with engine.connect() as conn:
        return conn.execute(query, {"t": ticker}).scalar() > 0

def record_new_entry(ticker, price, atr, ticker_type, engine):
    """Calculates risk/reward and inserts into OpenPositions."""
    initial_stop = price - (CONFIG['ATR_MULT'] * atr)
    initial_risk = price - initial_stop
    target_price = price + (initial_risk * CONFIG['R_TARGET'])
    
    with engine.begin() as conn:
        sql = text("""
            INSERT INTO OpenPositions (Ticker, TickerType, EntryDate, EntryPrice, CurrentStop, InitialRisk, TargetPrice, R_Multiple_Target)
            VALUES (:t, :st, :ed, :ep, :cs, :ir, :tp, :rm)
        """)
        conn.execute(sql, {
            "t": ticker, "st": ticker_type, "ed": datetime.now().date(),
            "ep": price, "cs": initial_stop, "ir": initial_risk, "tp": target_price, "rm": CONFIG['R_TARGET']
        })
    print(f"📈 POSITION OPENED: {ticker} at {price:.2f} | Stop: {initial_stop:.2f} | Target: {target_price:.2f}")

def manage_open_positions(engine):
    """Core logic for HOLD vs EXIT based on current stops and targets."""
    print("\n--- MANAGING OPEN POSITIONS ---")
    open_pos_df = pd.read_sql("""SELECT [PositionID]
      ,[Ticker]
      ,[TIckerType] as SymbolType
      ,[EntryDate]
      ,[EntryPrice]
      ,[CurrentStop]
      ,[DateUpdated]
      ,[InitialRisk]
      ,[TargetPrice]
      ,[R_Multiple_Target]
  FROM [Stock].[dbo].[OpenPositions]
""", engine)
    
    if open_pos_df.empty:
        print("No active trades to manage.")
        return

    for _, pos in open_pos_df.iterrows():
        ticker = pos['Ticker']
        table = "AI_stock_Prices" if pos['SymbolType'] == 'Stock' else "AI_ETF_Prices"
        date_col = "[Time]" if pos['SymbolType'] == 'Stock' else "[TradeDate]"
        
        df = get_data(table, date_col, ticker, "2024-01-01", datetime.now().strftime('%Y-%m-%d'))
        if df.empty: continue
        
        latest = df.iloc[-1]
        curr_price = latest['Close']
        
        # 1. Check Target Hit
        if pos['TargetPrice'] and curr_price >= pos['TargetPrice']:
            print(f"🎯 TARGET HIT: Closing {ticker} at {curr_price:.2f}")
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM OpenPositions WHERE Ticker = :t"), {"t": ticker})
            continue

        # 2. Update/Check Trailing Stop
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        latest_atr = tr.rolling(14).mean().iloc[-1]
        
        new_stop = latest['Low'] - (CONFIG['ATR_MULT'] * latest_atr)
        actual_stop = max(pos['CurrentStop'], new_stop)

        if curr_price < actual_stop:
            print(f"🚨 STOP HIT: Exiting {ticker} at {curr_price:.2f}")
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM OpenPositions WHERE Ticker = :t"), {"t": ticker})
        else:
            print(f"✅ HOLD {ticker}: Price {curr_price:.2f} | Stop {actual_stop:.2f}")
            with engine.begin() as conn:
                conn.execute(text("UPDATE OpenPositions SET CurrentStop = :s WHERE Ticker = :t"), {"s": actual_stop, "t": ticker})

# --- DATA PROCESSING (Provided by you) ---

def get_data(table, dateField, ticker, start, end):
    query = f"SELECT {dateField} as Date, [Open], [High], [Low], [Close], [Volume] FROM {table} WHERE Symbol = '{ticker}' AND {dateField} <= '{end}' ORDER BY Date ASC"
    df = pd.read_sql(query, engine)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def apply_breitstein_logic_Old(df, spy_df):
    if len(df) < 200: return df
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['VolRatio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['RS'] = df['Close'].pct_change(5) - spy_df['Close'].pct_change(5)
    df['Signal'] = ((df['High'] > df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1)) & (df['Close'] > df['VWAP']) & (df['Close'] > df['SMA20']) & (df['Close'] > df['SMA200']) & (df['VolRatio'] >= CONFIG['VOL_MULT']) & (df['RS'] > 0)).fillna(False)
    return df
def apply_breitstein_logic(df, spy_df, is_mega_cap=False):
    if len(df) < 200: return df
    
    # --- 1. Dynamic Parameter Setting ---
    # Mega-caps rarely hit 1.5x volume; 1.1x-1.2x is often a significant institutional move.
    vol_threshold = 1.1 if is_mega_cap else CONFIG['VOL_MULT']
    
    # Mega-caps move the SPY, so we use a 20-day RS to find true decoupling
    # while mid-caps use a 5-day RS for "explosive" relative strength.
    rs_window = 20 if is_mega_cap else 5
    
    # --- 2. Technical Indicators ---
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # ATR for Trailing Stop
    tr = pd.concat([df['High']-df['Low'], 
                    (df['High']-df['Close'].shift(1)).abs(), 
                    (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # --- 3. Dynamic Filters ---
    df['VolRatio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['RS'] = df['Close'].pct_change(rs_window) - spy_df['Close'].pct_change(rs_window)
    
    # --- 4. Signal Logic ---
    # We allow Mega-caps a bit more "room" by checking SMA50 if they are slightly under SMA20
    sma_filter = df['Close'] > df['SMA20']
    if is_mega_cap:
        df['SMA50'] = df['Close'].rolling(50).mean()
        sma_filter = (df['Close'] > df['SMA20']) | (df['Close'] > df['SMA50'])

    df['Signal'] = (
        (df['High'] > df['High'].shift(1)) & 
        (df['Low'] > df['Low'].shift(1)) & 
        (df['Close'] > df['VWAP']) & 
        sma_filter &             
        (df['Close'] > df['SMA200']) & 
        (df['VolRatio'] >= vol_threshold) & 
        (df['RS'] > 0)                                                       
    ).fillna(False)
    
    return df
# --- EXECUTION FLOW ---

def main():
    spy_data = get_data("AI_ETF_Prices", "TradeDate", CONFIG['BENCHMARK'], "2020-01-01", CONFIG['END_DATE'])
    #MEGA_CAPS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'LLY', 'V']
    conStr = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={CONFIG['SQL']['SERVER']};DATABASE={CONFIG['SQL']['DB']};Trusted_Connection=yes;"
    MEGA_CAPS = list(Util.get_data_from_sql(conStr, "SELECT Symbol FROM AI_Stock_Info s (nolock) WHERE s.[Index] like '%S&P 500%' "))  # 200B+
    # 1. Manage existing inventory first
    if CONFIG['MODE'] == "SCANNER":
        manage_open_positions(engine)

    # 2. Scan for new entries
    for asset_type in ['US', 'ETF']:
        print(f"\nScanning {asset_type} list...")
        table = "AI_stock_Prices" if asset_type == 'US' else "AI_ETF_Prices"
        date_col = "[Time]" if asset_type == 'US' else "[TradeDate]"
        
        tickers = list(Util.get_data_from_sql(conStr, f"exec stock..sp_GetStockList '{asset_type}'"))
        
        for t in tickers:
            try:
                data = get_data(table, date_col, t[0], "2020-01-01", CONFIG['END_DATE'])
                is_mega = t[0] in MEGA_CAPS
                if len(data) < 200: continue
                processed = apply_breitstein_logic(data, spy_data,is_mega )
                
                latest = processed.iloc[-1]
                if latest['Signal']:
                    # Use record_new_entry only if we don't already own it
                    if not check_if_open(t[0], engine):
                        record_new_entry(t[0], latest['Close'], latest['ATR'], "Stock" if asset_type == 'US' else "ETF", engine)
                    else:
                        print(f"Skipping {t[0]}: Signal active but already in OpenPositions.")
            except Exception as e:
                print(f"Error {t[0]}: {e}")

if __name__ == "__main__":
    main()