import pandas as pd
import numpy as np
import urllib
from sqlalchemy import create_engine, text
from datetime import datetime
import Util as Util # Assuming your custom Util module

# 1. Configuration
CONFIG = {
    "MODE": "SCANNER", # Options: 'SCANNER' or 'BACKTEST'
    "START_DATE": "2025-01-01",
    "END_DATE": "2025-12-31",
    "BENCHMARK": "SPY",
    "SQL": {
        "SERVER": "BEELINK",
        "DB": "Stock",
        "DRIVER": "ODBC Driver 17 for SQL Server"
    },
    "ATR_MULT": 3.0,
    "VOL_MULT": 1.5
}

# 2. SQLAlchemy Engine Setup (Avoids UserWarning)
def get_engine():
    # Use Trusted_Connection=yes as per your original connection string
    conn_str = (
        f"DRIVER={{{CONFIG['SQL']['DRIVER']}}};"
        f"SERVER={CONFIG['SQL']['SERVER']};"
        f"DATABASE={CONFIG['SQL']['DB']};"
        f"Trusted_Connection=yes;"
    )
    quoted_conn = urllib.parse.quote_plus(conn_str)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quoted_conn}")

engine = get_engine()

def get_data(table, dateField, ticker, start, end):
    query = f"""
    SELECT {dateField} as Date, [Open], [High], [Low], [Close], [Volume] 
    FROM {table} 
    WHERE Symbol = '{ticker}' AND {dateField} <= '{end}' 
    ORDER BY Date ASC
    """
    df = pd.read_sql(query, engine)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def apply_breitstein_logic(df, spy_df):
    if len(df) < 200: return df
    
    # Standard Indicators
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # ATR for Trailing Stop
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # Filters
    df['VolRatio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['RS'] = df['Close'].pct_change(5) - spy_df['Close'].pct_change(5)
    
    # 5 Rules + Confirmation
    df['Signal'] = (
        (df['High'] > df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1)) & 
        (df['Close'] > df['VWAP']) & (df['Close'] > df['SMA20']) &             
        (df['Close'] > df['SMA200']) & (df['VolRatio'] >= CONFIG['VOL_MULT']) & 
        (df['RS'] > 0)                                                       
    ).fillna(False)
    return df
# 3. Corrected Backtest Function using to_sql
def run_backtest(df, ticker, ticker_type):
    trades_list = []
    in_pos = False
    entry_price, entry_date, stop_price = 0, None, 0
    
    backtest_range = df[(df['Date'] >= CONFIG['START_DATE']) & (df['Date'] <= CONFIG['END_DATE'])]
    
    for i, row in backtest_range.iterrows():
        if not in_pos:
            if row['Signal']:
                in_pos, entry_price, entry_date = True, row['Close'], row['Date']
                stop_price = row['Low'] - (CONFIG['ATR_MULT'] * row['ATR'])
        else:
            # Trailing stop only moves UP
            stop_price = max(stop_price, row['Low'] - (CONFIG['ATR_MULT'] * row['ATR']))
            if row['Close'] < stop_price:
                pnl = (row['Close'] / entry_price) - 1
                trades_list.append({
                    "Ticker": ticker,
                    "TickerType": ticker_type,
                    "EntryDate": entry_date,
                    "ExitDate": row['Date'],
                    "EntryPrice": entry_price,
                    "ExitPrice": row['Close'],
                    "PnL_Pct": pnl
                })
                in_pos = False
    
    if trades_list:
        # Convert list of dictionaries to DataFrame and save to SQL
        trades_df = pd.DataFrame(trades_list)
        trades_df.to_sql("BacktestTrades", engine, if_exists="append", index=False)
        print(f"Backtest: Saved {len(trades_list)} trades for {ticker} ({ticker_type})")

def run_backtest_Old(df, ticker, ticker_type):
    trades = []
    in_pos = False
    entry_price, entry_date, stop_price = 0, None, 0
    
    backtest_range = df[(df['Date'] >= CONFIG['START_DATE']) & (df['Date'] <= CONFIG['END_DATE'])]
    
    for i, row in backtest_range.iterrows():
        if not in_pos:
            if row['Signal']:
                in_pos, entry_price, entry_date = True, row['Close'], row['Date']
                stop_price = row['Low'] - (CONFIG['ATR_MULT'] * row['ATR'])
        else:
            stop_price = max(stop_price, row['Low'] - (CONFIG['ATR_MULT'] * row['ATR']))
            if row['Close'] < stop_price:
                pnl = (row['Close'] / entry_price) - 1
                # Added ticker_type to the tuple
                trades.append((ticker, ticker_type, entry_date, row['Date'], entry_price, row['Close'], pnl))
                in_pos = False
    
    if trades:
        # Using SQLAlchemy connection to execute insert
        with engine.begin() as conn:
            sql = """INSERT INTO BacktestTrades (Ticker, TickerType, EntryDate, ExitDate, EntryPrice, ExitPrice, PnL_Pct) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)"""
            for t in trades:
                conn.execute(text(sql), t)
    print(f"Backtest: Saved {len(trades)} trades for {ticker} ({ticker_type})")

def run_scanner_Old(df, ticker, ticker_type):
    latest = df.iloc[-1]
    if latest['Signal']:
        with engine.begin() as conn:
            sql = """INSERT INTO TradingHotList (ScanDate, Ticker, TickerType, Price, VolRatio, RS_Score, MarketTheme) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)"""
            conn.execute(text(sql), (latest['Date'], ticker, ticker_type, latest['Close'], 
                                     latest['VolRatio'], latest['RS'], "Automated Trend Filter"))
        print(f"Scanner: 🔥 {ticker} ({ticker_type}) added to Hot List!")
# 4. Corrected Scanner Function using to_sql
def run_scanner(df, ticker, ticker_type):
    latest = df.iloc[-1]
    if latest['Signal']:
        scanner_data = pd.DataFrame([{
            "ScanDate": latest['Date'],
            "Ticker": ticker,
            "TickerType": ticker_type,
            "Price": latest['Close'],
            "VolRatio": latest['VolRatio'],
            "RS_Score": latest['RS'],
            "MarketTheme": "Automated Trend Filter"
        }])
        scanner_data.to_sql("TradingHotList", engine, if_exists="append", index=False)
        print(f"Scanner: 🔥 {ticker} ({ticker_type}) added to Hot List!")
# 1. NEW FEATURE: Manage Open Positions
def manage_open_positions(engine):
    print("\n--- MANAGING OPEN POSITIONS (Stop & Target Check) ---")
    
    query = "SELECT * FROM OpenPositions"
    open_pos_df = pd.read_sql(query, engine)
    
    if open_pos_df.empty:
        print("No open positions to manage.")
        return

    for idx, pos in open_pos_df.iterrows():
        ticker = pos['Ticker']
        table = "AI_stock_Prices" if pos['SymbolType'] == 'Stock' else "AI_ETF_Prices"
        
        # Get latest data
        df = get_data(table, "[Time]", ticker, "2024-01-01", datetime.now().strftime('%Y-%m-%d'))
        if df.empty: continue
        
        latest_row = df.iloc[-1]
        current_price = latest_row['Close']
        
        # 1. Check Target Hit (R-Multiple)
        # Target Price was calculated at entry: Entry + (InitialRisk * 3)
        if pos['TargetPrice'] and current_price >= pos['TargetPrice']:
            print(f"🎯 TARGET HIT: {ticker} reached {pos['R_Multiple_Target']}R at {current_price:.2f}!")
            # Logic to close position...
            continue 

        # 2. Check Trailing Stop (Existing Logic)
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), 
                        (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        new_calculated_stop = latest_row['Low'] - (CONFIG['ATR_MULT'] * latest_row['ATR'])
        actual_stop = max(pos['CurrentStop'], new_calculated_stop)

        if current_price < actual_stop:
            print(f"🚨 EXIT (STOP): {ticker} hit stop at {actual_stop:.2f}")
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM OpenPositions WHERE Ticker = :t"), {"t": ticker})
        else:
            print(f"✅ HOLD: {ticker} Price: {current_price:.2f} | Stop: {actual_stop:.2f} | Target: {pos['TargetPrice']:.2f}")
            with engine.begin() as conn:
                conn.execute(text("UPDATE OpenPositions SET CurrentStop = :s WHERE Ticker = :t"), 
                             {"s": actual_stop, "t": ticker})
                
def record_new_entry(ticker, price, atr, ticker_type, engine):
    # Calculate R-Values
    initial_stop = price - (CONFIG['ATR_MULT'] * atr)
    initial_risk = price - initial_stop  # This is 1R
    target_price = price + (initial_risk * 3.0) # 3R Target
    
    with engine.begin() as conn:
        sql = """
        INSERT INTO OpenPositions (Ticker, SymbolType, EntryDate, EntryPrice, CurrentStop, InitialRisk, TargetPrice, R_Multiple_Target)
        VALUES (:t, :st, :ed, :ep, :cs, :ir, :tp, 3.0)
        """
        conn.execute(text(sql), {
            "t": ticker, "st": ticker_type, "ed": datetime.now().date(),
            "ep": price, "cs": initial_stop, "ir": initial_risk, "tp": target_price
        })
# --- EXECUTION ---

# Get Benchmark data
spy_data = get_data("AI_ETF_Prices", "TradeDate", CONFIG['BENCHMARK'], "2020-01-01", CONFIG['END_DATE'])

# Process Stocks
print("Starting Stock Processing...")
stock_list = list(Util.get_data_from_sql(f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={CONFIG['SQL']['SERVER']};DATABASE={CONFIG['SQL']['DB']};Trusted_Connection=yes;", "exec stock..sp_GetStockList 'US'"))
for ticker in stock_list:
    try:
        data = get_data("AI_stock_Prices", "[Time]", ticker[0], "2020-01-01", CONFIG['END_DATE'])
        if len(data) < 200: continue
        processed = apply_breitstein_logic(data, spy_data)
        
        if CONFIG['MODE'] == "BACKTEST":
            run_backtest(processed, ticker[0], "Stock")
        else:
            run_scanner(processed, ticker[0], "Stock")
    except Exception as e:
        print(f"Error processing stock {ticker[0]}: {e}")

# Process ETFs
print("Starting ETF Processing...")
etf_list = list(Util.get_data_from_sql(f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={CONFIG['SQL']['SERVER']};DATABASE={CONFIG['SQL']['DB']};Trusted_Connection=yes;", "exec stock..sp_GetStockList 'ETF'"))
for ticker in etf_list:
    try:
        data = get_data("AI_ETF_Prices", "[TradeDate]", ticker[0], "2020-01-01", CONFIG['END_DATE'])
        if len(data) < 200: continue
        processed = apply_breitstein_logic(data, spy_data)
        
        if CONFIG['MODE'] == "BACKTEST":
            run_backtest(processed, ticker[0], "ETF")
        else:
            run_scanner(processed, ticker[0], "ETF")
    except Exception as e:
        print(f"Error processing ETF {ticker[0]}: {e}")