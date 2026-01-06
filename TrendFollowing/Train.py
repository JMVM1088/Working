import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime
import Util as Util

CONFIG = {
    "MODE": "BACKTEST", # Options: 'SCANNER' or 'BACKTEST'
    "START_DATE": "2025-01-01",
    "END_DATE": "2025-12-31",
    "TICKERS": ["HOOD", "GLD", "AAPL", "AMD", "MSFT"],
    "BENCHMARK": "SPY",
    "SQL": {
        "SERVER": "BEELINK",
        "DB": "Stock",
        "UID": "your_user",
        "PWD": "your_password"
    },
    "ATR_MULT": 3.0,
    "VOL_MULT": 1.5
}
conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={CONFIG['SQL']['SERVER']};DATABASE={CONFIG['SQL']['DB']};Trusted_Connection=yes;"
def get_conn():
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={CONFIG['SQL']['SERVER']};DATABASE={CONFIG['SQL']['DB']};Trusted_Connection=yes;"
    #conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={CONFIG['SQL']['SERVER']};DATABASE={CONFIG['SQL']['DB']};Trusted_Connection=yes;}"

    return pyodbc.connect(conn_str)

def get_data(table, dateField, ticker, start, end):
    with get_conn() as conn:
        # Fetching extra history for indicators (SMA200 requires ~300 days prior)
        query = f"SELECT {dateField} as Date, [Open], [High], [Low], [Close], [Volume] FROM {table} WHERE Symbol = '{ticker}' AND {dateField} <= '{end}' ORDER BY Date ASC"
        df = pd.read_sql(query, conn)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

def apply_breitstein_logic(df, spy_df):
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
        (df['High'] > df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1)) & # Rule 1 & 4
        (df['Close'] > df['VWAP']) & (df['Close'] > df['SMA20']) &             # Rule 2 & 3
        (df['Close'] > df['SMA200']) & (df['VolRatio'] >= CONFIG['VOL_MULT']) & # Rule 5 & Vol
        (df['RS'] > 0)                                                        # RS
    )
    return df

def run_backtest(df, ticker,ticker_type="Stock"):
    trades = []
    in_pos = False
    entry_price, entry_date, stop_price = 0, None, 0
    
    # Only backtest within specified date range
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
                trades.append((ticker, entry_date, row['Date'], entry_price, row['Close'], pnl))
                in_pos = False
    
    # Save to SQL
    if trades:
        with get_conn() as conn:
            trades['TickerType'] = ticker_type
            cursor = conn.cursor()
            cursor.executemany("INSERT INTO BacktestTrades (Ticker, EntryDate, ExitDate, EntryPrice, ExitPrice, PnL_Pct) VALUES (?, ?, ?, ?, ?, ?,?)", trades)
    print(f"Backtest: Saved {len(trades)} trades for {ticker}")

def run_scanner(df, ticker):
    latest = df.iloc[-1]
    if latest['Signal']:
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO TradingHotList (ScanDate, Ticker, Price, VolRatio, RS_Score, MarketTheme) VALUES (?, ?, ?, ?, ?, ?)",
                           (latest['Date'], ticker, latest['Close'], latest['VolRatio'], latest['RS'], "Automated Trend Filter"))
        print(f"Scanner: 🔥 {ticker} added to Hot List!")

# Execution
spy_data = get_data("AI_ETF_Prices","TradeDate",CONFIG['BENCHMARK'], "2020-01-01", CONFIG['END_DATE'])
#results_to_upload = []
stock_list = list(Util.get_data_from_sql(conn_str, "exec stock..sp_GetStockList 'US'"))

for ticker in stock_list:
    data = get_data("AI_stock_Prices","[Time]",ticker[0], "2020-01-01", CONFIG['END_DATE'])
    if len(data) < 200: continue
    
    processed = apply_breitstein_logic(data, spy_data)
    
    if CONFIG['MODE'] == "BACKTEST":
        run_backtest(processed, ticker[0],"Stock")
    else:
        run_scanner(processed, ticker[0])

## ETF Processing
etf_list = list(Util.get_data_from_sql(conn_str, "exec stock..sp_GetStockList 'ETF'"))

for ticker in etf_list:
    data = get_data("AI_ETF_Prices","[Time]",ticker[0], "2020-01-01", CONFIG['END_DATE'])
    if len(data) < 200: continue
    
    processed = apply_breitstein_logic(data, spy_data)
    
    if CONFIG['MODE'] == "BACKTEST":
        run_backtest(processed, ticker[0],"ETF")
    else:
        run_scanner(processed, ticker[0])