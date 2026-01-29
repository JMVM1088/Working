import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
import urllib
import datetime

# 1. Database Connection (Reusing your configuration)
SERVER = 'BEELINK'
DATABASE = 'Stock'
connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
params = urllib.parse.quote_plus(connection_string)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

def calculate_current_vwap(ticker):
    """Downloads daily data and returns only the most recent 5min VWAP record."""
    # Pull '1d' to get all bars for today to calculate cumulative VWAP correctly
    data = yf.download(ticker, period="1d", interval="5m", auto_adjust=True)
    
    if data.empty:
        return None
    
    # Flatten MultiIndex if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # VWAP Calculation (Cumulative for the day)
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['TP_Vol'] = data['TP'] * data['Volume']
    data['VWAP'] = data['TP_Vol'].cumsum() / data['Volume'].cumsum()
    
    # Get only the very last completed 5-minute bar
    latest_bar = data.tail(1).copy()
    latest_bar['Ticker'] = ticker
    latest_bar.index.name = 'Datetime'
    
    return latest_bar[['Ticker', 'Close', 'Volume', 'VWAP']]

def upsert_to_sql(df):
    """Inserts or Updates the record to prevent Primary Key errors."""
    if df is None or df.empty:
        return

    # Use a staging table approach for a clean MERGE
    with engine.begin() as conn:
        # 1. Create a temp table
        conn.execute(text("CREATE TABLE #StageVWAP (Datetime DATETIMEOFFSET, Ticker VARCHAR(10), [Close] FLOAT, Volume BIGINT, VWAP FLOAT)"))
        
        # 2. Upload the latest bar to temp table
        df.to_sql('#StageVWAP', conn, if_exists='append', index=True)
        
        # 3. Perform the MERGE (The 'Upsert')
        merge_sql = """
            MERGE Stock_VWAP_Results AS target
            USING #StageVWAP AS source
            ON (target.Ticker = source.Ticker AND target.Datetime = source.Datetime)
            WHEN MATCHED THEN
                UPDATE SET target.[Close] = source.[Close], target.Volume = source.Volume, target.VWAP = source.VWAP
            WHEN NOT MATCHED THEN
                INSERT (Datetime, Ticker, [Close], Volume, VWAP)
                VALUES (source.Datetime, source.Ticker, source.[Close], source.Volume, source.VWAP);
        """
        conn.execute(text(merge_sql))

def main():
    # Get ticker list from your existing table   aad
    with engine.connect() as conn:
        query = "SELECT symbol as Ticker FROM Stock..AI_Stock_Info WHERE Symbol in ('AAPL','MSFT','GOOGL')"
        tickers = pd.read_sql(query, conn)['Ticker'].tolist()

    for ticker in tickers:  
        try:
            print(f"Updating {ticker}...")
            latest_data = calculate_current_vwap(ticker)
            upsert_to_sql(latest_data)
        except Exception as e:
            print(f"Error updating {ticker}: {e}")

if __name__ == "__main__":
    main()