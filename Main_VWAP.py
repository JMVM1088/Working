import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
import urllib

# 1. Database Configuration
# Update these parameters for your SQL Server environment
SERVER = 'BEELINK'
DATABASE = 'Stock'
DRIVER = 'ODBC Driver 17 for SQL Server'
# If using Windows Auth, UID and PWD are not needed in the connection string
connection_string = f'DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
params = urllib.parse.quote_plus(connection_string)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

def get_stock_list():
    """Fetches the list of tickers from the SQL Server."""
    query = "SELECT symbol as Ticker FROM Stock..AI_Stock_Info WHERE Symbol in ('AAPL','MSFT','GOOGL')"
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df['Ticker'].tolist()

def calculate_vwap(df):
    """Calculates intraday VWAP with flattened columns."""
    # Ensure we are working with a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Calculate Typical Price
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = df['Typical_Price'] * df['Volume']
    
    # Reset cumulative sums daily
    df['Date_Only'] = df.index.date
    
    # Calculate Cumulative sums
    df['Cum_TP_Vol'] = df.groupby('Date_Only')['TP_Vol'].transform('cumsum')
    df['Cum_Vol'] = df.groupby('Date_Only')['Volume'].transform('cumsum')
    
    df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
    
    return df[['Close', 'Volume', 'VWAP']]
def calculate_vwap_Old(df):
    """Calculates intraday VWAP based on 5-minute intervals."""
    # Typical Price = (H + L + C) / 3
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = df['Typical_Price'] * df['Volume']
    
    # Cumulative Sums reset daily (grouping by date)
    df['Date_Only'] = df.index.date
    groups = df.groupby('Date_Only')
    
    df['Cumulative_TP_Vol'] = groups['TP_Vol'].cumsum()
    df['Cumulative_Vol'] = groups['Volume'].cumsum()
    
    df['VWAP'] = df['Cumulative_TP_Vol'] / df['Cumulative_Vol']
    return df[['Close', 'Volume', 'VWAP']]

def main_Old():
    tickers = get_stock_list()
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        
        # 2. Get 5m data (limit to last 60 days for intraday)
        data = yf.download(ticker, period="5d", interval="5m")
        
        if data.empty:
            continue
            
        # 3. Calculate VWAP
        vwap_data = calculate_vwap(data)
        vwap_data['Ticker'] = ticker
        
        # 4. Save back to SQL Server
        # 'append' adds new data; ensure your table schema matches
        vwap_data.to_sql('Stock_VWAP_Results', engine, if_exists='append', index=True)
def main():
    tickers = get_stock_list()
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        
        # FIX: Use auto_adjust=True and then flatten the columns
        data = yf.download(ticker, period="5d", interval="5m", auto_adjust=True)
        
        if data.empty:
            continue
            
        # If yfinance returns MultiIndex columns, flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        vwap_data = calculate_vwap(data)
        vwap_data.index.name = 'Datetime'
        vwap_data['Ticker'] = ticker
        
        # Save to SQL
        vwap_data.to_sql('Stock_VWAP_Results', engine, if_exists='append', index=True)
if __name__ == "__main__":
    main()