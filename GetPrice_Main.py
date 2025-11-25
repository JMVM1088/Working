import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

DB_FILE = r'C:\Users\jv2mk\OneDrive\Stock\Screener\DB\stage2'
TABLE_NAME = 'HistoricalPrices'

def convert_date_format(df):
    # Convert the index or Date column to yyyymmdd format, ensure it's first column
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df['Date'] = df.index.strftime('%Y%m%d')
    # Move Date to first column
    cols = ['Date'] + [c for c in df.columns if c != 'Date']
    df = df[cols]
    return df

def get_last_year_prices(ticker):
    end_date = datetime.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=10)
    print(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date.strftime('%Y-%m-%d'),
                       end=end_date.strftime('%Y-%m-%d'),
                       interval='1d')
    if not df.empty:
        df = convert_date_format(df)
        df['Ticker'] = ticker
        # Use only standard columns for SQL insert (add more if your table has them)
        fields = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        for col in fields:
            if col not in df.columns:
                df[col] = None
        df = df[fields]
    return df

def create_table_if_not_exists(conn):
    create_sql = f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Date TEXT,
            Ticker TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume REAL,
            Dividends REAL,
            [Stock Splits] REAL,
            PRIMARY KEY (Ticker, Date)
        )
    '''
    conn.execute(create_sql)
    conn.commit()

def insert_prices(conn, df,table_name):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        # Existence check to avoid duplicate (uses PK, or add WHERE clause)
        cursor.execute(
            f'SELECT 1 FROM {table_name} WHERE Ticker=? AND Date=?',
            (row['Ticker'], row['Date'])
        )
        if cursor.fetchone() is None:
            # Insert new row
            cursor.execute(
                f'''INSERT INTO {table_name}
                    (Date, Ticker, Open, High, Low, Close, Volume, Dividends, [Stock Splits])
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (row['Date'], row['Ticker'], row['Open'], row['High'], row['Low'],
                 row['Close'], row['Volume'], row['Dividends'], row['Stock Splits'])
            )
    conn.commit()
    print(f"  Inserted {df.shape[0]} rows for {df['Ticker'].iloc[0]} (skipped duplicates)")

def get_data_from_sqlite(db_file, query, params=None):
    """
    Retrieves data from an SQLite database and returns it as a list of tuples.

    Args:
        db_file (str): The path to the SQLite database file.
        query (str): The SQL SELECT query to execute.
        params (tuple, optional): Parameters to substitute into the query. 
                                  Defaults to None.

    Returns:
        list: A list of tuples, where each tuple represents a row of data.
              Returns an empty list if no data is found or an error occurs.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        data = list(cursor.fetchall())
        return data
    except sqlite3.Error as e:
        print(f"Error accessing SQLite database: {e}")
        return []
    finally:
        if conn:
            conn.close()    

def main():
    #stock_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA','SPY','QQQ','SOXX','LEU']  # Edit as needed


    conn = sqlite3.connect(DB_FILE)
    
    # ETF
    table_name = 'Historical_ETF'
    stock_list = list(get_data_from_sqlite(DB_FILE, "SELECT Ticker from stocklist where CompositeName = 'ETF' order by Ticker"))
    # for ticker in stock_list:
    #     ticker = ticker[0]
    #     print(f"Processing ticker: {ticker}")
    # create_table_if_not_exists(conn)
    for ticker in stock_list:
        df = get_last_year_prices(ticker[0])
        if df is not None and not df.empty:
            insert_prices(conn, df, table_name)
    

    # Dow Jones Industrial Average
    table_name = 'Historical_Dow'
    stock_list = list(get_data_from_sqlite(DB_FILE, "SELECT Ticker from stocklist where CompositeName = 'Dow' order by Ticker"))
    # for ticker in stock_list:
    #     ticker = ticker[0]
    #     print(f"Processing ticker: {ticker}")
    # create_table_if_not_exists(conn)
    for ticker in stock_list:
        df = get_last_year_prices(ticker[0])
        if df is not None and not df.empty:
            insert_prices(conn, df, table_name)
    # Nasdaq Composite
    table_name = 'Historical_Nasdaq'
    stock_list = list(get_data_from_sqlite(DB_FILE, "SELECT Ticker from stocklist where CompositeName = 'Nasdaq' order by Ticker"))
    # for ticker in stock_list:
    #     ticker = ticker[0]
    #     print(f"Processing ticker: {ticker}")
    # create_table_if_not_exists(conn)
    for ticker in stock_list:
        df = get_last_year_prices(ticker[0])
        if df is not None and not df.empty:
            insert_prices(conn, df, table_name)

    # SP 500
    table_name = 'Historical_SP500'
    stock_list = list(get_data_from_sqlite(DB_FILE, "SELECT Ticker from stocklist where CompositeName = 'SP500' order by Ticker"))
    # for ticker in stock_list:
    #     ticker = ticker[0]
    #     print(f"Processing ticker: {ticker}")
    # create_table_if_not_exists(conn)
    for ticker in stock_list:
        df = get_last_year_prices(ticker[0])
        if df is not None and not df.empty:
            insert_prices(conn, df, table_name)



    conn.close()
    print("✅ All data inserted into the database.")

if __name__ == "__main__":
    main()
