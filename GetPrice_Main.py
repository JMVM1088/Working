import yfinance as yf
import Util
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

DB_FILE = r'C:\Users\jv2mk\OneDrive\Stock\Screener\DB\stage2'
TABLE_NAME = 'HistoricalPrices'

def main():
    conn = sqlite3.connect(DB_FILE)
    
    # Create a list of table configurations
    tables = [
        {'table_name': 'Historical_ETF', 'db_table': 'ETF'},
        {'table_name': 'Historical_Dow', 'db_table': 'Dow'},
        {'table_name': 'Historical_Nasdaq', 'db_table': 'Nasdaq'},
        {'table_name': 'Historical_SP500', 'db_table': 'SP500'},
    ]
    
    # Loop through each table configuration
    for config in tables:
        table_name = config['table_name']
        db_table = config['db_table']
        
        # Handle SP500 special case
        composite_name = f"{db_table}0" if db_table == 'SP500' else db_table
        
        stock_list = list(Util.get_data_from_sqlite(DB_FILE, f"SELECT Ticker from stocklist where CompositeName = '{composite_name}' order by Ticker"))
        
        for ticker in stock_list:
            df = Util.get_prices(ticker[0], 1)
            if df is not None and not df.empty:
                Util.insert_prices(conn, df, table_name)
    
    conn.close()
    print("✅ All data inserted into the database.")    

if __name__ == "__main__":
    main()
