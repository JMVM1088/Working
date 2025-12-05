import yfinance as yf
import Util
import pandas as pd
import pyodbc
import datetime


# Example usage:
start_date = datetime.date(2024, 1, 1)
end_date = datetime.date(2025, 1, 1)
connection_string = (
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        r'SERVER=BEELINK;'  # Replace with your server name
        r'DATABASE=Stock;' # Replace with your database name
        r'Trusted_Connection=yes;'
    )
result_pairs = Util.get_business_days_decrement(start_date, end_date, 10)
conn = pyodbc.connect(connection_string)
Start_datetime = datetime.now()
# Print the result in descending order
print("Date pairs with 10-business-day decrement (descending order):")
for begin_date, end_date in reversed(result_pairs):
    print(f"Begin Date: {begin_date.strftime('%Y-%m-%d')}, End Date: {end_date.strftime('%Y-%m-%d')}")
     
    ##stock_list = list(Util.get_data_from_sql(connection_string, f"SELECT symbol FROM [Stock].[dbo].[StockList_US]"))
    stock_list = list(Util.get_data_from_sql(connection_string, "exec stock..sp_GetStockList"))

    for ticker in stock_list:
                df = Util.get_prices(ticker[0], 8)
                if df is not None and not df.empty:
                    Util.insert_prices_sql(conn, df, 'Historical_US')
        
    conn.close()
    # Print the complete datetime object
    print("Start date and time:", Start_datetime)
    print("End date and time:", datetime.now())
    print("✅ All data inserted into the database.")    
