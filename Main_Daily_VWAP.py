import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# 1. Database Connection Setup
# Replace with your actual credentials
connection_string = "mssql+pyodbc://@BEELINK/Stock?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
engine = create_engine(connection_string)

def calculate_and_save_weekly_vwap():
    # 2. Fetch Daily OHLC Data
    query = """
    SELECT symbol, [Time] as [Date], [High], [Low], [Close], [Volume]
    FROM ai_stock_prices
    ORDER BY symbol, [Date]
    """
    df = pd.read_sql(query, engine)
    
    # Ensure Date is datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 3. Calculate Typical Price
    df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['TypicalPrice'] * df['Volume']
    
    # 4. Create Weekly Grouping
    # 'W-SUN' identifies the week ending on Sunday, meaning the week resets on Monday
    df['Week_ID'] = df.groupby('symbol')['Date'].transform(lambda x: x.dt.to_period('W-SUN'))
    
    # 5. Calculate Cumulative Sums within each Symbol and Week
    # This creates the "Daily Running Total" for that specific week
    grouped = df.groupby(['symbol', 'Week_ID'])
    
    df['Cumulative_TPV'] = grouped['TPV'].cumsum()
    df['Cumulative_Vol'] = grouped['Volume'].cumsum()
    
    # 6. Calculate Weekly VWAP
    df['Weekly_VWAP'] = df['Cumulative_TPV'] / df['Cumulative_Vol']
    
    # 7. Prepare for Export
    # Selecting only the columns needed for the indicator table
    output_df = df[['symbol', 'Date', 'Weekly_VWAP']].copy()
    
    # 8. Upsert/Save to ai_stock_indicator
    # 'append' adds to the table; 'replace' would drop the table first
    try:
        output_df.to_sql('ai_stock_VWAP', engine, if_exists='append', index=False)
        print("Weekly VWAP calculated and saved successfully.")
    except Exception as e:
        print(f"Error saving to database: {e}")

def update_recent_weekly_vwap():
    # 2. Logic to find the most recent Monday
    # We look back 14 days to ensure we have the start of the current week 
    # even if today is Monday morning.
    lookback_date = (datetime.now() - timedelta(days=140)).strftime('%Y-%m-%d')

    # 3. Fetch limited data from SQL
    # Filtering at the database level saves RAM and Network I/O
    query = f"""
    SELECT symbol, [Time] as [Date], [High], [Low], [Close], [Volume]
    FROM ai_stock_prices
    WHERE [Time] >= '{lookback_date}'
    ORDER BY symbol, [Date]
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("No recent data found.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    
    # 4. VWAP Components
    df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['TypicalPrice'] * df['Volume']
    
    # 5. Group by Symbol and Week
    # 'W-MON' ensures the "bucket" resets every Monday
    df['Week_ID'] = df.groupby('symbol')['Date'].transform(lambda x: x.dt.to_period('W-MON'))
    
    # 6. Cumulative totals for the current week only
    grouped = df.groupby(['symbol', 'Week_ID'])
    df['Cumulative_TPV'] = grouped['TPV'].cumsum()
    df['Cumulative_Vol'] = grouped['Volume'].cumsum()
    
    df['Weekly_VWAP'] = df['Cumulative_TPV'] / df['Cumulative_Vol']
    
    # 7. Filter for only the most recent records (e.g., today's or yesterday's results)
    # This prevents overwriting historical data if you are only running a daily update
    latest_date = df['Date'].max()
    output_df = df[df['Date'] == latest_date][['symbol', 'Date', 'Weekly_VWAP']]
    
    # 8. Save to ai_stock_indicator
    try:
        # We use 'append' for daily updates
        output_df.to_sql('ai_stock_VWAP', engine, if_exists='append', index=False)
        print(f"Successfully updated Weekly VWAP for {latest_date.date()}")
    except Exception as e:
        print(f"Error: {e}")


# def calculate_backtest_vwap_30d():
#     # 1. Define Windows
#     # We pull 40 days of price data to ensure the 'Running Total' 
#     # for the first week of the backtest is accurate.
#     backtest_window = 30
#     buffer_days = 40 
    
#     start_date = (datetime.now() - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
#     cutoff_date = (datetime.now() - timedelta(days=backtest_window)).strftime('%Y-%m-%d')

#     # 2. Fetch Data
#     query = f"""
#     SELECT symbol, [Time] as [Date], [High], [Low], [Close], [Volume]
#     FROM ai_stock_prices
#     WHERE [Time] >= '{start_date}'
#     and symbol = 'AAPL'  -- Example for a single symbol; remove or modify as needed
#     ORDER BY symbol, [Date]
#     """
#     df = pd.read_sql(query, engine)
    
#     if df.empty:
#         return "No data found for the specified range."

def calculate_backtest_vwap_30d():
    backtest_window = 30
    buffer_days = 45 # Slightly more buffer to ensure we hit a Monday
    
    start_date = (datetime.now() - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    cutoff_date = (datetime.now() - timedelta(days=backtest_window)).strftime('%Y-%m-%d')

    # 1. Fetch Data
    query = f"SELECT symbol, [Time] as [Date], [High], [Low], [Close], [Volume] FROM ai_stock_prices WHERE [Time] >= '{start_date}' ORDER BY symbol, [Time]"
    df = pd.read_sql(query, engine)
    
    if df.empty: return "No data"

    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Calculation
    df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['TypicalPrice'] * df['Volume']
    
    # We use the Period for grouping, but we won't export it
    df['Week_Group'] = df.groupby('symbol')['Date'].transform(lambda x: x.dt.to_period('W-MON'))
    
    grouped = df.groupby(['symbol', 'Week_Group'])
    df['Cumulative_TPV'] = grouped['TPV'].cumsum()
    df['Cumulative_Vol'] = grouped['Volume'].cumsum()
    
    # Convert to float to ensure SQL compatibility
    df['Weekly_VWAP'] = (df['Cumulative_TPV'] / df['Cumulative_Vol']).astype(float)
    
    # 3. CRITICAL STEP: Filter columns and rows
    # We create a clean copy containing ONLY the three target columns
    final_df = df[df['Date'] >= cutoff_date][['symbol', 'Date', 'Weekly_VWAP']].copy()
    
    # Ensure no hidden 'Period' index or columns remain
    final_df = final_df.reset_index(drop=True)

    # 4. Save to Database
    try:
        # Clear existing data for the backtest range to avoid primary key/duplicate errors
        with engine.begin() as conn:
            conn.execute(text(f"DELETE FROM ai_stock_VWAP WHERE [Date] >= '{cutoff_date}'"))
        
        # Upload
        final_df.to_sql('ai_stock_VWAP', engine, if_exists='append', index=False)
        print(f"Successfully uploaded {len(final_df)} rows of VWAP data.")
        
    except Exception as e:
        print(f"Error during SQL upload: {e}")



if __name__ == "__main__":
    update_recent_weekly_vwap()
