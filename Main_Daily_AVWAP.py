import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
import sys

# --- CONFIGURATION ---
DB_STR = "mssql+pyodbc://@BEELINK/Stock?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
engine = create_engine(DB_STR)

# Institutional Settings
PRD = 50             # VWAP Rolling Window
ZONE_LIMIT = 0.985   # 1.5% below VWAP is the floor of our 'Buy Zone'
TREND_REQ = 3        # Require 3 of last 5 days above VWAP

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_production_pipeline():
    try:
        logging.info("Step 1: Ingesting Market Data...")
        # We only pull enough data to calculate our windows (e.g., last 200 days)
        query = "SELECT Symbol, [time], [high], [low], [close], [Volume] FROM AI_stock_prices where [time] >= DateAdd(day, -250, getdate()) ORDER BY Symbol, [time]"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logging.error("No data found in source table.")
            return

        logging.info("Step 2: Calculating Quant Metrics...")
        results = []
        
        for symbol, group in df.groupby('Symbol'):
            group = group.sort_values('time').copy()
            
            # A. Robust VWAP Calculation
            group['hlc3'] = (group['high'] + group['low'] + group['close']) / 3
            group['pv'] = group['hlc3'] * group['Volume']
            
            # Cumulative Rolling sums
            group['cum_pv'] = group['pv'].rolling(PRD, min_periods=10).sum()
            group['cum_vol'] = group['Volume'].rolling(PRD, min_periods=10).sum()
            group['vwap'] = group['cum_pv'] / group['cum_vol']
            
            # B. Trend Scoring (Days above VWAP)
            group['is_above'] = (group['close'] > group['vwap']).astype(int)
            group['trend_score'] = group['is_above'].rolling(5).sum()
            
            # C. Signal Logic (The Pullback)
            # 1. Trend is intact (3/5 days above)
            # 2. Today's Low touches or pierces VWAP
            # 3. Today's Close holds the 1.5% Value Zone
            group['is_pullback'] = (
                (group['trend_score'] >= TREND_REQ) & 
                (group['low'] <= group['vwap']) & 
                (group['close'] >= group['vwap'] * ZONE_LIMIT)
            ).astype(int)
            
            # D. Forward Returns (For Performance Tracking)
            group['fwd_return_5d'] = (group['close'].shift(-5) - group['close']) / group['close']
            
            results.append(group)

        final_data = pd.concat(results)
        
        # Prepare for DB Upload
        output_cols = ['Symbol', 'time', 'close', 'vwap', 'trend_score', 'is_pullback', 'fwd_return_5d']
        upload_df = final_data[output_cols].dropna(subset=['vwap'])
        
        logging.info(f"Step 3: Loading {len(upload_df)} records to Production Table...")
        # Use 'replace' for the full refresh or 'append' if you are only processing today
        upload_df.to_sql("AI_Stock_AVWAP_signals", engine, if_exists='append', index=False)
        
        logging.info("Success: Production Run Complete.")

    except Exception as e:
        logging.error(f"Production Pipeline Failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_production_pipeline()