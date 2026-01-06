import pandas as pd
import datetime
from data_loader import DataLoader
from indicators import Indicators
from engine import EnsembleEngine
from backtester import Backtester
from sqlalchemy import create_engine

# --- Configuration ---
SQL_CONFIG = {
    "server": "YOUR_SERVER_NAME",
    "database": "TradingDB",
    "symbols": ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"],
    "output_table": "StrategyScores"
}
def get_last_score(symbol, config):
    """
    Queries SQL for the most recent Final_Score for a specific symbol.
    """
    conn_str = f"mssql+pyodbc://{config['server']}/{config['database']}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    engine = create_engine(conn_str)
    
    query = f"""
    SELECT TOP 1 Final_Score 
    FROM {config['output_table']} 
    WHERE Symbol = '{symbol}' 
    ORDER BY RunID DESC
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            return result[0] if result else None
    except:
        return None
def main():
    # 1. Generate a Unique RunID (Format: YYYYMMDD_HHMMSS)
    # This allows you to track multiple runs per day and identify exactly when they occurred.
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Strategy Execution. RunID: {run_id}")

    loader = DataLoader(server=SQL_CONFIG["server"], database=SQL_CONFIG["database"])
    engine = EnsembleEngine(threshold=0.4, risk_per_trade=0.02)
    tester = Backtester(fee=0.0002)
    
    all_latest_signals = []

    for symbol in SQL_CONFIG["symbols"]:
        try:
            print(f"Processing {symbol}...")
            raw_data = loader.get_ohlc(symbol)
            if raw_data.empty: continue

            data_with_inds = Indicators.add_all(raw_data)
            data_with_signals = engine.generate_signals(data_with_inds)

            # Extract latest row
            latest_row = data_with_signals.iloc[[-1]].copy()
            latest_row['Symbol'] = symbol
            latest_row['ReportDate'] = latest_row.index
            
            # 2. Attach the RunID to this specific record
            latest_row['RunID'] = run_id
            
            all_latest_signals.append(latest_row)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # --- 3. Perform Bulk Insert ---
    if all_latest_signals:
        final_upload_df = pd.concat(all_latest_signals)
        
        # Include RunID in the output columns
        output_cols = ['ReportDate', 'Symbol', 'Ensemble_Score', 'Final_Score', 'Regime', 'Position_Size', 'RunID']
        final_upload_df = final_upload_df[output_cols]

        conn_str = f"mssql+pyodbc://{SQL_CONFIG['server']}/{SQL_CONFIG['database']}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        db_engine = create_engine(conn_str)
        
        final_upload_df.to_sql(
            SQL_CONFIG['output_table'], 
            db_engine, 
            if_exists='append', 
            index=False,
            chunksize=50
        )
        print(f"\nSUCCESS: Inserted {len(final_upload_df)} records for RunID {run_id}.")
    else:
        print("No signals generated.")

if __name__ == "__main__":
    main()