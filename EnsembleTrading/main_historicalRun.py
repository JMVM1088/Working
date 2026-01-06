import datetime
import pandas as pd
from data_loader import DataLoader
from indicators import Indicators
from engine import EnsembleEngine
import Util

CONFIG = {
    "server": "BEELINK",
    "database": "Stock",
    "symbols": ["MSFT", "HOOD", "AAPL", "NVDA", "TSLA"],
    "output_table": "StrategyScores",
    "mode": "PROD",
    "target_date": "2025-12-26",  # Set this to the date you want to "backfill"
    "months_to_backfill": 9,
    "threshold": 0.4
}
connection_string = (
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        r'SERVER=BEELINK;'  # Replace with your server name
        r'DATABASE=Stock;' # Replace with your database name
        r'Trusted_Connection=yes;'
    )
def main():
    # We use the target_date as part of the RunID so you know it was a backfill run
    run_id = f"BACKFILL_{CONFIG['target_date'].replace('-', '')}"
    
    loader = DataLoader(CONFIG["server"], CONFIG["database"])
    engine = EnsembleEngine(threshold=CONFIG["threshold"], risk_per_trade=0.02)
    
# 1. Generate the date range (Last 3 months up to today)
    end_date = datetime.date(2025,9,30)  ##datetime.date.today()
    start_date = datetime.date(2025,8,5)
    ##start_date = end_date - datetime.timedelta(days=CONFIG["months_to_backfill"] * 30)
    
    # Create a list of all business days in that range
    date_range = pd.bdate_range(start=start_date, end=end_date)
    
    print(f"Starting Batch Backfill from {start_date} to {end_date}")
    print(f"Total Trading Days to process: {len(date_range)}")


    #results_to_upload = []
    stock_list = list(Util.get_data_from_sql(connection_string, "exec stock..sp_GetStockList 'US'"))

    # 2. Loop through each day in the sequence
    for current_date in date_range:
        date_str = current_date.strftime('%Y-%m-%d')
        run_id = f"BATCH_{current_date.strftime('%Y%m%d')}"
        results_to_upload = []
        
        print(f"Processing Date: {date_str}...")
    
        #for symbol in CONFIG["symbols"]:
        for ticker in stock_list:
            try:
                # 1. Pull data UP TO the target date
                raw_data = loader.get_ohlc(ticker[0], end_date=date_str)
                if raw_data.empty: continue
                
                # 2. Add Indicators
                data_with_inds = Indicators.add_all(raw_data)

                # 3. Get the score from the run IMMEDIATELY PRECEDING the target date
                # This ensures the Stability Filter still works correctly
                prev_score = loader.get_last_score(ticker[0], CONFIG["output_table"])
                
                # 4. Generate signals as if it were the target_date
                data_signals = engine.generate_signals(data_with_inds, last_run_score=prev_score)
                
                # 5. Extract the row for the target_date
                latest_row = data_signals.iloc[[-1]].copy()
                latest_row['Symbol'] = ticker[0]
                latest_row['ReportDate'] = pd.to_datetime(date_str)
                latest_row['RunID'] = run_id
                
                results_to_upload.append(latest_row)
                print(f"Saved {ticker[0]} for {date_str}")

            except Exception as e:
                print(f"Error on {ticker[0]}: {e}")

        # 6. Bulk Insert to SQL
        if results_to_upload:
            final_df = pd.concat(results_to_upload)
            cols = ['ReportDate', 'Symbol', 'Ensemble_Score', 'Final_Score', 'Regime', 'Position_Size', 'RunID']
            loader.bulk_insert(final_df[cols], CONFIG["output_table"])
            print(f"\nSuccessfully backfilled {len(results_to_upload)} records.")

if __name__ == "__main__":
    main()