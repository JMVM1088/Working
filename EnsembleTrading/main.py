import datetime
import pandas as pd
from data_loader import DataLoader
from indicators import Indicators
from engine import EnsembleEngine
from backtester import Backtester

# --- 1. Configuration Gate ---
CONFIG = {
    "server": "BEELINK",
    "database": "Stock",
    "symbols": ["MSFT", "HOOD", "AAPL", "NVDA", "TSLA"],
    "output_table": "StrategyScores",
    
    # TOGGLE: "BACKTEST" for research, "PROD" for daily execution
    "mode": "PROD", 
    
    # Backtest Window (Ignored in PROD mode)
    "start_date": "2023-01-01", 
    "end_date": "2024-12-31",
    
    # Strategy Settings
    "threshold": 0.4,          # Conviction required to trade
    "risk_per_trade": 0.02,    # 2% risk normalized by ATR
    "stability_limit": 0.5     # Max allowed score drift between runs
}

def main():
    # Generate unique ID for this execution
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"--- INITIALIZING ENSEMBLE SYSTEM | Mode: {CONFIG['mode']} | RunID: {run_id} ---")

    # Initialize Modules
    loader = DataLoader(CONFIG["server"], CONFIG["database"])
    engine = EnsembleEngine(
        threshold=CONFIG["threshold"], 
        risk_per_trade=CONFIG["risk_per_trade"],
        stability_limit=CONFIG["stability_limit"]
    )
    tester = Backtester(fee=0.0005) # 5 basis points for slippage
    
    results_to_upload = []
    performance_summary = []

    for symbol in CONFIG["symbols"]:
        try:
            # 2. Data Retrieval
            # In PROD, we pull all data to ensure indicators (SMA200) are accurate
            # In BACKTEST, we pull data up to the end_date
            load_end = None if CONFIG["mode"] == "PROD" else CONFIG["end_date"]
            raw_data = loader.get_ohlc(symbol, end_date=load_end)
            
            if raw_data.empty:
                print(f"Warning: No data for {symbol}. Skipping.")
                continue

            # 3. Indicator Calculation (The Weak Learners + Filters)
            data_with_inds = Indicators.add_all(raw_data)

            # --- BRANCH: BACKTEST MODE ---
            if CONFIG["mode"] == "BACKTEST":
                # Slice the data to the user-defined backtest window
                # We do this AFTER indicator calculation so SMA200 is ready on Day 1
                bt_data = data_with_inds.loc[CONFIG["start_date"]:CONFIG["end_date"]]
                
                if bt_data.empty:
                    print(f"No data for {symbol} within date range.")
                    continue

                # Run historical signals (Stability filter usually off for long-term BT)
                data_signals = engine.generate_signals(bt_data)
                backtest_results, metrics = tester.run(data_signals)
                
                performance_summary.append({
                    "Symbol": symbol,
                    "Sharpe": metrics["Sharpe"],
                    "MaxDD": metrics["MaxDD"] * 100,
                    "Final_Equity": backtest_results['Equity'].iloc[-1]
                })
                print(f"Backtest {symbol}: Sharpe {metrics['Sharpe']:.2f}")

            # --- BRANCH: PROD MODE ---
            elif CONFIG["mode"] == "PROD":
                # Get previous score from SQL to apply Intraday Stability Filter
                prev_score = loader.get_last_score(symbol, CONFIG["output_table"])
                
                # Calculate latest signal
                data_signals = engine.generate_signals(data_with_inds, last_run_score=prev_score)
                
                # Extract the absolute latest signal row
                latest_row = data_signals.iloc[[-1]].copy()
                latest_row['Symbol'] = symbol
                latest_row['ReportDate'] = latest_row.index
                latest_row['RunID'] = run_id
                
                results_to_upload.append(latest_row)
                print(f"Signal Generated for {symbol}: Score {latest_row['Final_Score'].values[0]}")

        except Exception as e:
            print(f"CRITICAL ERROR on {symbol}: {e}")

    # --- 4. Finalizing Execution ---
    
    if CONFIG["mode"] == "BACKTEST":
        print("\n" + "="*50)
        print(f"{'SYMBOL':<10} | {'SHARPE':<8} | {'MAX DD %':<10} | {'EQUITY'}")
        print("-" * 50)
        for p in performance_summary:
            print(f"{p['Symbol']:<10} | {p['Sharpe']:<8.2f} | {p['MaxDD']:<10.2f} | ${p['Final_Equity']:,.0f}")
        print("="*50)

    elif CONFIG["mode"] == "PROD" and results_to_upload:
        # Bulk Insert the daily signals into SQL
        final_upload_df = pd.concat(results_to_upload)
        cols = ['ReportDate', 'Symbol', 'Ensemble_Score', 'Final_Score', 'Regime', 'Position_Size', 'RunID']
        
        loader.bulk_insert(final_upload_df[cols], CONFIG["output_table"])
        print(f"\n--- PRODUCTION RUN COMPLETE ---")
        print(f"RunID {run_id} saved to {CONFIG['output_table']}.")

if __name__ == "__main__":
    main()