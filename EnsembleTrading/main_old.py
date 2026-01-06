import datetime
import pandas as pd
from data_loader import DataLoader
from indicators import Indicators
from engine import EnsembleEngine
from backtester import Backtester

CONFIG = {
    "server": "BEELINK",
    "database": "Stock",
    "symbols": ["MSFT", "HOOD", "AAPL", "NVDA", "TSLA"],
    "output_table": "StrategyScores"
}

def main():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    loader = DataLoader(CONFIG["server"], CONFIG["database"])
    engine = EnsembleEngine(threshold=0.4, risk_per_trade=0.02)
    
    results_to_upload = []

    for symbol in CONFIG["symbols"]:
        print(f"Executing Run {run_id} for {symbol}...")
        
        # Pipeline
        raw_data = loader.get_ohlc(symbol)
        if raw_data.empty: continue
        
        data_with_inds = Indicators.add_all(raw_data)
        
        # Stability Check: Get score from previous SQL RunID
        prev_score = loader.get_last_score(symbol, CONFIG["output_table"])
        data_final = engine.generate_signals(data_with_inds, last_run_score=prev_score)
        
        # Prepare for Bulk Upload
        latest = data_final.iloc[[-1]].copy()
        latest['Symbol'] = symbol
        latest['ReportDate'] = latest.index
        latest['RunID'] = run_id
        results_to_upload.append(latest)

    # Bulk Insert
    if results_to_upload:
        final_df = pd.concat(results_to_upload)
        cols = ['ReportDate', 'Symbol', 'Ensemble_Score', 'Final_Score', 'Regime', 'Position_Size', 'RunID']
        loader.bulk_insert(final_df[cols], CONFIG["output_table"])
        print(f"Successfully processed {len(results_to_upload)} symbols.")

if __name__ == "__main__":
    main()