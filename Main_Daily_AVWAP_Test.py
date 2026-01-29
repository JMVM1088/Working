import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from numba import jit
import logging

# --- Configuration & Hyperparameters ---
DB_CONFIG = {
    "driver": "ODBC Driver 17 for SQL Server",
    "server": "BEELINK",
    "database": "Stock",
    "trusted_connection": "Yes"
}

PARAMS = {
    "prd": 50,           # Swing Period
    "base_apt": 20.0,    # Adaptive Price Tracking
    "use_adapt": False,  # Set to True to enable ATR-based adaptation
    "vol_bias": 10.0,    # Volatility Sensitivity
    "atr_len": 50        # ATR Lookback
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Core Logic (Numba Optimized) ---
@jit(nopython=True)
def compute_dynamic_avwap_logic(high, low, hlc3, volume, atr, atr_avg, 
                               prd, base_apt, use_adapt, vol_bias):
    n = len(hlc3)
    vwap_values = np.full(n, np.nan)
    
    # State tracking
    p_state = 0.0
    vol_state = 0.0
    last_dir = 0
    phL = 0  # Pivot High Index
    plL = 0  # Pivot Low Index
    
    for i in range(1, n):
        # A. Pivot Detection (Lookback extremes)
        is_highest = True
        is_lowest = True
        start_idx = max(0, i - prd + 1)
        
        for j in range(start_idx, i):
            if high[j] > high[i]: is_highest = False
            if low[j] < low[i]: is_lowest = False
            
        if is_highest: phL = i
        if is_lowest:  plL = i
            
        current_dir = 1 if phL > plL else -1
        
        # B. Alpha Calculation
        ratio = atr[i] / atr_avg[i] if atr_avg[i] > 0 else 1.0
        apt_raw = base_apt / (ratio ** vol_bias) if use_adapt else base_apt
        apt_clamped = max(5.0, min(300.0, apt_raw))
        alpha = 1.0 - np.exp(-np.log(2.0) / max(1.0, apt_clamped))
        
        # C. VWAP Accumulation & Pivot Reset
        if current_dir != last_dir:
            # Anchor Reset: Start from the price/volume of the pivot bar
            pivot_idx = plL if current_dir > 0 else phL
            pivot_price = low[pivot_idx] if current_dir > 0 else high[pivot_idx]
            
            p_state = pivot_price * volume[pivot_idx]
            vol_state = volume[pivot_idx]
            
            # Update state with current bar data
            p_state = (1.0 - alpha) * p_state + alpha * (hlc3[i] * volume[i])
            vol_state = (1.0 - alpha) * vol_state + alpha * volume[i]
        else:
            # Recursive Update
            p_state = (1.0 - alpha) * p_state + alpha * (hlc3[i] * volume[i])
            vol_state = (1.0 - alpha) * vol_state + alpha * volume[i]
            
        vwap_values[i] = p_state / vol_state if vol_state > 0 else np.nan
        last_dir = current_dir
        
    return vwap_values

# --- 2. Data Processing Wrapper ---
class QuantEngine:
    def __init__(self):
        conn_str = "mssql+pyodbc://@BEELINK/Stock?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
        self.engine = create_engine(conn_str)

    def fetch_data(self):
        query = "SELECT Symbol, time, [high], [low], [close], [Volume] FROM AI_stock_prices where [Time] >= '2025-12-01' ORDER BY Symbol, time"
        logging.info("Fetching data from SQL Server...")
        return pd.read_sql(query, self.engine)

    def run_analysis(self, df):
        logging.info("Starting signal generation...")
        results = []
        
        for symbol, group in df.groupby('Symbol'):
            group = group.sort_values('time').copy()
            
            # HLC3 and ATR (RMA implementation)
            group['hlc3'] = (group['high'] + group['low'] + group['close']) / 3
            tr = np.maximum(group['high'] - group['low'], 
                    np.maximum(abs(group['high'] - group['close'].shift(1)), 
                               abs(group['low'] - group['close'].shift(1))))
            
            # Pine ta.rma is equivalent to an EWM with alpha = 1/length
            group['atr'] = tr.ewm(alpha=1/PARAMS['atr_len'], adjust=False).mean()
            group['atr_avg'] = group['atr'].ewm(alpha=1/PARAMS['atr_len'], adjust=False).mean()
            
            # Generate VWAP
            group['dynamic_vwap'] = compute_dynamic_avwap_logic(
                group['high'].values, group['low'].values, group['hlc3'].values, 
                group['Volume'].values.astype(float), group['atr'].values, group['atr_avg'].values,
                PARAMS['prd'], PARAMS['base_apt'], PARAMS['use_adapt'], PARAMS['vol_bias']
            )
            # Add this logic inside the run_analysis loop, after generating 'dynamic_vwap'

            # 1. Check if Close was above VWAP for the previous 5 days
            # We use a rolling min on a boolean: if the min is 1, all 5 days were True.
            group['above_vwap'] = (group['close'] > group['dynamic_vwap']).astype(int)
            group['past_5_above'] = group['above_vwap'].shift(1).rolling(window=5).min() == 1

            # 2. Define the "Touch"
            # In quant terms, a 'touch' occurs when the VWAP is within the High-Low range of the day.
            group['touches_vwap'] = (group['low'] <= group['dynamic_vwap']) & (group['high'] >= group['dynamic_vwap'])

            # 3. Combine for the specific Bullish Setup
            group['bullish_pullback'] = (group['past_5_above'] & group['touches_vwap']).astype(int)

            
            # Add Trading Signals
            group['signal'] = np.where(group['close'] > group['dynamic_vwap'], 'Bullish', 'Bearish')
            group['crossover'] = (group['close'] > group['dynamic_vwap']) & (group['close'].shift(1) <= group['dynamic_vwap'].shift(1))
            
            results.append(group)
            
        return pd.concat(results)

    def save_to_qa_table(self, df):
        # We save only the essential columns to a new results table
        output_df = df[['Symbol', 'time', 'close', 'dynamic_vwap', 'signal', 'crossover','bullish_pullback']].copy()
        logging.info(f"Saving {len(output_df)} records to [AI_stock_VWAP_signals]...")
        output_df.to_sql("AI_stock_VWAP_signals", self.engine, if_exists='replace', index=False)

def calculate_backtest_stats(df, horizon=5):
    """
    Calculates the success rate of Bullish Pullback signals 
    over a specific trading horizon.
    """
    # 1. Calculate Forward Return (N days into the future)
    # We use shift(-horizon) to look ahead
    df[f'fwd_close_{horizon}d'] = df.groupby('Symbol')['close'].shift(-horizon)
    df[f'fwd_return_{horizon}d'] = (df[f'fwd_close_{horizon}d'] - df['close']) / df['close']
    
    # 2. Filter for only the Bullish Pullback signals
    pullbacks = df[df['bullish_pullback'] == 1].copy()
    
    if pullbacks.empty:
        return None, 0
    
    # 3. Calculate Metrics
    win_rate = (pullbacks[f'fwd_return_{horizon}d'] > 0).mean() * 100
    avg_return = pullbacks[f'fwd_return_{horizon}d'].mean() * 100
    total_signals = len(pullbacks)
    
    stats = {
        "Horizon": f"{horizon} Days",
        "Total Signals": total_signals,
        "Win Rate": f"{win_rate:.2f}%",
        "Avg Return": f"{avg_return:.4f}%"
    }
    
    return stats, pullbacks


# --- 3. Main Execution ---
if __name__ == "__main__":
    try:
        engine = QuantEngine()
        raw_data = engine.fetch_data()
        
        if raw_data.empty:
            logging.warning("No data found in AI_stock_prices table.")
        else:
            processed_data = engine.run_analysis(raw_data)
            #engine.save_to_qa_table(processed_data)
            logging.info("QA implementation complete. Results ready in [AI_stock_signals].")
            
            # Sample output for verification
            print("\n--- Recent Signal Preview ---")
            #print(processed_data.tail(10))
            stats, pullback_results = calculate_backtest_stats(processed_data, horizon=5)
            if stats:
                print("\n--- Bullish Pullback Success Rate (Backtest) ---")
                for k, v in stats.items():
                    print(f"{k}: {v}")
            
    except Exception as e:
        logging.error(f"Execution failed: {e}")