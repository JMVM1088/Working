import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import os
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass

# Suppress warnings
warnings.simplefilter('ignore')

# --- CONFIGURATION ---

@dataclass
class BacktestConfig:
    # Simulation Settings
    # Using 2024 to ensure we have enough "future" data to test the 50% gain
    scan_anchor_date: str = "2024-06-30"  
    target_profit_pct: float = 0.50       # 50% Profit Target
    
    # Pattern Settings
    lookback_days: int = 90        # Look back 90 days from anchor date
    pivot_window: int = 10         # Days to confirm a local top/bottom
    min_pivots: int = 3            # Min points to define a line
    
    # Filters
    apex_min_maturity: float = 0.50 
    apex_max_maturity: float = 0.98
    
    # Database
    db_path: str = r"C:\Users\jv2mk\OneDrive\Stock\Screener\DB\stage2"
    output_file: str = "Result/Backtest_Jun30_Convergence_50pct.csv"

# --- HELPER FUNCTIONS ---

def get_pivot_points(series: pd.Series, window: int, high_low: str) -> pd.Series:
    """Vectorized Pivot Detection."""
    pivots = pd.Series(np.nan, index=series.index)
    if high_low == 'high':
        rolling = series.rolling(window=2*window+1, center=True).max()
        is_pivot = (series == rolling)
    else:
        rolling = series.rolling(window=2*window+1, center=True).min()
        is_pivot = (series == rolling)
    pivots[is_pivot] = series[is_pivot]
    return pivots

def get_boundary_line(x_values, y_values, high_low: str):
    """Calculates 'Outer Tangent' Trendlines."""
    if len(x_values) < 2: return 0.0, 0.0
    
    points = sorted(zip(x_values, y_values))
    
    if high_low == 'high':
        # Connect Global Max and Last Pivot
        idx_max = np.argmax([p[1] for p in points])
        p1 = points[idx_max]
        p2 = points[-2] if idx_max == len(points) - 1 else points[-1]
    else:
        # Connect Global Min and Last Pivot
        idx_min = np.argmin([p[1] for p in points])
        p1 = points[idx_min]
        p2 = points[-2] if idx_min == len(points) - 1 else points[-1]

    if (p2[0] - p1[0]) == 0: return 0.0, 0.0
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    intercept = p1[1] - (slope * p1[0])
    
    return slope, intercept

def calculate_apex_maturity(start_idx, current_idx, m1, c1, m2, c2) -> float:
    if abs(m1 - m2) < 1e-9: return 0.0
    apex_x = (c2 - c1) / (m1 - m2)
    total_len = apex_x - start_idx
    curr_len = current_idx - start_idx
    if total_len == 0: return 0.0
    return curr_len / total_len

# --- CORE BACKTESTER LOGIC ---

class HistoricConvergenceBacktester:
    def __init__(self, config: BacktestConfig):
        self.cfg = config
        self.anchor_date = pd.Timestamp(self.cfg.scan_anchor_date)

    def get_tickers_from_db(self):
        if not os.path.exists(self.cfg.db_path):
            print(f"Error: DB not found at {self.cfg.db_path}")
            return []
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            # Assuming 'Ticker' column in StockList
            query = "SELECT DISTINCT Ticker FROM StockList" 
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df['Ticker'].tolist()
        except Exception as e:
            print(f"Error reading StockList: {e}")
            return []

    def load_full_history(self, ticker):
        """Loads data surrounding the anchor date (Lookback + Future for testing)."""
        # Start: Anchor - (Lookback + Buffer)
        # End: Today (to test if profit was hit later)
        start_date = self.anchor_date - timedelta(days=self.cfg.lookback_days + 50)
        
        df = None
        # 1. Try DB
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            query = f"""
                SELECT Date, Open, High, Low, Close, Volume 
                FROM Historical_SP500 
                WHERE Ticker='{ticker}' AND Date >= '{start_date.strftime('%Y-%m-%d')}'
                ORDER BY Date
            """
            df = pd.read_sql_query(query, conn, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            conn.close()
        except Exception:
            pass

        # 2. Fallback YFinance
        if df is None or df.empty:
            try:
                # Ensure we fetch enough data
                df = yf.download(ticker, start=start_date, progress=False, multi_level_index=False)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except:
                return None
        return df

    def detect_pattern_at_anchor(self, df):
        """Checks if a convergence existed EXACTLY on the anchor date."""
        # 1. Slice data strictly up to Anchor Date
        df_scan = df[df.index <= self.anchor_date]
        
        # Need enough data
        if len(df_scan) < self.cfg.lookback_days: return None
        
        # 2. Define the "Window" (Last 90 days ending on Anchor Date)
        df_window = df_scan.iloc[-self.cfg.lookback_days:]
        start_date = df_window.index[0]
        
        # 3. Calculate Pivots (Use enough history before window to confirm pivots)
        # We calculate on `df_scan` then slice to window
        high_pivs = get_pivot_points(df_scan['High'], self.cfg.pivot_window, 'high')
        low_pivs = get_pivot_points(df_scan['Low'], self.cfg.pivot_window, 'low')
        
        win_high = high_pivs.loc[df_window.index].dropna()
        win_low = low_pivs.loc[df_window.index].dropna()
        
        if len(win_high) < self.cfg.min_pivots or len(win_low) < self.cfg.min_pivots:
            return None

        # 4. Trendlines
        hx = (win_high.index - start_date).days
        hy = win_high.values
        m_res, c_res = get_boundary_line(hx, hy, 'high')
        
        lx = (win_low.index - start_date).days
        ly = win_low.values
        m_sup, c_sup = get_boundary_line(lx, ly, 'low')
        
        # 5. Convergence Logic
        current_day_idx = (df_window.index[-1] - start_date).days
        width_start = c_res - c_sup
        width_end = (m_res * current_day_idx + c_res) - (m_sup * current_day_idx + c_sup)
        
        # Must be narrowing (converging) and not yet crossed
        if width_end >= width_start: return None 
        if width_end <= 0: return None 
        
        # Maturity Check
        maturity = calculate_apex_maturity(0, current_day_idx, m_res, c_res, m_sup, c_sup)
        if not (self.cfg.apex_min_maturity <= maturity <= self.cfg.apex_max_maturity):
            return None
            
        return {
            'Close': df_window['Close'].iloc[-1],
            'Maturity': maturity
        }

    def run_forward_test(self, df, entry_price):
        """Checks future data to see if 50% profit was hit."""
        # Slice data AFTER anchor date
        df_future = df[df.index > self.anchor_date].copy()
        
        if df_future.empty:
            return {'Hit': False, 'Days': 0, 'Max_Return': 0.0}
            
        target_price = entry_price * (1 + self.cfg.target_profit_pct)
        
        # Check if High ever exceeded Target
        # .idxmax returns index of first occurrence of max, but we want boolean filter
        hits = df_future[df_future['High'] >= target_price]
        
        if not hits.empty:
            hit_date = hits.index[0]
            days_taken = (hit_date - self.anchor_date).days
            return {
                'Hit': True, 
                'Days': days_taken, 
                'Hit_Date': hit_date.strftime('%Y-%m-%d'),
                'Max_Return': self.cfg.target_profit_pct
            }
        else:
            # Did not hit target. What was the max return?
            max_high = df_future['High'].max()
            max_ret = (max_high - entry_price) / entry_price
            return {
                'Hit': False, 
                'Days': len(df_future), # Held until end of data
                'Hit_Date': 'N/A',
                'Max_Return': max_ret
            }

    def run(self):
        print(f"--- BACKTEST STARTING ---")
        print(f"Anchor Date: {self.cfg.scan_anchor_date}")
        print(f"Lookback: {self.cfg.lookback_days} days")
        print(f"Target: +{self.cfg.target_profit_pct*100}%")
        
        tickers = self.get_tickers_from_db()
        if not tickers:
            print("No tickers found. Using [AAPL, NVDA, TSLA, AMD, MSFT] for demo.")
            tickers = ['AAPL', 'NVDA', 'TSLA', 'AMD', 'MSFT']

        results = []
        
        for i, ticker in enumerate(tickers):
            print(f"[{i+1}/{len(tickers)}] Testing {ticker}...", end='\r')
            
            try:
                # 1. Load Data
                df = self.load_full_history(ticker)
                if df is None: continue
                
                # 2. Check if Pattern existed on Anchor Date
                pattern = self.detect_pattern_at_anchor(df)
                
                if pattern:
                    entry_price = pattern['Close']
                    
                    # 3. If pattern found, simulate "Buy" and hold
                    fwd_result = self.run_forward_test(df, entry_price)
                    
                    results.append({
                        'Ticker': ticker,
                        'Entry_Date': self.cfg.scan_anchor_date,
                        'Entry_Price': round(entry_price, 2),
                        'Pattern_Maturity': round(pattern['Maturity'], 2),
                        'Target_Hit': fwd_result['Hit'],
                        'Days_To_Target': fwd_result['Days'],
                        'Date_Hit': fwd_result['Hit_Date'],
                        'Max_Potential_Return': round(fwd_result['Max_Return'] * 100, 2)
                    })
            except Exception as e:
                continue
                
        print("\n--- Backtest Complete ---")
        
        if results:
            df_res = pd.DataFrame(results)
            # Sort: Hits first, then by days taken (fastest first)
            df_res = df_res.sort_values(by=['Target_Hit', 'Days_To_Target'], ascending=[False, True])
            
            os.makedirs(os.path.dirname(self.cfg.output_file), exist_ok=True)
            df_res.to_csv(self.cfg.output_file, index=False)
            
            # Summary Stats
            total_trades = len(df_res)
            hits = df_res['Target_Hit'].sum()
            avg_days = df_res[df_res['Target_Hit']]['Days_To_Target'].mean()
            
            print(f"Found Convergences: {total_trades}")
            print(f"Success (Hit +50%): {hits} ({(hits/total_trades)*100:.1f}%)")
            if hits > 0:
                print(f"Avg Days to Hit:    {avg_days:.1f} days")
            print(f"Results saved to: {self.cfg.output_file}")
            print(df_res.head(10))
        else:
            print("No tickers matched the convergence pattern on that date.")

# --- EXECUTION ---

if __name__ == '__main__':
    # You can change the date to any historical date
    # Defaults to Jun 30, 2024 to allow for ~1.5 years of forward data
    config = BacktestConfig(
        scan_anchor_date="2025-08-20",
        target_profit_pct=0.20,  # 50%
        lookback_days=90,
        db_path = r"C:\Users\jv2mk\OneDrive\Stock\Screener\DB\stage2"
    )
    
    bt = HistoricConvergenceBacktester(config)
    bt.run()