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
class ScannerConfig:
    # Pattern Settings
    lookback_window: int = 90      # User specified "Last 90 days" window for pattern
    pivot_window: int = 10         # Days to confirm a local top/bottom
    min_pivots: int = 3            # Min points to define a line
    min_slope_diff: float = 0.0001 # Minimum angle to call it a triangle
    
    # Filters
    vol_contraction_thresh: float = 0.85
    apex_min_maturity: float = 0.50 # Relaxed slightly for scanning
    apex_max_maturity: float = 0.98
    
    # Database
    db_path: str = r"C:\Users\jv2mk\OneDrive\Stock\Screener\DB\stage2"
    output_file: str = "Result/Converging_Tickers.csv"

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

# --- CORE SCANNER LOGIC ---

class ConvergenceScanner:
    def __init__(self, config: ScannerConfig):
        self.cfg = config

    def get_tickers_from_db(self):
        """Fetches list of tickers from StockList table."""
        if not os.path.exists(self.cfg.db_path):
            print(f"Error: DB not found at {self.cfg.db_path}")
            return []
            
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            # Assuming 'Ticker' is the column name in 'StockList'
            query = "SELECT distinct Ticker FROM StockList where compositeName = 'SP500'" 
            df = pd.read_sql_query(query, conn)
            conn.close()
            tickers = df['Ticker'].tolist()
            print(f"Found {len(tickers)} tickers in database.")
            return tickers
        except Exception as e:
            print(f"Error reading StockList: {e}")
            return []

    def load_data(self, ticker):
        """Fetches last ~1 year of data to ensure valid 90-day calculation."""
        # We need enough buffer for Pivot Windows (10 days) + Lookback (90 days) + Safety
        # Fetching 250 days (~1 year) is safe.
        
        df = None
        # 1. Try DB
        try:
            conn = sqlite3.connect(self.cfg.db_path)
            # Limit query to last 365 days for speed
            end_date_str = datetime.now().strftime('%Y-%m-%d')
            start_date_str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            query = f"""
                SELECT Date, Open, High, Low, Close, Volume 
                FROM Historical_SP500 
                WHERE Ticker='{ticker}' AND Date >= '{start_date_str}'
                ORDER BY Date
            """
            df = pd.read_sql_query(query, conn, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            conn.close()
        except Exception:
            pass # Fail silently, try YF

        # 2. Fallback YFinance
        if df is None or df.empty:
            try:
                df = yf.download(ticker, period="1y", progress=False, multi_level_index=False)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except:
                return None

        return df

    def analyze_ticker(self, ticker, df):
        """Checks if the ticker is CURRENTLY in a convergence pattern."""
        if len(df) < self.cfg.lookback_window + self.cfg.pivot_window:
            return None
        
        # 1. Slice the "Last 90 Days" window
        # We assume the analysis is for "Today". 
        df_window = df.iloc[-self.cfg.lookback_window:]
        start_date = df_window.index[0]
        
        # 2. Calculate Pivots on the full DF first (to get valid context)
        # Then slice to window
        all_high_pivs = get_pivot_points(df['High'], self.cfg.pivot_window, 'high')
        all_low_pivs = get_pivot_points(df['Low'], self.cfg.pivot_window, 'low')
        
        # Slice to current window
        win_high = all_high_pivs.loc[df_window.index]
        win_low = all_low_pivs.loc[df_window.index]
        
        # Filter NaNs
        valid_high = win_high.dropna()
        valid_low = win_low.dropna()
        
        if len(valid_high) < self.cfg.min_pivots or len(valid_low) < self.cfg.min_pivots:
            return None

        # 3. Fit Trendlines
        # Convert dates to integer days from window start
        hx = (valid_high.index - start_date).days
        hy = valid_high.values
        m_res, c_res = get_boundary_line(hx, hy, 'high')
        
        lx = (valid_low.index - start_date).days
        ly = valid_low.values
        m_sup, c_sup = get_boundary_line(lx, ly, 'low')
        
        # 4. Check Convergence
        # Resistance slope should be < Support slope? No, usually Res is down (-), Sup is up (+)
        # Strict definition: Res Slope < Sup Slope checks if they cross in future? 
        # Actually: Convergence means (Res Slope) < (Sup Slope) is FALSE if Res is above Sup.
        # We need Res Slope < Sup Slope to be FALSE. We need Res > Sup?
        # Let's use simple logic: Resistance should be falling (or flat), Support rising (or flat).
        # OR just check they are getting closer.
        
        # Calculate width at start vs width at end
        width_start = c_res - c_sup
        current_day_idx = (df_window.index[-1] - start_date).days
        width_end = (m_res * current_day_idx + c_res) - (m_sup * current_day_idx + c_sup)
        
        if width_end >= width_start: return None # Diverging or Parallel
        if width_end <= 0: return None # Already crossed over
        
        # 5. Maturity Check
        maturity = calculate_apex_maturity(0, current_day_idx, m_res, c_res, m_sup, c_sup)
        if maturity < self.cfg.apex_min_maturity: return None
        if maturity > self.cfg.apex_max_maturity: return None
        
        # 6. Success - Return Data
        return {
            'Ticker': ticker,
            'Date': df_window.index[-1].strftime('%Y-%m-%d'),
            'Close': df_window['Close'].iloc[-1],
            'Maturity': round(maturity, 2),
            'Slope_Res': round(m_res, 4),
            'Slope_Sup': round(m_sup, 4),
            'Pattern_Width_Start': round(width_start, 2),
            'Pattern_Width_End': round(width_end, 2)
        }

    def run(self):
        print(f"--- Starting Scan ---")
        print(f"Target DB: {self.cfg.db_path}")
        
        # 1. Get List
        tickers = self.get_tickers_from_db()
        if not tickers:
            print("No tickers found. Exiting.")
            return

        results = []
        total = len(tickers)
        
        # 2. Loop
        for i, ticker in enumerate(tickers):
            print(f"[{i+1}/{total}] Scanning {ticker}...", end='\r')
            
            try:
                df = self.load_data(ticker)
                if df is not None:
                    res = self.analyze_ticker(ticker, df)
                    if res:
                        results.append(res)
            except Exception as e:
                # print(f"Skipping {ticker}: {e}")
                continue
        
        print("\nScan Complete.")
        
        # 3. Save
        if results:
            df_res = pd.DataFrame(results)
            # Sort by Maturity (closest to breakout first)
            df_res = df_res.sort_values(by='Maturity', ascending=False)
            
            os.makedirs(os.path.dirname(self.cfg.output_file), exist_ok=True)
            df_res.to_csv(self.cfg.output_file, index=False)
            
            print(f"Found {len(results)} converging tickers.")
            print(f"Saved to: {self.cfg.output_file}")
            print(df_res.head())
        else:
            print("No converging tickers found.")

# --- EXECUTION ---

if __name__ == '__main__':
    # Adjust config as needed
    config = ScannerConfig(
        db_path = r"C:\Users\jv2mk\OneDrive\Stock\Screener\DB\stage2",
        lookback_window = 90  # "Last 90 days"
    )
    
    scanner = ConvergenceScanner(config)
    scanner.run()