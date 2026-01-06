import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
# Data Generation
USE_REAL_DATA = False   # Set True if you hook up a real API
SYMBOL = 'MOCK_TECH'
START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Technical Parameters
ATR_PERIOD = 14
SMA_TREND_PERIOD = 200  # For Trend Filter
VOL_SMA_PERIOD = 20     # For Volume Filter

# Risk Management
SL_ATR_MULTIPLIER = 2.0
TP_RR_RATIO = 2.0
MAX_HOLD_DAYS = 60
RISK_PER_TRADE = 0.01   # 1% of capital

# Pattern Parameters (Cup)
CUP_MIN_LEN = 40
CUP_MAX_LEN = 250
CUP_RIM_TOLERANCE = 0.07
CUP_BOTTOM_DROP = 0.10
HANDLE_MAX_DROP = 0.08  # Slightly looser to catch more handles
BREAKOUT_BUFFER = 0.01

# Pattern Parameters (Rectangle)
RECT_MIN_DAYS = 20
RECT_MAX_DAYS = 100
RECT_RANGE_TOL = 0.08   # Loosened slightly

# ==========================================
# 2. DATA & INDICATOR HELPERS
# ==========================================

def calculate_technical_indicators(df: pd.DataFrame):
    """Calculates SMA, ATR, and Volume MA."""
    df = df.copy()
    # Trend Filter
    df['SMA_200'] = df['Close'].rolling(window=SMA_TREND_PERIOD).mean()
    
    # Volume Filter
    df['Vol_SMA_20'] = df['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    
    # ATR for Position Sizing
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=ATR_PERIOD).mean()
    
    return df.dropna()

def generate_mock_data(symbol, days=3000):
    """Generates synthetic OHLCV data with trends and random noise."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, days) # slight positive drift
    price_path = 100 * np.cumprod(1 + returns)
    
    data = []
    for i, date in enumerate(dates):
        close = price_path[i]
        # Simulate OHLC variations
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_p = (high + low) / 2
        # Simulate Volume
        vol = np.random.randint(100000, 5000000)
        
        data.append([date, open_p, high, low, close, vol])
        
    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.set_index('Date', inplace=True)
    return df

# ==========================================
# 3. PATTERN DETECTION LOGIC
# ==========================================

def detect_cup_and_handle(df: pd.DataFrame):
    """
    Enhanced Cup & Handle Detector.
    Includes Trend Filter (SMA 200) and Volume Confirmation.
    """
    signals = []
    close = df['Close'].values
    volume = df['Volume'].values
    sma_200 = df['SMA_200'].values
    vol_sma = df['Vol_SMA_20'].values
    dates = df.index
    n = len(df)

    # Optimization: Early exit if not enough data
    if n < CUP_MIN_LEN: return []

    for start in range(0, n - CUP_MIN_LEN):
        
        # 1. TREND FILTER: Context is King
        # Only look for bullish cups if price is above 200 SMA
        if close[start] < sma_200[start]:
            continue

        for cup_len in range(CUP_MIN_LEN, min(CUP_MAX_LEN, n - start), 5):
            end = start + cup_len
            if end >= n: break
            
            window = close[start:end]
            
            # Define Rims
            rim_window_size = max(3, int(0.15 * len(window)))
            left_region = window[:rim_window_size]
            right_region = window[-rim_window_size:]
            
            left_max = left_region.max()
            right_max = right_region.max()
            rim_mean = (left_max + right_max) / 2.0
            
            # Check 1: Rim Symmetry
            if abs(left_max - right_max) / rim_mean > CUP_RIM_TOLERANCE:
                continue
                
            # Check 2: Depth
            bottom = window.min()
            bottom_idx = window.argmin()
            depth_pct = (rim_mean - bottom) / rim_mean
            
            if depth_pct < CUP_BOTTOM_DROP: # Not deep enough
                continue
            if depth_pct > 0.50: # Too deep (likely a crash, not a cup)
                continue

            # Check 3: Handle (must be after the bottom)
            # handle starts from bottom to end of window
            if bottom_idx > len(window) - 5: continue # Bottom too late
            
            handle_region = window[bottom_idx:]
            handle_min = handle_region.min()
            handle_drop = (rim_mean - handle_min) / rim_mean
            
            if handle_drop > HANDLE_MAX_DROP:
                continue

            # Check 4: Breakout & Volume Confirmation at `end` (Signal Day)
            if close[end] > right_max * (1 + BREAKOUT_BUFFER):
                
                # Volume Filter: Breakout volume > 1.2x Average
                if volume[end] > vol_sma[end] * 1.2:
                    signal = {
                        'type': 'cup_handle',
                        'date': dates[end],
                        'trigger_price': float(close[end]),
                        'stop_loss_price': float(handle_min) # Tight SL below handle
                    }
                    signals.append(signal)
                    # Jump forward to avoid overlapping signals for same pattern
                    break 

    # Deduplicate: Keep first signal per day
    unique = {}
    for s in signals:
        if s['date'] not in unique:
            unique[s['date']] = s
    return list(unique.values())

def detect_bullish_rectangle(df: pd.DataFrame):
    """
    Enhanced Bullish Rectangle Detector.
    """
    signals = []
    close = df['Close'].values
    sma_200 = df['SMA_200'].values
    dates = df.index
    n = len(df)

    for window_len in range(RECT_MIN_DAYS, RECT_MAX_DAYS, 5):
        for start in range(0, n - window_len - 1):
            
            # Trend Filter
            if close[start] < sma_200[start]:
                continue
            
            end = start + window_len
            if end >= n: break

            window = close[start:end]
            mean_price = window.mean()
            price_range = window.max() - window.min()
            
            # Check 1: Tightness
            if (price_range / mean_price) > RECT_RANGE_TOL:
                continue

            upper = window.max()
            lower = window.min()
            
            # Check 2: Touches (Support/Resistance validation)
            # We look for prices within 1% of top/bottom
            upper_touches = np.sum(window > upper * 0.99)
            lower_touches = np.sum(window < lower * 1.01)
            
            if upper_touches < 2 or lower_touches < 2:
                continue

            # Check 3: Breakout
            if close[end] > upper * (1 + BREAKOUT_BUFFER):
                signal = {
                    'type': 'bullish_rectangle',
                    'date': dates[end],
                    'trigger_price': float(close[end]),
                    'stop_loss_price': float(lower) # SL below structure
                }
                signals.append(signal)
                break # Jump to next start

    unique = {}
    for s in signals:
        if s['date'] not in unique:
            unique[s['date']] = s
    return list(unique.values())

# ==========================================
# 4. BACKTESTING ENGINE
# ==========================================

class EnhancedBacktester:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.capital = 100000.0
        self.initial_capital = 100000.0
        self.trades = []
        self.equity_curve = []
        self.open_trade = None
        
        # Prepare Signal Column for O(1) Access
        self.df['Signal'] = None
    
    def load_signals(self, signals):
        """Map list of signal dicts to the DataFrame for fast access."""
        for s in signals:
            if s['date'] in self.df.index:
                self.df.at[s['date'], 'Signal'] = s

    def run(self):
        print(f"Starting Backtest with ${self.capital:,.2f}...")
        
        for i in range(1, len(self.df)):
            current_date = self.df.index[i]
            prev_date = self.df.index[i-1]
            
            row = self.df.iloc[i]
            prev_row = self.df.iloc[i-1]
            
            # 1. Manage Existing Trade
            if self.open_trade:
                self._manage_trade(current_date, row)
            
            # 2. Check for New Entry (Based on Yesterday's Signal)
            # We enter on OPEN of current day if signal existed yesterday
            signal = prev_row.get('Signal')
            
            if not self.open_trade and signal is not None:
                atr = prev_row['ATR']
                entry_price = row['Open'] # Realistic Entry: Next Day Open
                
                # Determine Position Size
                risk_amt = self.capital * RISK_PER_TRADE
                sl_dist = atr * SL_ATR_MULTIPLIER
                
                # Dynamic Stop Loss: Use structure (from signal) or ATR, whichever is tighter/safer
                # Here we default to ATR for consistency, but you could use signal['stop_loss_price']
                sl_price = entry_price - sl_dist
                tp_price = entry_price + (sl_dist * TP_RR_RATIO)
                
                if sl_dist > 0:
                    shares = int(risk_amt / sl_dist)
                    
                    if shares > 0:
                        self.open_trade = {
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'shares': shares,
                            'sl': sl_price,
                            'tp': tp_price,
                            'type': signal['type'],
                            'days_held': 0
                        }
                        print(f"[{current_date.date()}] ENTER LONG ({signal['type']}) @ {entry_price:.2f} | Shares: {shares}")

            # Record Equity
            self.equity_curve.append({'Date': current_date, 'Equity': self.capital})

        self._generate_report()

    def _manage_trade(self, date, row):
        t = self.open_trade
        exit_price = None
        reason = None
        
        # Check Stops/Targets based on High/Low
        if row['Low'] <= t['sl']:
            exit_price = t['sl'] # Slippage not modeled, assuming fill at SL
            reason = 'Stop Loss'
        elif row['High'] >= t['tp']:
            exit_price = t['tp']
            reason = 'Take Profit'
        
        # Time Exit
        t['days_held'] += 1
        if not reason and t['days_held'] >= MAX_HOLD_DAYS:
            exit_price = row['Close']
            reason = 'Time Exit'
            
        if exit_price:
            pnl = (exit_price - t['entry_price']) * t['shares']
            self.capital += pnl
            
            self.trades.append({
                'entry_date': t['entry_date'],
                'exit_date': date,
                'pnl': pnl,
                'reason': reason,
                'type': t['type']
            })
            print(f"[{date.date()}] EXIT {reason} | PnL: ${pnl:.2f} | Cap: ${self.capital:,.0f}")
            self.open_trade = None

    def _generate_report(self):
        os.makedirs('Result', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # 1. Trades CSV
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv(f'Result/{timestamp}_trades.csv', index=False)
            
            # Stats
            win_rate = (trades_df['pnl'] > 0).mean() * 100
            total_pnl = trades_df['pnl'].sum()
            print(f"\n--- RESULTS ---")
            print(f"Total Trades: {len(trades_df)}")
            print(f"Win Rate:     {win_rate:.1f}%")
            print(f"Total PnL:    ${total_pnl:,.2f}")
            print(f"Final Equity: ${self.capital:,.2f}")
        else:
            print("No trades executed.")
            
        # 2. Equity Curve CSV
        pd.DataFrame(self.equity_curve).to_csv(f'Result/{timestamp}_equity.csv', index=False)


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    print("--- Pattern Recognition Backtest (Enhanced) ---")
    
    # 1. Get Data
    if USE_REAL_DATA:
        # Placeholder for real data fetching logic
        print("Fetching Real Data...")
        # df = fetch_real_data(SYMBOL) 
        df = pd.DataFrame() # fail-safe
    else:
        print(f"Generating Mock Data for {SYMBOL}...")
        df = generate_mock_data(SYMBOL, days=2500)

    # 2. Indicators
    print("Calculating Indicators (Trend, Volume, ATR)...")
    df = calculate_technical_indicators(df)

    # 3. Detect Patterns
    print("Scanning for Cup & Handle...")
    cups = detect_cup_and_handle(df)
    print(f"Found {len(cups)} Cups.")

    print("Scanning for Bullish Rectangles...")
    rects = detect_bullish_rectangle(df)
    print(f"Found {len(rects)} Rectangles.")

    all_signals = cups + rects

    # 4. Run Backtest
    if not all_signals:
        print("No signals found. Try adjusting parameters or regenerating mock data.")
    else:
        bt = EnhancedBacktester(df)
        bt.load_signals(all_signals)
        bt.run()