import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import yfinance as yf

# --- STRATEGY CONFIGURATION ---
# Pattern Settings
LOOKBACK_MONTHS = 6
PIVOT_LOOKBACK_WINDOW = 10 
N_PIVOTS_FIT = 5           
CONSOLIDATION_DAYS = 20    
MIN_SLOPE_DIFF = 0.00005   

# Filters
SCORE_THRESHOLD = 75       
VOLUME_CONTRACTION_THRESHOLD = 0.85
APEX_MIN_MATURITY = 0.60   # Breakout must occur at least 60% of the way to apex
APEX_MAX_MATURITY = 0.95   # Breakout must occur before 95% (avoid running out of room)
MARKET_REGIME_SMA = 200    # Days for Market (SPY) Trend

# Risk Management
ATR_PERIOD = 14            
SL_ATR_MULTIPLIER = 2.0    
TP_RR_RATIO = 2.0          
MAX_HOLD_DAYS = 60         

# --- INDICATOR FUNCTIONS ---

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates ATR, OBV, and SMAs in one pass."""
    df = df.copy()
    
    # 1. ATR
    df['H_L'] = df['High'] - df['Low']
    df['H_PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L_PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H_L', 'H_PC', 'L_PC']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span=ATR_PERIOD, adjust=False).mean()

    # 2. On-Balance Volume (OBV)
    # OBV = Previous OBV + Volume (if Close > Prev Close) - Volume (if Close < Prev Close)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # 3. OBV Slope (20-day linear regression for trend)
    # We will compute this dynamically during backtest to avoid lookahead, 
    # but we prep the series here.
    
    return df

def get_pivot_points(series: pd.Series, window: int, high_low: str) -> pd.Series:
    """Identifies strict local extrema."""
    pivots = pd.Series(np.nan, index=series.index)
    for i in range(window, len(series) - window):
        sub_series = series.iloc[i - window : i + window + 1]
        if high_low == 'high' and series.iloc[i] == sub_series.max():
            pivots.iloc[i] = series.iloc[i]
        elif high_low == 'low' and series.iloc[i] == sub_series.min():
            pivots.iloc[i] = series.iloc[i]
    return pivots.dropna()

def calculate_apex_maturity(start_idx, current_idx, high_slope, high_intercept, low_slope, low_intercept) -> float:
    """
    Calculates where we are relative to the triangle's Apex (Intersection).
    Returns 0.0 to 1.0 (1.0 = We are AT the apex).
    """
    # Algebra: mx + c = mx + c  ->  x = (c2 - c1) / (m1 - m2)
    if abs(high_slope - low_slope) < 1e-9: return 0.0 # Parallel lines
    
    apex_x = (low_intercept - high_intercept) / (high_slope - low_slope)
    
    # Length of pattern so far
    pattern_duration = current_idx - start_idx
    # Total distance to apex from start
    total_distance_to_apex = apex_x - start_idx
    
    if total_distance_to_apex == 0: return 1.0
    
    maturity = pattern_duration / total_distance_to_apex
    return maturity

def calculate_obv_slope(obv_series: pd.Series, lookback: int = 20) -> float:
    """Returns the slope of the OBV line over the last N days."""
    y = obv_series.iloc[-lookback:].values
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope

# --- SIGNAL GENERATION LOGIC ---

def analyze_market_regime(market_df: pd.DataFrame, current_date) -> str:
    """
    Checks if the broad market (SPY) is Bullish or Bearish.
    Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL' (if data missing)
    """
    # Locate the market data for this date
    if current_date not in market_df.index:
        return 'NEUTRAL'
    
    # Simple logic: Is Price > 200 SMA?
    row = market_df.loc[current_date]
    if pd.isna(row['SMA_200']): return 'NEUTRAL'
    
    return 'BULLISH' if row['Close'] > row['SMA_200'] else 'BEARISH'

def generate_signal(df_window: pd.DataFrame, market_status: str) -> dict:
    """
    Core Logic: Detects pattern, Applies Filters (Regime, Apex, OBV).
    """
    result = {'signal': 'NONE', 'reason': 'No Pattern', 'atr': 0, 'entry_price': 0}
    
    # 1. PIVOTS & TRENDLINES
    high_pivots = get_pivot_points(df_window['High'], PIVOT_LOOKBACK_WINDOW, 'high')
    low_pivots = get_pivot_points(df_window['Low'], PIVOT_LOOKBACK_WINDOW, 'low')

    if len(high_pivots) < N_PIVOTS_FIT or len(low_pivots) < N_PIVOTS_FIT:
        return result

    # Fit Trendlines
    # Map dates to integer indices relative to the start of the window
    idx_map = np.arange(len(df_window))
    
    high_x = idx_map[df_window.index.isin(high_pivots.index)]
    high_y = high_pivots.values
    high_slope, high_intercept = np.polyfit(high_x[-N_PIVOTS_FIT:], high_y[-N_PIVOTS_FIT:], 1)

    low_x = idx_map[df_window.index.isin(low_pivots.index)]
    low_y = low_pivots.values
    low_slope, low_intercept = np.polyfit(low_x[-N_PIVOTS_FIT:], low_y[-N_PIVOTS_FIT:], 1)

    # 2. CONVERGENCE CHECK
    is_converging = (high_slope < low_slope) and (abs(high_slope - low_slope) > MIN_SLOPE_DIFF)
    if not is_converging: return result

    # 3. SCORING (Tightness/Volume)
    # (Simplified scoring from previous script for brevity, focusing on new filters)
    vol_short = df_window['Volume'].iloc[-CONSOLIDATION_DAYS:].mean()
    vol_long = df_window['Volume'].mean()
    vol_score = 15 if (vol_short / vol_long) < VOLUME_CONTRACTION_THRESHOLD else 0
    
    # Tightness
    last_range = (df_window['High'].iloc[-CONSOLIDATION_DAYS:].max() - df_window['Low'].iloc[-CONSOLIDATION_DAYS:].min())
    tight_score = 45 if (last_range / df_window['Close'].mean()) < 0.05 else 0 # Simple threshold
    
    total_score = 40 + tight_score + vol_score # Base convergence score assumed 40 if passed
    
    if total_score < SCORE_THRESHOLD:
        result['reason'] = 'Low Score'
        return result

    # --- NEW FILTERS ---

    # A. Apex Maturity Filter
    # Where are we relative to the start of the window?
    maturity = calculate_apex_maturity(0, len(df_window), high_slope, high_intercept, low_slope, low_intercept)
    if maturity < APEX_MIN_MATURITY:
        result['reason'] = f'Too Early ({maturity:.2f})'
        return result
    if maturity > APEX_MAX_MATURITY:
        result['reason'] = f'Too Late ({maturity:.2f})'
        return result

    # B. OBV Divergence Filter
    obv_slope = calculate_obv_slope(df_window['OBV'])
    
    # C. Determine Potential Signal
    current_idx = len(df_window) - 1
    resistance_price = high_slope * current_idx + high_intercept
    support_price = low_slope * current_idx + low_intercept
    current_close = df_window['Close'].iloc[-1]
    
    potential_signal = 'NONE'
    if current_close > resistance_price * 0.99: # Pressing Resistance
        potential_signal = 'LONG'
    elif current_close < support_price * 1.01: # Pressing Support
        potential_signal = 'SHORT'
        
    if potential_signal == 'NONE': return result

    # D. Apply Market Regime & OBV Confirmation
    if potential_signal == 'LONG':
        if market_status == 'BEARISH':
            result['reason'] = 'Market Bearish'
            return result
        if obv_slope < 0: # Price pressing up, but OBV going down
            result['reason'] = 'OBV Divergence'
            return result
            
    elif potential_signal == 'SHORT':
        if market_status == 'BULLISH':
            result['reason'] = 'Market Bullish'
            return result
        if obv_slope > 0:
            result['reason'] = 'OBV Divergence'
            return result

    # If we passed all gauntlets:
    result['signal'] = potential_signal
    result['atr'] = df_window['ATR'].iloc[-1]
    # IMPORTANT: The "entry_price" is the TRENDLINE price, not current price.
    # We will set a Limit Order at this price.
    result['entry_price'] = resistance_price if potential_signal == 'LONG' else support_price
    result['reason'] = 'Confirmed'
    
    return result

# --- MOCK DATA GENERATION ---

def generate_mock_data(symbol, days=3650, market_trend='BULL'):
    """Generates synthetic price data for Stock + SPY."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # 1. Generate SPY (Benchmark)
    spy_base = 200
    spy_prices = [spy_base]
    for _ in range(1, len(dates)):
        change = np.random.normal(0.0003, 0.01) # Slight upward drift
        spy_prices.append(spy_prices[-1] * (1 + change))
    
    spy_df = pd.DataFrame({'Date': dates, 'Close': spy_prices})
    spy_df.set_index('Date', inplace=True)
    spy_df['SMA_200'] = spy_df['Close'].rolling(window=200).mean()

    # 2. Generate Target Stock (Correlated but with patterns)
    stock_base = 50
    prices = [stock_base]
    volumes = [1000000]
    
    for i in range(1, len(dates)):
        # Correlation to SPY
        spy_change = (spy_prices[i] - spy_prices[i-1]) / spy_prices[i-1]
        
        # Add idiosyncratic noise and occasional consolidation/breakout
        noise = np.random.normal(0, 0.015)
        
        # inject pattern logic (simplified)
        if i % 200 > 150: # Artificial Consolidation
            noise = noise * 0.2 # low vol
        
        change = spy_change * 0.8 + noise 
        prices.append(prices[-1] * (1 + change))
        
        # Volume logic
        vol_change = np.random.normal(0, 0.1)
        volumes.append(volumes[-1] * (1 + vol_change))

    df = pd.DataFrame({'Date': dates, 'Close': prices, 'Volume': volumes})
    df['High'] = df['Close'] * (1 + np.abs(np.random.normal(0, 0.01, len(df))))
    df['Low'] = df['Close'] * (1 - np.abs(np.random.normal(0, 0.01, len(df))))
    df.set_index('Date', inplace=True)
    
    return df, spy_df


def fetch_real_data(symbol: str, years: int = 10):
    """Fetches real OHLCV for `symbol` and SPY (market) using yfinance.
    Returns (stock_df, market_df) with expected columns and Date index.
    """
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)

    stock = yf.download(symbol, start=start, end=end, progress=False)
    market = yf.download('SPY', start=start, end=end, progress=False)

    if stock is None or stock.empty:
        raise RuntimeError(f"Failed to download data for {symbol}")

    # Ensure expected column names and index
    stock = stock.rename(columns={c: c for c in stock.columns})
    stock = stock[['High', 'Low', 'Close', 'Volume']].copy()
    stock.index = pd.to_datetime(stock.index)

    market = market[['Close']].copy()
    market.index = pd.to_datetime(market.index)
    market['SMA_200'] = market['Close'].rolling(window=200).mean()

    return stock, market

# --- BACKTESTER ENGINE ---

class AdvancedBacktester:
    def __init__(self, stock_df, market_df):
        self.df = calculate_technical_indicators(stock_df)
        self.market_df = market_df
        self.initial_capital = 100000.0
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        
        # Order Management
        self.open_trade = None
        self.pending_order = None # Logic for Retest Entry

    def process_pending_order(self, current_date, row):
        """
        Simulates Limit Order logic.
        If we have a pending order at $100, and today's Low is $99, we get filled at $100.
        """
        if not self.pending_order: return
        
        # Expire order if pending too long (e.g., 5 days)
        if (current_date - self.pending_order['created_date']).days > 5:
            self.pending_order = None
            return

        target_price = self.pending_order['price']
        direction = self.pending_order['direction']
        filled = False
        
        # Check for Fill (Retest Logic)
        # For LONG: We want price to dip to our limit (Low <= Target)
        # Note: In real life, if we gap down, we get filled lower. Here we assume limit fill.
        if direction == 'LONG' and row['Low'] <= target_price <= row['High']:
            filled = True
        elif direction == 'SHORT' and row['Low'] <= target_price <= row['High']:
            filled = True
            
        if filled:
            self.execute_trade_entry(current_date, target_price, direction, self.pending_order['atr'])
            self.pending_order = None # Clear order

    def execute_trade_entry(self, date, price, direction, atr):
        risk_per_share = atr * SL_ATR_MULTIPLIER
        # Risk 1% of current capital
        risk_amt = self.capital * 0.01
        shares = risk_amt / risk_per_share
        
        if direction == 'LONG':
            sl = price - risk_per_share
            tp = price + (risk_per_share * TP_RR_RATIO)
        else:
            sl = price + risk_per_share
            tp = price - (risk_per_share * TP_RR_RATIO)
            
        self.open_trade = {
            'entry_date': date,
            'entry_price': price,
            'shares': shares,
            'direction': direction,
            'sl': sl,
            'tp': tp,
            'days_held': 0,
            'atr': atr
        }
        print(f"[{date.date()}] ENTER {direction} (Retest Fill) @ ${price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")

    def manage_open_trade(self, date, row):
        if not self.open_trade: return
        
        t = self.open_trade
        exit_price = None
        reason = None
        
        # Check SL/TP
        if t['direction'] == 'LONG':
            if row['Low'] <= t['sl']: exit_price, reason = t['sl'], 'SL'
            elif row['High'] >= t['tp']: exit_price, reason = t['tp'], 'TP'
        else:
            if row['High'] >= t['sl']: exit_price, reason = t['sl'], 'SL'
            elif row['Low'] <= t['tp']: exit_price, reason = t['tp'], 'TP'
            
        # Check Time
        t['days_held'] += 1
        if not reason and t['days_held'] > MAX_HOLD_DAYS:
            exit_price, reason = row['Close'], 'TIME'
            
        if exit_price:
            pnl = (exit_price - t['entry_price']) * t['shares']
            if t['direction'] == 'SHORT':
                pnl *= -1

            # Capture capital before applying PnL
            capital_before = self.capital
            self.capital += pnl
            capital_after = self.capital

            # Build full trade record including entry and exit details
            trade_record = {
                'entry_date': t['entry_date'],
                'entry_price': t['entry_price'],
                'exit_date': date,
                'exit_price': exit_price,
                'direction': t['direction'],
                'shares': t['shares'],
                'sl': t['sl'],
                'tp': t['tp'],
                'atr': t.get('atr', None),
                'days_held': t.get('days_held', None),
                'pnl': pnl,
                'reason': reason,
                'capital_before': capital_before,
                'capital_after': capital_after
            }
            self.trades.append(trade_record)
            self.open_trade = None
            print(f"[{date.date()}] EXIT {reason} PnL: ${pnl:.2f} | Capital: ${capital_after:,.2f}")

    def run(self):
        # Start after enough data for Market SMA(200)
        start_idx = max(MARKET_REGIME_SMA, LOOKBACK_MONTHS * 22)
        
        for i in range(start_idx, len(self.df)):
            current_date = self.df.index[i]
            row = self.df.iloc[i]
            
            # 1. Manage Positions
            self.manage_open_trade(current_date, row)
            
            # 2. Check Pending Orders (Retest Entry)
            if not self.open_trade:
                self.process_pending_order(current_date, row)
            
            # 3. Scan for NEW Signals (only if flat and no pending)
            if not self.open_trade and not self.pending_order:
                # Window for analysis
                window_start = i - (LOOKBACK_MONTHS * 22)
                df_window = self.df.iloc[window_start : i]
                
                # Market Regime Check
                mkt_status = analyze_market_regime(self.market_df, current_date)
                
                # Signal Generation
                signal_data = generate_signal(df_window, mkt_status)
                
                if signal_data['signal'] != 'NONE':
                    # CREATE LIMIT ORDER (Wait for Retest)
                    # We set the limit order at the trendline price calculated in signal
                    self.pending_order = {
                        'created_date': current_date,
                        'price': signal_data['entry_price'],
                        'direction': signal_data['signal'],
                        'atr': signal_data['atr']
                    }
                    print(f"[{current_date.date()}] SIGNAL DETECTED ({signal_data['signal']}). Placing Limit Order for Retest @ ${signal_data['entry_price']:.2f}")

            # Record Equity
            self.equity_curve.append({'Date': current_date, 'Equity': self.capital})
            
        self.generate_report()

    def generate_report(self):
        if not self.equity_curve: return
        
        eq_df = pd.DataFrame(self.equity_curve)
        
        # Stats
        total_ret = (self.capital - self.initial_capital) / self.initial_capital
        win_trades = [t for t in self.trades if t['pnl'] > 0]
        loss_trades = [t for t in self.trades if t['pnl'] <= 0]
        win_rate = len(win_trades) / len(self.trades) if self.trades else 0
        
        print("\n" + "="*40)
        print("ADVANCED STRATEGY PERFORMANCE REPORT")
        print("="*40)
        print(f"Final Capital:   ${self.capital:,.2f}")
        print(f"Total Return:    {total_ret*100:.2f}%")
        print(f"Total Trades:    {len(self.trades)}")
        print(f"Win Rate:        {win_rate*100:.2f}%")
        print("="*40)
        
        # Export trades and equity to CSV (Result folder). Filenames include current date.
        os.makedirs('Result', exist_ok=True)
        current_date = datetime.now().strftime('%Y%m%d')

        trades_df = pd.DataFrame(self.trades)
        # Normalize date columns to YYYY-MM-DD strings if present
        if not trades_df.empty:
            for col in ['entry_date', 'exit_date']:
                if col in trades_df.columns:
                    trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime('%Y-%m-%d')

        equity_file = f"Result/{current_date}_triangle_equity.csv"
        trades_file = f"Result/{current_date}_triangle_trades.csv"
        eq_df.to_csv(equity_file, index=False)
        trades_df.to_csv(trades_file, index=False)
        print(f"Saved equity -> {equity_file}")
        print(f"Saved trades -> {trades_file}")

        # Aggregated summary
        summary = {}
        if not trades_df.empty:
            summary['total_trades'] = len(trades_df)
            summary['total_pnl'] = trades_df['pnl'].sum()
            summary['avg_pnl'] = trades_df['pnl'].mean()
            summary['win_rate'] = (trades_df['pnl'] > 0).mean()
            summary['avg_days_held'] = trades_df['days_held'].mean()
            summary['final_capital'] = self.capital
            # Direction breakdown
            dir_break = trades_df.groupby('direction')['pnl'].agg(['count','sum','mean']).reset_index()
        else:
            summary['total_trades'] = 0
            summary['total_pnl'] = 0
            summary['avg_pnl'] = 0
            summary['win_rate'] = 0
            summary['avg_days_held'] = 0
            summary['final_capital'] = self.capital
            dir_break = pd.DataFrame()

        summary_df = pd.DataFrame([summary])
        summary_file = f"Result/{current_date}_triangle_trades_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved trades summary -> {summary_file}")

        # Save direction breakdown if available
        if not dir_break.empty:
            dir_file = f"Result/{current_date}_triangle_trades_direction_breakdown.csv"
            dir_break.to_csv(dir_file, index=False)
            print(f"Saved direction breakdown -> {dir_file}")

        plt.figure(figsize=(12, 6))
        plt.plot(eq_df['Date'], eq_df['Equity'], label='Strategy Equity')
        plt.title('Advanced Triangle Strategy (with Regime, Apex & Retest Filters)')
        plt.ylabel('Capital ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

# --- EXECUTION ---
if __name__ == '__main__':
    print("Generating 10 years of High-Fidelity Mock Data...")
    stock_df, market_df = generate_mock_data('MOCK_TECH')
    
    print("Running Backtest...")
    bt = AdvancedBacktester(stock_df, market_df)
    bt.run()