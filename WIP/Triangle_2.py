import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import os
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.simplefilter('ignore')

# --- CONFIGURATION ---

@dataclass
class StrategyConfig:
    # Asset
    ticker: str = "AAPL"
    
    # Pattern Settings
    lookback_months: int = 6       # How much history to look at for the pattern
    pivot_window: int = 10         # Days to confirm a local top/bottom
    min_pivots: int = 3            # Min points to define a line
    consolidation_days: int = 20   # Check for volume contraction in this tail
    min_slope_diff: float = 0.0001 # Minimum angle between lines to call it a triangle
    
    # Filters
    score_threshold: int = 70
    vol_contraction_thresh: float = 0.85
    apex_min_maturity: float = 0.60
    apex_max_maturity: float = 0.95
    market_sma_period: int = 200
    
    # Risk Management
    atr_period: int = 14
    initial_risk_sl: float = 2.0   # Initial Stop Loss in ATR
    trailing_sl_mult: float = 3.0  # Trailing Stop in ATR
    risk_per_trade: float = 0.01   # 1% of equity
    max_hold_days: int = 60
    
    # Paths
    db_path: str = r"C:\Users\jv2mk\OneDrive\Stock\Screener\DB\stage2"
    output_dir: str = "Result"

# --- HELPER FUNCTIONS ---

def calculate_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Calculates ATR, OBV, and Market Regime indicators."""
    df = df.copy()
    
    # 1. ATR
    df['H_L'] = df['High'] - df['Low']
    df['H_PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L_PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H_L', 'H_PC', 'L_PC']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span=cfg.atr_period, adjust=False).mean()

    # 2. OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return df

def get_pivot_points(series: pd.Series, window: int, high_low: str) -> pd.Series:
    """
    Vectorized Pivot Detection.
    Returns a Series where non-pivot indices are NaN.
    """
    pivots = pd.Series(np.nan, index=series.index)
    
    if high_low == 'high':
        # Roll max with center=True checks if the middle value is the max of the window
        rolling = series.rolling(window=2*window+1, center=True).max()
        is_pivot = (series == rolling)
    else:
        rolling = series.rolling(window=2*window+1, center=True).min()
        is_pivot = (series == rolling)
    
    pivots[is_pivot] = series[is_pivot]
    return pivots

def get_boundary_line(x_values, y_values, high_low: str):
    """
    Calculates a 'True' Trendline (Outer Tangent).
    Instead of regression (which cuts price), this connects the best outer points.
    """
    if len(x_values) < 2: return 0.0, 0.0
    
    # Sort points by time
    points = sorted(zip(x_values, y_values))
    
    # LOGIC: Connect the Global Extreme and the Last Extreme
    # This ensures the line covers the range and respects recent price action.
    
    if high_low == 'high':
        # Find the highest peak (Global Max)
        idx_max = np.argmax([p[1] for p in points])
        p1 = points[idx_max]
        
        # Determine P2: The last pivot. 
        # (If Last Pivot == Global Max, take the second to last)
        if idx_max == len(points) - 1:
            p2 = points[-2]
        else:
            p2 = points[-1]
            
    else: # Lows
        # Find the lowest trough (Global Min)
        idx_min = np.argmin([p[1] for p in points])
        p1 = points[idx_min]
        
        if idx_min == len(points) - 1:
            p2 = points[-2]
        else:
            p2 = points[-1]

    # Calculate Slope (m) and Intercept (c)
    # Avoid vertical line error
    if (p2[0] - p1[0]) == 0: return 0.0, 0.0
    
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    intercept = p1[1] - (slope * p1[0])
    
    return slope, intercept

def calculate_apex_maturity(start_idx, current_idx, m1, c1, m2, c2) -> float:
    """Calculates pattern maturity (0.0 to 1.0)."""
    if abs(m1 - m2) < 1e-9: return 0.0
    
    # Intersection X = (c2 - c1) / (m1 - m2)
    apex_x = (c2 - c1) / (m1 - m2)
    
    total_len = apex_x - start_idx
    curr_len = current_idx - start_idx
    
    if total_len == 0: return 0.0
    return curr_len / total_len

# --- VISUALIZATION (PLOTLY) ---

def plot_interactive_chart(df_window, high_pivs, low_pivs, lines, signal, date, cfg):
    """Generates an HTML interactive chart."""
    m_res, c_res, m_sup, c_sup = lines
    
    # Calc Line Arrays
    start_offset = df_window.index[0]
    days = (df_window.index - start_offset).days
    res_line = m_res * days + c_res
    sup_line = m_sup * days + c_sup
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_window.index, open=df_window['Open'], high=df_window['High'],
        low=df_window['Low'], close=df_window['Close'], name='Price'
    ), row=1, col=1)
    
    # Pivots
    fig.add_trace(go.Scatter(x=high_pivs.index, y=high_pivs.values, mode='markers', 
                             marker=dict(color='red', size=8, symbol='triangle-down'), name='High Pivots'), row=1, col=1)
    fig.add_trace(go.Scatter(x=low_pivs.index, y=low_pivs.values, mode='markers', 
                             marker=dict(color='green', size=8, symbol='triangle-up'), name='Low Pivots'), row=1, col=1)
    
    # Trendlines
    fig.add_trace(go.Scatter(x=df_window.index, y=res_line, mode='lines', 
                             line=dict(color='red', width=1, dash='dash'), name='Resistance'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_window.index, y=sup_line, mode='lines', 
                             line=dict(color='green', width=1, dash='dash'), name='Support'), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df_window.index, y=df_window['Volume'], marker_color='rgba(100,100,100,0.5)', name='Volume'), row=2, col=1)

    fig.update_layout(title=f"{cfg.ticker} {signal} Setup | {date.date()}", 
                      xaxis_rangeslider_visible=False, height=800)
    
    # Save
    chart_dir = os.path.join(cfg.output_dir, "Charts")
    os.makedirs(chart_dir, exist_ok=True)
    fname = f"{date.strftime('%Y%m%d')}_{cfg.ticker}_{signal}.html"
    fig.write_html(os.path.join(chart_dir, fname))

# --- BACKTESTER ENGINE ---

class AdvancedBacktester:
    def __init__(self, stock_df, market_df, config: StrategyConfig):
        self.cfg = config
        self.df = calculate_indicators(stock_df, config)
        self.market_df = market_df
        
        self.capital = 100000.0
        self.equity_curve = []
        self.trades = []
        
        # State
        self.open_trade = None   # dict
        self.pending_order = None # dict
        
        # Pre-calc Pivots
        print("    > Identifying Pivot Points...")
        self.pivots_high = get_pivot_points(self.df['High'], self.cfg.pivot_window, 'high')
        self.pivots_low = get_pivot_points(self.df['Low'], self.cfg.pivot_window, 'low')

    def analyze_market_regime(self, date):
        """Returns BULLISH/BEARISH based on SPY SMA200."""
        # Find nearest date
        idx = self.market_df.index.get_indexer([date], method='pad')[0]
        if idx == -1: return 'NEUTRAL'
        
        row = self.market_df.iloc[idx]
        if pd.isna(row['SMA_200']): return 'NEUTRAL'
        return 'BULLISH' if row['Close'] > row['SMA_200'] else 'BEARISH'

    def scan_pattern(self, current_idx):
        """Core Logic: Detects Triangle Pattern."""
        i = current_idx
        # Define Time Window
        start_idx = i - (self.cfg.lookback_months * 22)
        if start_idx < 0: return None
        
        df_window = self.df.iloc[start_idx : i + 1]
        
        # 1. Get VALID Pivots (Avoid Lookahead)
        # A pivot at time T is only confirmed at T + PivotWindow.
        # So we can only see pivots where pivot_date <= current_date - pivot_window
        cutoff_date = self.df.index[i - self.cfg.pivot_window]
        
        win_high = self.pivots_high.loc[df_window.index]
        win_low = self.pivots_low.loc[df_window.index]
        
        valid_high = win_high[win_high.index <= cutoff_date].dropna()
        valid_low = win_low[win_low.index <= cutoff_date].dropna()
        
        if len(valid_high) < self.cfg.min_pivots or len(valid_low) < self.cfg.min_pivots:
            return None

        # 2. Fit Boundary Lines (Outer Tangents)
        # Convert dates to days offset for calculation
        start_date = df_window.index[0]
        
        hx = (valid_high.index - start_date).days
        hy = valid_high.values
        m_res, c_res = get_boundary_line(hx, hy, 'high')
        
        lx = (valid_low.index - start_date).days
        ly = valid_low.values
        m_sup, c_sup = get_boundary_line(lx, ly, 'low')
        
        # 3. Check Convergence
        # Resistance should slope down (negative) or be flat
        # Support should slope up (positive) or be flat
        # Or simply: High Slope < Low Slope
        if not (m_res < m_sup): return None
        if abs(m_res - m_sup) < self.cfg.min_slope_diff: return None

        # 4. Filters (Maturity & Volume)
        window_days = (df_window.index[-1] - start_date).days
        maturity = calculate_apex_maturity(0, window_days, m_res, c_res, m_sup, c_sup)
        
        if maturity < self.cfg.apex_min_maturity: return None
        if maturity > self.cfg.apex_max_maturity: return None
        
        # Volume Contraction
        vol_recent = df_window['Volume'].iloc[-self.cfg.consolidation_days:].mean()
        vol_avg = df_window['Volume'].mean()
        if (vol_recent / (vol_avg+1)) > self.cfg.vol_contraction_thresh:
            return None # Failed volume check

        # 5. Check Proximity (Signal)
        curr_price = df_window['Close'].iloc[-1]
        res_price = m_res * window_days + c_res
        sup_price = m_sup * window_days + c_sup
        
        signal = 'NONE'
        entry_price = 0
        
        # Pressing Resistance?
        if curr_price >= res_price * 0.985 and curr_price <= res_price * 1.02:
            signal = 'LONG'
            entry_price = res_price
        # Pressing Support?
        elif curr_price <= sup_price * 1.015 and curr_price >= sup_price * 0.98:
            signal = 'SHORT'
            entry_price = sup_price
            
        if signal == 'NONE': return None
        
        return {
            'signal': signal,
            'entry_price': entry_price,
            'lines': (m_res, c_res, m_sup, c_sup),
            'df_window': df_window,
            'pivots': (valid_high, valid_low)
        }

    def run(self):
        start_sim = max(self.cfg.market_sma_period, self.cfg.lookback_months * 22)
        print(f"    > Starting simulation on {len(self.df) - start_sim} days...")
        
        for i in range(start_sim, len(self.df)):
            curr_date = self.df.index[i]
            row = self.df.iloc[i]
            
            # --- 1. Manage Open Trade (Trailing Stop) ---
            if self.open_trade:
                t = self.open_trade
                exit_price = None
                reason = None
                
                # Update Trailing Stop
                atr_current = row['ATR']
                if pd.isna(atr_current): atr_current = t['atr_entry']
                
                trail_dist = atr_current * self.cfg.trailing_sl_mult
                
                if t['direction'] == 'LONG':
                    # Ratchet up
                    new_sl = row['High'] - trail_dist
                    if new_sl > t['sl']: t['sl'] = new_sl
                    # Check Hit
                    if row['Low'] <= t['sl']: exit_price, reason = t['sl'], 'Trail_SL'
                    
                else: # SHORT
                    # Ratchet down
                    new_sl = row['Low'] + trail_dist
                    if new_sl < t['sl']: t['sl'] = new_sl
                    # Check Hit
                    if row['High'] >= t['sl']: exit_price, reason = t['sl'], 'Trail_SL'
                
                # Time Exit
                t['days'] += 1
                if not reason and t['days'] > self.cfg.max_hold_days:
                    exit_price, reason = row['Close'], 'Time_Limit'
                
                if exit_price:
                    pnl = (exit_price - t['entry']) * t['shares']
                    if t['direction'] == 'SHORT': pnl *= -1
                    self.capital += pnl
                    self.trades.append({
                        'entry_date': t['date'], 'exit_date': curr_date,
                        'dir': t['direction'], 'pnl': pnl, 'reason': reason
                    })
                    self.open_trade = None
                    print(f"[{curr_date.date()}] EXIT {reason} (${pnl:.2f})")

            # --- 2. Check Pending Order (Retest) ---
            elif self.pending_order:
                po = self.pending_order
                # Expire
                if (curr_date - po['created']).days > 5:
                    self.pending_order = None
                else:
                    filled = False
                    if po['dir'] == 'LONG' and row['Low'] <= po['price'] <= row['High']:
                        filled = True
                    elif po['dir'] == 'SHORT' and row['Low'] <= po['price'] <= row['High']:
                        filled = True
                        
                    if filled:
                        risk_amt = self.capital * self.cfg.risk_per_trade
                        shares = risk_amt / (po['atr'] * self.cfg.initial_risk_sl)
                        
                        # Initial SL
                        sl = po['price'] - (po['atr']*self.cfg.initial_risk_sl) if po['dir']=='LONG' else po['price'] + (po['atr']*self.cfg.initial_risk_sl)
                        
                        self.open_trade = {
                            'date': curr_date, 'entry': po['price'],
                            'shares': shares, 'direction': po['dir'],
                            'sl': sl, 'atr_entry': po['atr'], 'days': 0
                        }
                        self.pending_order = None
                        print(f"[{curr_date.date()}] ENTER {po['dir']} @ {po['price']:.2f}")

            # --- 3. Scan for New Pattern ---
            if not self.open_trade and not self.pending_order:
                res = self.scan_pattern(i)
                if res:
                    # Filter by Market Regime
                    regime = self.analyze_market_regime(curr_date)
                    valid_regime = False
                    if res['signal'] == 'LONG' and regime != 'BEARISH': valid_regime = True
                    if res['signal'] == 'SHORT' and regime != 'BULLISH': valid_regime = True
                    
                    if valid_regime:
                        print(f"[{curr_date.date()}] SIGNAL DETECTED: {res['signal']} (Regime: {regime})")
                        
                        # Plot
                        plot_interactive_chart(
                            res['df_window'], res['pivots'][0], res['pivots'][1],
                            res['lines'], res['signal'], curr_date, self.cfg
                        )
                        
                        # Set Pending
                        self.pending_order = {
                            'created': curr_date,
                            'price': res['entry_price'],
                            'dir': res['signal'],
                            'atr': row['ATR']
                        }

            # Record Equity
            self.equity_curve.append({'Date': curr_date, 'Equity': self.capital})

        self.report()

    def report(self):
        if not self.equity_curve: return
        eq = pd.DataFrame(self.equity_curve).set_index('Date')
        
        # Metrics
        eq['Ret'] = eq['Equity'].pct_change().fillna(0)
        total_ret = (self.capital - 100000) / 100000
        
        # Max DD
        eq['Peak'] = eq['Equity'].cummax()
        eq['DD'] = (eq['Equity'] - eq['Peak']) / eq['Peak']
        max_dd = eq['DD'].min()
        
        # Sharpe (Annualized)
        sharpe = (eq['Ret'].mean() / eq['Ret'].std()) * np.sqrt(252) if eq['Ret'].std() != 0 else 0
        
        # Trade Stats
        if self.trades:
            tdf = pd.DataFrame(self.trades)
            win_rate = len(tdf[tdf['pnl'] > 0]) / len(tdf)
            gross_win = tdf[tdf['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(tdf[tdf['pnl'] < 0]['pnl'].sum())
            pf = gross_win / gross_loss if gross_loss > 0 else 0
            
            # Save CSVs
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            tdf.to_csv(os.path.join(self.cfg.output_dir, f"{self.cfg.ticker}_trades.csv"), index=False)
        else:
            win_rate, pf = 0, 0
            
        print("\n" + "="*40)
        print(f" PERFORMANCE REPORT: {self.cfg.ticker}")
        print("="*40)
        print(f" Final Capital:  ${self.capital:,.2f}")
        print(f" Total Return:   {total_ret*100:.2f}%")
        print(f" Sharpe Ratio:   {sharpe:.2f}")
        print(f" Max Drawdown:   {max_dd*100:.2f}%")
        print(f" Trades:         {len(self.trades)}")
        print(f" Win Rate:       {win_rate*100:.2f}%")
        print(f" Profit Factor:  {pf:.2f}")
        print("="*40)
        print(f" Charts saved to: {os.path.join(self.cfg.output_dir, 'Charts')}")

# --- DATA LOADER ---

def load_data_robust(ticker, db_path):
    print(f"--- Loading Data for {ticker} ---")
    stock_df, market_df = None, None

    # 1. Try SQLite
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            q1 = f"SELECT Date, Open, High, Low, Close, Volume FROM Historical_SP500 WHERE Ticker='{ticker}' ORDER BY Date"
            stock_df = pd.read_sql_query(q1, conn, parse_dates=['Date']).set_index('Date')
            
            q2 = "SELECT Date, Close FROM Historical_SP500 WHERE Ticker='SPY' ORDER BY Date"
            market_df = pd.read_sql_query(q2, conn, parse_dates=['Date']).set_index('Date')
            conn.close()
            print("    > Loaded from local DB.")
        except Exception as e:
            print(f"    ! DB Error: {e}")

    # 2. Fallback YFinance
    if stock_df is None or stock_df.empty:
        print("    > Downloading from YFinance...")
        start = datetime.now() - timedelta(days=365*10)
        try:
            stock_df = yf.download(ticker, start=start, progress=False, multi_level_index=False)
            market_df = yf.download('SPY', start=start, progress=False, multi_level_index=False)
            
            stock_df = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            market_df = market_df[['Close']].dropna()
        except Exception as e:
            print(f"    ! Download failed: {e}")
            return None, None
            
    # Prep Market Data
    market_df['SMA_200'] = market_df['Close'].rolling(200).mean()
    
    return stock_df, market_df

# --- MAIN ---

if __name__ == '__main__':
    # 1. Setup
    config = StrategyConfig(
        ticker="NVDA",        # Change Ticker Here
        lookback_months=6,
        pivot_window=10
    )
    
    # 2. Load
    stock, market = load_data_robust(config.ticker, config.db_path)
    
    # 3. Run
    if stock is not None:
        bt = AdvancedBacktester(stock, market, config)
        bt.run()