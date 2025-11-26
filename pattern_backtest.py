"""
Pattern Backtest
- Detects Cup & Handle and Bullish Rectangle patterns on daily Close prices
- Uses similar execution, sizing and risk rules as `Triangle_Advance.py` for backtesting

Usage: run the script directly. By default it uses the mock data generator
from `Triangle_Advance.py`. To use real data, edit the `USE_REAL_DATA` flag
or call `fetch_real_data(symbol)` (helper available in `Triangle_Advance.py`).
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Reuse helpers and constants from Triangle_Advance
from Triangle_Advance import (
    calculate_technical_indicators,
    generate_mock_data,
    fetch_real_data,
    analyze_market_regime,
    ATR_PERIOD,
    SL_ATR_MULTIPLIER,
    TP_RR_RATIO,
    MAX_HOLD_DAYS,
)

# Parameters for pattern detection
CUP_MIN_LEN = 40
CUP_MAX_LEN = 250
CUP_RIM_TOLERANCE = 0.07   # rims within 7%
CUP_BOTTOM_DROP = 0.10     # bottom at least 10% below rims
HANDLE_MAX_DROP = 0.06     # handle must not drop more than 6% from rim
RECT_MIN_DAYS = 20
RECT_MAX_DAYS = 80
RECT_RANGE_TOL = 0.06      # range (max-min)/mean < 6%
BREAKOUT_BUFFER = 0.01     # breakout must exceed resistance by 1%


def detect_cup_and_handle(df: pd.DataFrame):
    """Heuristic cup & handle detector.
    Returns list of signals: dicts with keys: date (breakout), entry_price, type, meta
    """
    signals = []
    close = df['Close'].values
    dates = df.index
    n = len(df)

    # Slide window for possible cups
    for start in range(0, n - CUP_MIN_LEN):
        # try a few cup lengths
        for cup_len in range(CUP_MIN_LEN, min(CUP_MAX_LEN, n - start), 10):
            end = start + cup_len
            if end >= n:
                break

            window = close[start:end]
            if len(window) < CUP_MIN_LEN:
                continue

            left_region = window[: max(3, int(0.25 * len(window)))]
            right_region = window[- max(3, int(0.25 * len(window))):]
            left_max = left_region.max()
            right_max = right_region.max()
            rim_mean = (left_max + right_max) / 2.0

            # rim symmetry
            if abs(left_max - right_max) / rim_mean > CUP_RIM_TOLERANCE:
                continue

            bottom = window.min()
            bottom_idx_rel = window.argmin()
            # depth
            if (rim_mean - bottom) / rim_mean < CUP_BOTTOM_DROP:
                continue

            # Handle region: after bottom to end
            handle_region = window[bottom_idx_rel:]
            if len(handle_region) < 5:
                continue
            handle_min = handle_region.min()
            handle_drop = (rim_mean - handle_min) / rim_mean
            if handle_drop > HANDLE_MAX_DROP:
                continue

            # Candidate breakout at `end` (next day) if price breaks above rim
            if end < n and close[end] > right_max * (1 + BREAKOUT_BUFFER):
                signal = {
                    'type': 'cup_handle',
                    'date': dates[end],
                    'entry_price': float(close[end]),
                    'cup_start': dates[start],
                    'cup_bottom': dates[start + bottom_idx_rel],
                    'rim_price': float(rim_mean),
                    'window_len': cup_len,
                }
                signals.append(signal)

    # Deduplicate signals by date (keep first)
    unique = {}
    for s in signals:
        unique.setdefault(s['date'], s)
    return list(unique.values())


def detect_bullish_rectangle(df: pd.DataFrame):
    """Detect simple bullish rectangles (sideways consolidation) and breakout signals.
    Returns list of signals similar to cup handler.
    """
    signals = []
    close = df['Close'].values
    dates = df.index
    n = len(df)

    for window_len in range(RECT_MIN_DAYS, min(RECT_MAX_DAYS, n // 2), 5):
        for start in range(0, n - window_len - 1):
            end = start + window_len
            window = close[start:end]
            mean = window.mean()
            rng = window.max() - window.min()
            range_pct = rng / mean if mean != 0 else 1.0
            if range_pct > RECT_RANGE_TOL:
                continue

            # Count touches near upper and lower bounds
            upper = window.max()
            lower = window.min()
            upper_touches = np.sum(np.isclose(window, upper, atol=upper * 0.01))
            lower_touches = np.sum(np.isclose(window, lower, atol=lower * 0.01))

            if upper_touches < 2 or lower_touches < 2:
                continue

            # breakout on next day
            if end < n and close[end] > upper * (1 + BREAKOUT_BUFFER):
                signal = {
                    'type': 'bullish_rectangle',
                    'date': dates[end],
                    'entry_price': float(close[end]),
                    'rect_start': dates[start],
                    'rect_end': dates[end - 1],
                    'resistance': float(upper),
                    'support': float(lower),
                    'window_len': window_len,
                }
                signals.append(signal)

    unique = {}
    for s in signals:
        unique.setdefault(s['date'], s)
    return list(unique.values())


# Backtester: reuse execution & sizing rules similar to Triangle_Advance
class PatternBacktester:
    def __init__(self, stock_df: pd.DataFrame, market_df: pd.DataFrame):
        self.df = calculate_technical_indicators(stock_df)
        self.market_df = market_df
        self.initial_capital = 100000.0
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.open_trade = None

    def execute_trade_entry(self, date, price, direction, atr):
        risk_per_share = atr * SL_ATR_MULTIPLIER
        risk_amt = self.capital * 0.01
        shares = risk_amt / risk_per_share if risk_per_share > 0 else 0

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
            'atr': atr,
        }
        print(f"[{date.date()}] ENTER {direction} @ ${price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")

    def manage_open_trade(self, date, row):
        if not self.open_trade:
            return
        t = self.open_trade
        exit_price = None
        reason = None

        if t['direction'] == 'LONG':
            if row['Low'] <= t['sl']:
                exit_price, reason = t['sl'], 'SL'
            elif row['High'] >= t['tp']:
                exit_price, reason = t['tp'], 'TP'
        else:
            if row['High'] >= t['sl']:
                exit_price, reason = t['sl'], 'SL'
            elif row['Low'] <= t['tp']:
                exit_price, reason = t['tp'], 'TP'

        t['days_held'] += 1
        if not reason and t['days_held'] > MAX_HOLD_DAYS:
            exit_price, reason = row['Close'], 'TIME'

        if exit_price:
            pnl = (exit_price - t['entry_price']) * t['shares']
            if t['direction'] == 'SHORT':
                pnl *= -1

            capital_before = self.capital
            self.capital += pnl
            capital_after = self.capital

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
                'capital_after': capital_after,
            }
            self.trades.append(trade_record)
            self.open_trade = None
            print(f"[{date.date()}] EXIT {reason} PnL: ${pnl:.2f} | Capital: ${capital_after:,.2f}")

    def run(self, signals):
        # signals: list of dicts with date, entry_price, type
        sig_by_date = {s['date']: s for s in signals}

        for i in range(len(self.df)):
            current_date = self.df.index[i]
            row = self.df.iloc[i]

            # manage positions
            self.manage_open_trade(current_date, row)

            # if no open trade and signal today, enter
            if not self.open_trade and current_date in sig_by_date:
                s = sig_by_date[current_date]
                atr = row.get('ATR', None)
                if pd.isna(atr) or atr is None:
                    atr = self.df['ATR'].iloc[i] if 'ATR' in self.df.columns else 0
                # use breakout entry price
                self.execute_trade_entry(current_date, s['entry_price'], 'LONG', atr)

            self.equity_curve.append({'Date': current_date, 'Equity': self.capital})

        # generate report and save csvs
        self.generate_report()

    def generate_report(self):
        eq_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        os.makedirs('Result', exist_ok=True)
        current_date = datetime.now().strftime('%Y%m%d')

        equity_file = f'Result/{current_date}_pattern_equity.csv'
        trades_file = f'Result/{current_date}_pattern_trades.csv'
        eq_df.to_csv(equity_file, index=False)
        trades_df.to_csv(trades_file, index=False)
        print(f'Saved equity -> {equity_file}')
        print(f'Saved trades -> {trades_file}')

        # summary
        summary = {}
        if not trades_df.empty:
            summary['total_trades'] = len(trades_df)
            summary['total_pnl'] = trades_df['pnl'].sum()
            summary['avg_pnl'] = trades_df['pnl'].mean()
            summary['win_rate'] = (trades_df['pnl'] > 0).mean()
            summary['avg_days_held'] = trades_df['days_held'].mean()
            summary['final_capital'] = self.capital
            dir_break = trades_df.groupby('direction')['pnl'].agg(['count', 'sum', 'mean']).reset_index()
        else:
            summary['total_trades'] = 0
            summary['total_pnl'] = 0
            summary['avg_pnl'] = 0
            summary['win_rate'] = 0
            summary['avg_days_held'] = 0
            summary['final_capital'] = self.capital
            dir_break = pd.DataFrame()

        summary_df = pd.DataFrame([summary])
        summary_file = f'Result/{current_date}_pattern_trades_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f'Saved summary -> {summary_file}')
        if not dir_break.empty:
            dir_file = f'Result/{current_date}_pattern_trades_direction_breakdown.csv'
            dir_break.to_csv(dir_file, index=False)
            print(f'Saved direction breakdown -> {dir_file}')


if __name__ == '__main__':
    # Toggle data source
    USE_REAL_DATA = False
    SYMBOL = 'MOCK_TECH'

    if USE_REAL_DATA:
        stock_df, market_df = fetch_real_data('AAPL', years=10)
    else:
        print('Generating mock data...')
        stock_df, market_df = generate_mock_data(SYMBOL)

    # Prepare data
    stock_df = calculate_technical_indicators(stock_df)

    # Detect patterns
    print('Detecting Cup & Handle patterns...')
    cups = detect_cup_and_handle(stock_df)
    print(f'Found {len(cups)} cup & handle signals')

    print('Detecting Bullish Rectangle patterns...')
    rects = detect_bullish_rectangle(stock_df)
    print(f'Found {len(rects)} rectangle signals')

    signals = cups + rects
    print(f'Total pattern signals: {len(signals)}')

    # Run backtest using same execution rules
    bt = PatternBacktester(stock_df, market_df)
    bt.run(signals)
