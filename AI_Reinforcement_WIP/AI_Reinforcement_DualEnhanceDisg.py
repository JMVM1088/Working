"""
🔍 DIAGNOSTIC: Why is the model losing money?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🔍 BACKTEST DIAGNOSTIC ANALYSIS\n")

# Load data
df = pd.read_csv('backtest_results_advanced.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

print(f"📊 DATASET: {len(df):,} days from {df['date'].min()} to {df['date'].max()}\n")

# ===== 1. CUMULATIVE PERFORMANCE =====
print("="*60)
print("1️⃣  CUMULATIVE PERFORMANCE ANALYSIS")
print("="*60)

df['cumulative_pnl'] = df['daily_pnl'].cumsum()

print(f"   Starting Equity:     $1.0000")
print(f"   Final Equity:        ${df['equity'].iloc[-1]:.4f}")
print(f"   Total P&L:           ${df['cumulative_pnl'].iloc[-1]:.4f}")
print(f"   Total Return:        {((df['equity'].iloc[-1] - 1.0) * 100):+.2f}%")
print(f"   Best Day:            ${df['daily_pnl'].max():.4f}")
print(f"   Worst Day:           ${df['daily_pnl'].min():.4f}")

# ===== 2. BY YEAR =====
print(f"\n{'='*60}")
print("2️⃣  PERFORMANCE BY YEAR")
print("="*60)

df['year'] = df['date'].dt.year

for year in df['year'].unique():
    year_data = df[df['year'] == year]
    year_return = (year_data['equity'].iloc[-1] / year_data['equity'].iloc[0] - 1.0) * 100
    wins = (year_data['daily_pnl'] > 0).sum()
    losses = (year_data['daily_pnl'] < 0).sum()
    
    print(f"   {year}: {year_return:+7.2f}% | {wins:3d} wins, {losses:3d} losses")

# ===== 3. BY SIGNAL =====
print(f"\n{'='*60}")
print("3️⃣  PERFORMANCE BY SIGNAL (FLAT/LONG/SHORT)")
print("="*60)

for signal in ['FLAT', 'LONG', 'SHORT']:
    signal_data = df[df['signal'] == signal]
    
    if len(signal_data) > 0:
        pnl = signal_data['daily_pnl'].sum()
        days = len(signal_data)
        avg_pnl = signal_data['daily_pnl'].mean()
        wins = (signal_data['daily_pnl'] > 0).sum()
        
        print(f"\n   {signal}:")
        print(f"      Days: {days}")
        print(f"      Total P&L: ${pnl:.4f}")
        print(f"      Avg P&L: ${avg_pnl:.6f}")
        print(f"      Win Rate: {wins/days*100:.1f}%")

# ===== 4. FEATURE CORRELATION WITH P&L =====
print(f"\n{'='*60}")
print("4️⃣  FEATURE CORRELATION WITH DAILY P&L")
print("="*60)

features = [
    'ret_1d', 'ret_5d', 'vol_10d', 'vol_20d',
    'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20',
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'atr_14', 'vol_ratio', 'vol_trend',
    'adx_14', 'bb_position',
    'range_norm', 'ret_skew', 'obv_ratio'
]

correlations = {}
for feature in features:
    if feature in df.columns:
        corr = df[feature].corr(df['daily_pnl'])
        correlations[feature] = corr

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n   TOP 10 FEATURES BY P&L CORRELATION:")
for i, (feat, corr) in enumerate(sorted_corr[:10], 1):
    symbol = "✅" if abs(corr) > 0.1 else "⚠️ "
    print(f"   {i:2d}. {feat:20s}: {corr:+.4f} {symbol}")

print("\n   BOTTOM 5 FEATURES (WORST CORRELATION):")
for i, (feat, corr) in enumerate(sorted_corr[-5:], 1):
    print(f"      {feat:20s}: {corr:+.4f}")

# ===== 5. POSITION ANALYSIS =====
print(f"\n{'='*60}")
print("5️⃣  POSITION ANALYSIS")
print("="*60)

flat_days = (df['signal'] == 'FLAT').sum()
long_days = (df['signal'] == 'LONG').sum()
short_days = (df['signal'] == 'SHORT').sum()

print(f"   FLAT:  {flat_days:4d} days ({flat_days/len(df)*100:5.1f}%)")
print(f"   LONG:  {long_days:4d} days ({long_days/len(df)*100:5.1f}%)")
print(f"   SHORT: {short_days:4d} days ({short_days/len(df)*100:5.1f}%)")

# ===== 6. TRANSACTION COSTS =====
print(f"\n{'='*60}")
print("6️⃣  TRANSACTION COSTS")
print("="*60)

total_cost = df['trade_cost'].sum()
total_trades = (df['position_change'] > 0).sum()
cost_per_trade = total_cost / total_trades if total_trades > 0 else 0

print(f"   Total Trades: {total_trades}")
print(f"   Total Cost:   ${total_cost:.4f}")
print(f"   Cost/Trade:   ${cost_per_trade:.6f}")
print(f"   Cost as % of return: {abs(total_cost / ((df['equity'].iloc[-1] - 1.0)) * 100) if (df['equity'].iloc[-1] - 1.0) != 0 else 0:.1f}%")

# ===== 7. VOLATILITY & DRAWDOWN =====
print(f"\n{'='*60}")
print("7️⃣  RISK METRICS")
print("="*60)

daily_returns = df['daily_pnl'] / df['equity'].shift(1)
daily_vol = daily_returns.std() * np.sqrt(252) * 100

running_max = df['equity'].expanding().max()
drawdown = (df['equity'] - running_max) / running_max * 100
max_dd = drawdown.min()

print(f"   Daily Volatility: {daily_vol:.2f}%")
print(f"   Max Drawdown: {max_dd:.2f}%")
print(f"   Sharpe Ratio: {(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0:.2f}")

# ===== 8. COMPARISON: LONG-ONLY BENCHMARK =====
print(f"\n{'='*60}")
print("8️⃣  COMPARISON: WHAT IF WE JUST HELD SPY?")
print("="*60)

# Buy and hold return (opposite of model)
first_price = df['price'].iloc[0]
last_price = df['price'].iloc[-1]
buy_hold_return = (last_price - first_price) / first_price * 100

print(f"   Buy & Hold Return: {buy_hold_return:+.2f}%")
print(f"   Model Return:      {((df['equity'].iloc[-1] - 1.0) * 100):+.2f}%")
print(f"   Underperformance:  {((df['equity'].iloc[-1] - 1.0) * 100) - buy_hold_return:.2f}%")

if buy_hold_return > 0 and ((df['equity'].iloc[-1] - 1.0) * 100) < 0:
    print(f"\n   ⚠️  MODEL LOST MONEY WHILE SPY GAINED!")
    print(f"   🔴 The model is SHORTING at wrong times")

# ===== 9. SIGNAL QUALITY =====
print(f"\n{'='*60}")
print("9️⃣  SIGNAL QUALITY ANALYSIS")
print("="*60)

print(f"\n   When LONG, what happened next?")
long_df = df[df['signal'] == 'LONG'].copy()
if len(long_df) > 0:
    long_wins = (long_df['daily_pnl'] > 0).sum()
    long_win_rate = long_wins / len(long_df) * 100
    print(f"      Win Rate: {long_win_rate:.1f}% ({long_wins}/{len(long_df)})")
    print(f"      Avg P&L: ${long_df['daily_pnl'].mean():.6f}")

print(f"\n   When SHORT, what happened next?")
short_df = df[df['signal'] == 'SHORT'].copy()
if len(short_df) > 0:
    short_wins = (short_df['daily_pnl'] > 0).sum()
    short_win_rate = short_wins / len(short_df) * 100
    print(f"      Win Rate: {short_win_rate:.1f}% ({short_wins}/{len(short_df)})")
    print(f"      Avg P&L: ${short_df['daily_pnl'].mean():.6f}")

print(f"\n   When FLAT, what happened?")
flat_df = df[df['signal'] == 'FLAT'].copy()
if len(flat_df) > 0:
    flat_wins = (flat_df['daily_pnl'] > 0).sum()
    flat_win_rate = flat_wins / len(flat_df) * 100
    print(f"      Win Rate: {flat_win_rate:.1f}% ({flat_wins}/{len(flat_df)})")
    print(f"      Avg P&L: ${flat_df['daily_pnl'].mean():.6f}")

# ===== SUMMARY =====
print(f"\n{'='*60}")
print("🎯 DIAGNOSIS SUMMARY")
print("="*60)

print(f"\n⚠️  PROBLEMS IDENTIFIED:")

if buy_hold_return > 0 and ((df['equity'].iloc[-1] - 1.0) * 100) < 0:
    print(f"   1. Model LOSES when market GAINS")
    print(f"      → Taking SHORT positions at wrong times")

if short_df is not None and len(short_df) > 0 and (short_df['daily_pnl'] > 0).mean() < 0.4:
    print(f"   2. SHORT signals are POOR (win rate <40%)")
    print(f"      → Should reduce or eliminate SHORT signals")

if flat_win_rate < 50:
    print(f"   3. FLAT position worse than random")
    print(f"      → Maybe should hold LONG instead of going FLAT")

if max_dd < -20:
    print(f"   4. Extreme drawdown ({max_dd:.1f}%)")
    print(f"      → Position sizing too aggressive OR bad signal timing")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   1. Check if model trained on BULL market data")
print(f"   2. Remove SHORT signals (not working)")
print(f"   3. Increase FLAT days (safer than bad trades)")
print(f"   4. Re-train on 2015-2024 (include market crashes)")
print(f"   5. Use base model (8 features) - probably better")

print(f"\n{'='*60}\n")
