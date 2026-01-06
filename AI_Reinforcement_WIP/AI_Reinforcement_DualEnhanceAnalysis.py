"""
Export backtest results with ALL 20 features
"""

import pandas as pd
import sqlalchemy as sa

print("📊 EXPORTING BACKTEST RESULTS WITH ALL FEATURES\n")

# SQL Connection
SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

engine = sa.create_engine(SQL_CONN)

# Get latest advanced backtest run_id
query_run_id = """
SELECT TOP 1 run_id 
FROM dqn_prediction_summary 
WHERE model_version = 'advanced'
ORDER BY run_date DESC
"""

try:
    run_id = pd.read_sql(query_run_id, engine)['run_id'].iloc[0]
    print(f"📋 Found latest run: {run_id}\n")
except Exception as e:
    print(f"❌ Error finding run: {e}")
    print("Make sure you have an 'advanced' model in dqn_prediction_summary")
    exit(1)

# Query to get ALL columns including features
query = f"""
SELECT 
    run_id,
    [date],
    symbol,
    price,
    [action],
    signal,
    [position],
    position_change,
    daily_pnl,
    trade_cost,
    equity,
    equity_pct,
    daily_return,
    ret_5d_forecast,
    ret_60d_forecast,
    ret_1d,
    ret_5d,
    vol_10d,
    vol_20d,
    dist_sma_20,
    dist_sma_50,
    dist_sma_200,
    vol_z_20,
    rsi_14,
    macd_line,
    macd_signal,
    macd_hist,
    atr_14,
    vol_ratio,
    vol_trend,
    adx_14,
    bb_position,
    range_norm,
    ret_skew,
    obv_ratio
FROM dqn_daily_predictions
WHERE run_id = '{run_id}'
ORDER BY [date]
"""

print("📥 Loading data from SQL...")
try:
    df = pd.read_sql(query, engine)
    print(f"✅ Loaded {len(df):,} rows")
    print(f"   Columns: {len(df.columns)}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Check for missing columns
required_cols = [
    'ret_1d', 'ret_5d', 'vol_10d', 'vol_20d',
    'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20',
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'atr_14', 'vol_ratio', 'vol_trend',
    'adx_14', 'bb_position',
    'range_norm', 'ret_skew', 'obv_ratio',
    'daily_pnl'
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"\n⚠️  WARNING: Missing columns: {missing_cols}")
    print(f"   Available columns: {df.columns.tolist()}")
else:
    print(f"\n✅ All required columns present")

# Fill NaN values with 0
df = df.fillna(0)

# Save to CSV
filename = 'backtest_results_advanced.csv'
df.to_csv(filename, index=False)

print(f"\n✅ EXPORTED: {filename}")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")
print(f"   Size: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Show first few rows
print(f"\n📋 FIRST 3 ROWS:")
print(df.head(3))

# Show column names
print(f"\n📊 ALL COLUMNS:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

# Summary stats
print(f"\n📈 DATA SUMMARY:")
print(f"   Run ID: {run_id}")
print(f"   Symbol: {df['symbol'].iloc[0]}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
print(f"   Final Equity: ${df['equity'].iloc[-1]:.4f}")
print(f"   Total Return: {((df['equity'].iloc[-1] - 1.0) * 100):+.2f}%")
print(f"   Max P&L: ${df['daily_pnl'].max():.4f}")
print(f"   Min P&L: ${df['daily_pnl'].min():.4f}")

print(f"\n✅ Ready for feature analysis!")
