"""
🚀 DUAL-HORIZON DQN BACKTEST - COMPLETE SCRIPT
✅ 5-year backtest (2020-2025) ✅ Saves to SQL ✅ Full analysis
"""

import pandas as pd
import sqlalchemy as sa
import numpy as np
from stable_baselines3 import DQN
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🎯 DUAL-HORIZON DQN BACKTEST")

# ----------------------------
# 1. CONFIGURATION
# ----------------------------

SYMBOL = "SPY"
FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
           'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']
BACKTEST_START = "2020-01-01"
BACKTEST_END = "2025-01-01"
MODEL_NAME = "dqn_spy_dual_horizon"

# ----------------------------
# 2. DATA LOADING
# ----------------------------

def load_data(conn_str: str, symbol: str, start_date: str, end_date: str):
    """Load price data from SQL."""
    print(f"📊 Loading {symbol} ({start_date} to {end_date})...")
    
    engine = sa.create_engine(conn_str)
    
    query = f"""
    SELECT TradeDate, [close], volume
    FROM AI_ETF_Prices
    WHERE symbol = '{symbol}' AND TradeDate >= '{start_date}' AND TradeDate <= '{end_date}'
    ORDER BY TradeDate
    """
    
    try:
        df = pd.read_sql(query, engine)
        if len(df) == 0:
            raise ValueError(f"No data found for {symbol}")
        
        df['TradeDate'] = pd.to_datetime(df['TradeDate'])
        print(f"✅ Loaded {len(df)} days")
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

# ----------------------------
# 3. FEATURE ENGINEERING
# ----------------------------

def compute_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 8 technical features + forward labels."""
    print("🔧 Computing features...")
    
    df = raw_df.copy()
    df = df.sort_values('TradeDate').reset_index(drop=True)
    
    if len(df) < 250:
        raise ValueError(f"Need at least 250 days, got {len(df)}")
    
    # ------- 8 FEATURES -------
    
    # 1. Returns
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    
    # 2. Volatility
    df['vol_10d'] = df['ret_1d'].rolling(10, min_periods=5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20, min_periods=10).std()
    
    # 3. Moving Averages (Distance from SMA)
    for win in [20, 50, 200]:
        sma = df['close'].rolling(win, min_periods=win//2).mean()
        df[f'dist_sma_{win}'] = (df['close'] - sma) / (sma + 1e-8)
    
    # 4. Volume Z-Score
    vol_ma = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z_20'] = (df['volume'] - vol_ma) / (vol_std + 1e-8)
    
    # ------- FORWARD LABELS -------
    
    # 5-day forward return
    df['ret_5d_fwd'] = df['close'].shift(-5) / df['close'] - 1.0
    
    # 60-day forward return
    df['ret_60d_fwd'] = df['close'].shift(-60) / df['close'] - 1.0
    
    # Clean NaNs
    result = df[FEATURES + ['TradeDate', 'close', 'ret_5d_fwd', 'ret_60d_fwd']].copy()
    result = result.fillna(method='bfill').fillna(method='ffill').dropna()
    result = result.reset_index(drop=True)
    
    print(f"✅ {len(result)} days with features")
    
    return result

# ----------------------------
# 4. OBSERVATION FUNCTION
# ----------------------------

def get_observation(df: pd.DataFrame, i: int, position: float, equity: float, 
                   feat_mean: np.ndarray, feat_std: np.ndarray):
    """
    Get normalized observation for model.
    
    Observation = [8 normalized features] + [position, equity]
    """
    
    # Extract raw features
    feat_row = df.iloc[i][FEATURES].values.astype(np.float32)
    
    # Normalize
    normalized_features = (feat_row - feat_mean) / feat_std
    
    # Add state
    state = np.array([position, equity], dtype=np.float32)
    
    # Concatenate
    obs = np.concatenate([normalized_features, state]).astype(np.float32)
    
    assert obs.shape == (10,), f"Obs shape {obs.shape} != (10,)"
    return obs

# ----------------------------
# 5. BACKTEST FUNCTION
# ----------------------------

def run_backtest(model_path: str, df: pd.DataFrame, symbol: str = "SPY"):
    """
    Run 5-year backtest on model.
    
    Returns: DataFrame with daily predictions and P&L
    """
    
    print(f"🔮 Loading model: {model_path}")
    
    # Load normalization stats
    feat_mean = np.load("feat_mean.npy")
    feat_std = np.load("feat_std.npy")
    
    assert feat_mean.shape[0] == 8, f"Mean shape {feat_mean.shape[0]} != 8"
    assert feat_std.shape[0] == 8, f"Std shape {feat_std.shape[0]} != 8"
    
    print(f"✅ Loaded normalization stats")
    
    # Load model
    try:
        model = DQN.load(model_path)
        print(f"✅ Loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Initialize trading state
    position = 0.0           # Current position: 0=FLAT, 1=LONG, -1=SHORT
    equity = 1.0            # Portfolio value (starts at $1)
    last_price = df['close'].iloc[0]
    
    predictions = []
    
    print(f"🔮 Running {len(df)} day backtest...")
    
    # Loop through each day
    for i in range(len(df)):
        try:
            # Get data for this day
            trade_date = pd.to_datetime(df['TradeDate'].iloc[i]).date()
            price = float(df['close'].iloc[i])
            ret_5d_actual = float(df['ret_5d_fwd'].iloc[i])
            ret_60d_actual = float(df['ret_60d_fwd'].iloc[i])
            
            # Get observation
            obs = get_observation(df, i, position, equity, feat_mean, feat_std)
            obs = obs.reshape(1, -1)  # (1, 10) for model
            
            # Get model prediction
            action, _ = model.predict(obs, deterministic=True)
            action = int(action[0])
            
            # Map action to position
            # 0 = FLAT, 1 = LONG, 2 = SHORT
            target_pos = [0.0, 1.0, -1.0][action]
            
            # Calculate transaction cost
            pos_change = abs(target_pos - position)
            cost = pos_change * equity * 0.00005  # 0.5 basis points
            
            # Calculate daily return and P&L
            if i > 0:
                ret = (price - last_price) / last_price
            else:
                ret = 0.0
            
            pnl = position * ret * equity
            
            # Update state
            equity += pnl - cost
            position = target_pos
            last_price = price
            
            # Store prediction
            pred_row = {
                'date': trade_date,
                'symbol': symbol,
                'price': price,
                'action': action,
                'signal': ['FLAT', 'LONG', 'SHORT'][action],
                'position': position,
                'position_change': pos_change,
                'daily_pnl': pnl,
                'trade_cost': cost,
                'equity': equity,
                'equity_pct': equity * 100,
                'daily_return': ret * 100,
                'ret_5d_forecast': ret_5d_actual * 100,
                'ret_60d_forecast': ret_60d_actual * 100,
            }
            
            # Add all features
            for feat in FEATURES:
                pred_row[feat] = float(df[feat].iloc[i])
            
            predictions.append(pred_row)
            
            # Progress update
            if (i + 1) % 252 == 0:
                print(f"  Day {i+1:4d}: Equity={equity:.4f} | Signal={['FLAT', 'LONG', 'SHORT'][action]} | Price=${price:.2f}")
        
        except Exception as e:
            print(f"⚠️  Day {i}: {str(e)[:50]}")
            continue
    
    # Create DataFrame
    df_preds = pd.DataFrame(predictions)
    
    print(f"\n✅ Backtest complete: {len(df_preds)} trading days")
    
    return df_preds

# ----------------------------
# 6. ANALYSIS FUNCTION
# ----------------------------

def analyze_backtest(df_preds: pd.DataFrame) -> dict:
    """Analyze backtest results."""
    
    print(f"\n📊 BACKTEST ANALYSIS")
    print(f"{'='*60}")
    
    # Basic stats
    final_equity = df_preds['equity'].iloc[-1]
    total_return = (final_equity - 1.0) * 100
    
    # Sharpe ratio
    daily_returns = df_preds['daily_pnl'] / df_preds['equity'].shift(1)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    
    # Trades
    total_trades = (df_preds['position_change'] > 0).sum()
    
    # Win rate
    wins = (df_preds['daily_pnl'] > 0).sum()
    losses = (df_preds['daily_pnl'] < 0).sum()
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    # Max drawdown
    running_max = df_preds['equity'].expanding().max()
    drawdown = (df_preds['equity'] - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # By signal
    by_signal = df_preds.groupby('signal').agg({
        'daily_pnl': ['count', 'sum', 'mean'],
        'daily_return': 'mean'
    })
    
    # By forecast
    by_5d = df_preds.groupby(pd.cut(df_preds['ret_5d_forecast'], bins=[-np.inf, -0.02, 0, 0.02, np.inf], 
                                    labels=['Very Negative', 'Negative', 'Positive', 'Very Positive'])).agg({
        'daily_pnl': ['count', 'mean'],
        'daily_return': 'mean'
    })
    
    by_60d = df_preds.groupby(pd.cut(df_preds['ret_60d_forecast'], bins=[-np.inf, -0.02, 0, 0.02, np.inf],
                                     labels=['Very Negative', 'Negative', 'Positive', 'Very Positive'])).agg({
        'daily_pnl': ['count', 'mean'],
        'daily_return': 'mean'
    })
    
    # Results
    results = {
        'final_equity': final_equity,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'daily_wins': wins,
        'daily_losses': losses
    }
    
    # Print
    print(f"\n💰 RETURNS:")
    print(f"   Final Equity:    ${final_equity:.4f}")
    print(f"   Total Return:    {total_return:+.2f}%")
    print(f"   Sharpe Ratio:    {sharpe:.2f}")
    
    print(f"\n📈 TRADES:")
    print(f"   Total Trades:    {total_trades}")
    print(f"   Daily Wins:      {wins}")
    print(f"   Daily Losses:    {losses}")
    print(f"   Win Rate:        {win_rate:.1f}%")
    
    print(f"\n📉 RISK:")
    print(f"   Max Drawdown:    {max_drawdown:.2f}%")
    
    print(f"\n📊 BY SIGNAL:")
    print(by_signal)
    
    print(f"\n📊 BY 5-DAY FORECAST:")
    print(by_5d)
    
    print(f"\n📊 BY 60-DAY FORECAST:")
    print(by_60d)
    
    print(f"\n{'='*60}")
    
    return results

# ----------------------------
# 7. SQL SAVE FUNCTION
# ----------------------------

def save_to_sql(conn_str: str, df_preds: pd.DataFrame, run_id: str):
    """Save backtest results to SQL."""
    
    print(f"\n💾 Saving to SQL...")
    
    engine = sa.create_engine(conn_str)
    
    # Add metadata
    df_preds['run_id'] = run_id
    df_preds['run_date'] = datetime.now()
    
    # Ensure date is DATE type (not datetime)
    df_preds['date'] = pd.to_datetime(df_preds['date']).dt.date
    
    # Insert in chunks
    chunk_size = 500
    total_chunks = (len(df_preds) + chunk_size - 1) // chunk_size
    
    print(f"   Inserting {len(df_preds):,} rows in {total_chunks} chunks...")
    
    for i in range(0, len(df_preds), chunk_size):
        chunk = df_preds.iloc[i:i+chunk_size].copy()
        chunk.to_sql('dqn_daily_predictions', engine, if_exists='append', index=False)
        chunk_num = (i // chunk_size) + 1
        print(f"   ✅ Chunk {chunk_num}/{total_chunks}")
    
    # Save summary
    final_equity = df_preds['equity'].iloc[-1]
    total_return = (final_equity - 1.0) * 100
    daily_returns = df_preds['daily_pnl'] / df_preds['equity'].shift(1)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    trades = (df_preds['position_change'] > 0).sum()
    
    summary = pd.DataFrame([{
        'run_id': run_id,
        'symbol': 'SPY',
        'period_start': df_preds['date'].min(),
        'period_end': df_preds['date'].max(),
        'days': len(df_preds),
        'total_return_pct': total_return,
        'final_equity': final_equity,
        'sharpe_ratio': sharpe,
        'total_trades': int(trades),
        'dqn_trades': int(trades),
        'random_trades': 0,
        'run_date': datetime.now()
    }])
    
    summary.to_sql('dqn_prediction_summary', engine, if_exists='append', index=False)
    
    print(f"\n✅ SAVED TO SQL")
    print(f"   Run ID: {run_id}")
    print(f"   Predictions: 1 table")
    print(f"   Summary: 1 row")

# ----------------------------
# 8. MAIN ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    
    SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    
    try:
        print(f"\n{'='*60}")
        print(f"🚀 DUAL-HORIZON BACKTEST")
        print(f"{'='*60}\n")
        
        # ===== STEP 1: Load Data =====
        print(f"STEP 1: Load backtest data")
        print(f"-" * 60)
        df_raw = load_data(SQL_CONN, SYMBOL, BACKTEST_START, BACKTEST_END)
        
        # ===== STEP 2: Compute Features =====
        print(f"\nSTEP 2: Compute features")
        print(f"-" * 60)
        df = compute_features(df_raw)
        
        if len(df) == 0:
            raise ValueError("No valid data after feature computation!")
        
        # ===== STEP 3: Run Backtest =====
        print(f"\nSTEP 3: Run backtest")
        print(f"-" * 60)
        df_preds = run_backtest(MODEL_NAME, df, SYMBOL)
        
        # ===== STEP 4: Analyze =====
        print(f"\nSTEP 4: Analyze results")
        print(f"-" * 60)
        results = analyze_backtest(df_preds)
        
        # ===== STEP 5: Save to SQL =====
        print(f"\nSTEP 5: Save to SQL")
        print(f"-" * 60)
        run_id = f"backtest_{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_to_sql(SQL_CONN, df_preds, run_id)
        
        # ===== FINAL SUMMARY =====
        print(f"\n{'='*60}")
        print(f"✅ BACKTEST COMPLETE")
        print(f"{'='*60}")
        print(f"\n📊 KEY METRICS:")
        print(f"   Return:        {results['total_return_pct']:+.2f}%")
        print(f"   Final Equity:  ${results['final_equity']:.4f}")
        print(f"   Sharpe Ratio:  {results['sharpe_ratio']:.2f}")
        print(f"   Total Trades:  {results['total_trades']}")
        print(f"   Win Rate:      {results['win_rate']:.1f}%")
        print(f"   Max Drawdown:  {results['max_drawdown']:.2f}%")
        print(f"\n📋 Run ID: {run_id}")
        print(f"\n📈 Query results:")
        print(f"   SELECT * FROM dqn_daily_predictions")
        print(f"   WHERE run_id = '{run_id}'")
        print(f"\n{'='*60}\n")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
