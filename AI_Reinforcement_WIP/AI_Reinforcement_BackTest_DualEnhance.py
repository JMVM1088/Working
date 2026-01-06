"""
🚀 DUAL-HORIZON DQN BACKTEST - RETRAINED MODEL (2015-2024)
✅ Backtest on 2024 data (unseen) ✅ Full analysis ✅ Compare to old model
"""

import pandas as pd
import sqlalchemy as sa
import numpy as np
from stable_baselines3 import DQN
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🎯 DUAL-HORIZON DQN BACKTEST - RETRAINED MODEL")

# ----------------------------
# 1. CONFIGURATION
# ----------------------------

SYMBOL = "SPY"

# ALL 20 FEATURES
BASE_FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
                 'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']

ADVANCED_FEATURES = [
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'atr_14', 'vol_ratio', 'vol_trend',
    'adx_14', 'bb_position',
    'range_norm', 'ret_skew', 'obv_ratio'
]

ALL_FEATURES = BASE_FEATURES + ADVANCED_FEATURES

# ===== BACKTEST ON 2024 DATA (Unseen by retrained model) =====
BACKTEST_START = "2024-01-01"
BACKTEST_END = "2025-01-01"
MODEL_NAME = "dqn_spy_retrained_2015_2024"

# ----------------------------
# 2. DATA LOADING
# ----------------------------

def load_data(conn_str: str, symbol: str, start_date: str, end_date: str):
    """Load price data from SQL."""
    print(f"📊 Loading {symbol} ({start_date} to {end_date})...")
    
    engine = sa.create_engine(conn_str)
    
    query = f"""
    SELECT TradeDate, [close], [high], [low], volume
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
# 3. TECHNICAL INDICATOR FUNCTIONS
# ----------------------------

def calculate_rsi(prices, period=14):
    """Calculate RSI (normalized to -1, 1)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return (rsi - 50) / 50

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    
    macd_line_norm = macd_line / prices * 100
    macd_signal_norm = macd_signal / prices * 100
    macd_hist_norm = macd_hist / prices * 100
    
    return {
        'macd_line': macd_line_norm,
        'macd_signal': macd_signal_norm,
        'macd_hist': macd_hist_norm
    }

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range (normalized to price)."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    
    return atr / close

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (trend strength, normalized 0-1)."""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period, min_periods=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / (atr + 1e-8))
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / (atr + 1e-8))
    
    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di + 1e-8
    
    adx = 100 * di_diff / di_sum
    
    return adx.clip(0, 100) / 100

def calculate_bb_position(close, period=20, num_std=2):
    """Calculate Bollinger Band position (0-1)."""
    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    
    bb_high = sma + (std * num_std)
    bb_low = sma - (std * num_std)
    
    bb_pos = (close - bb_low) / (bb_high - bb_low + 1e-8)
    
    return bb_pos.clip(0, 1)

def calculate_obv(close, volume):
    """Calculate On-Balance Volume (normalized)."""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ma = obv.rolling(window=20, min_periods=20).mean()
    obv_std = obv.rolling(window=20, min_periods=20).std()
    
    obv_norm = (obv - obv_ma) / (obv_std + 1e-8)
    obv_ratio = obv / (obv_ma + 1e-8)
    
    return obv_ratio.clip(-5, 5) / 5

# ----------------------------
# 4. FEATURE ENGINEERING
# ----------------------------

def compute_advanced_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 20 features: 8 base + 12 advanced."""
    print("🔧 Computing 20 advanced features...")
    
    df = raw_df.copy()
    df = df.sort_values('TradeDate').reset_index(drop=True)
    
    if len(df) < 250:
        raise ValueError(f"Need at least 250 days, got {len(df)}")
    
    # ===== ORIGINAL 8 FEATURES =====
    
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    
    df['vol_10d'] = df['ret_1d'].rolling(10, min_periods=5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20, min_periods=10).std()
    
    for win in [20, 50, 200]:
        sma = df['close'].rolling(win, min_periods=win//2).mean()
        df[f'dist_sma_{win}'] = (df['close'] - sma) / (sma + 1e-8)
    
    vol_ma = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z_20'] = (df['volume'] - vol_ma) / (vol_std + 1e-8)
    
    # ===== NEW 12 ADVANCED FEATURES =====
    
    print("   Computing momentum indicators...")
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    
    macd_dict = calculate_macd(df['close'])
    df['macd_line'] = macd_dict['macd_line']
    df['macd_signal'] = macd_dict['macd_signal']
    df['macd_hist'] = macd_dict['macd_hist']
    
    print("   Computing volatility indicators...")
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    df['vol_ratio'] = df['vol_20d'] / (df['vol_10d'] + 1e-8)
    df['vol_trend'] = df['vol_20d'].pct_change(5)
    
    print("   Computing trend indicators...")
    df['adx_14'] = calculate_adx(df['high'], df['low'], df['close'], 14)
    df['bb_position'] = calculate_bb_position(df['close'], 20)
    
    print("   Computing price action indicators...")
    df['high_20'] = df['close'].rolling(20).max()
    df['low_20'] = df['close'].rolling(20).min()
    df['range_norm'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
    
    print("   Computing return patterns...")
    df['ret_skew'] = df['ret_1d'].rolling(20, min_periods=10).skew()
    
    print("   Computing volume-price indicators...")
    df['obv_ratio'] = calculate_obv(df['close'], df['volume'])
    
    # ===== FORWARD LABELS =====
    
    df['ret_5d_fwd'] = df['close'].shift(-5) / df['close'] - 1.0
    df['ret_60d_fwd'] = df['close'].shift(-60) / df['close'] - 1.0
    
    result = df[ALL_FEATURES + ['TradeDate', 'close', 'high', 'low', 'volume', 
                                'ret_5d_fwd', 'ret_60d_fwd']].copy()
    result = result.fillna(method='bfill').fillna(method='ffill').dropna()
    result = result.reset_index(drop=True)
    
    print(f"✅ {len(result)} days with 20 features")
    
    return result

# ----------------------------
# 5. OBSERVATION FUNCTION
# ----------------------------

def get_observation(df: pd.DataFrame, i: int, position: float, equity: float, 
                   feat_mean: np.ndarray, feat_std: np.ndarray):
    """Get normalized observation for model (22 dimensions)."""
    
    # Extract raw features (20)
    feat_row = df.iloc[i][ALL_FEATURES].values.astype(np.float32)
    
    # Normalize
    normalized_features = (feat_row - feat_mean) / feat_std
    
    # Add state (2)
    state = np.array([position, equity], dtype=np.float32)
    
    # Concatenate (22 total)
    obs = np.concatenate([normalized_features, state]).astype(np.float32)
    
    assert obs.shape == (22,), f"Obs shape {obs.shape} != (22,)"
    return obs

# ----------------------------
# 6. BACKTEST FUNCTION
# ----------------------------

def run_backtest(model_path: str, df: pd.DataFrame, symbol: str = "SPY"):
    """
    Run backtest on retrained model.
    
    Returns: DataFrame with daily predictions and all 20 features
    """
    
    print(f"🔮 Loading retrained model: {model_path}")
    
    # Load normalization stats for 20 features (from retrained model)
    feat_mean = np.load("feat_mean_retrained.npy")
    feat_std = np.load("feat_std_retrained.npy")
    
    assert feat_mean.shape[0] == 20, f"Mean shape {feat_mean.shape[0]} != 20"
    assert feat_std.shape[0] == 20, f"Std shape {feat_std.shape[0]} != 20"
    
    print(f"✅ Loaded normalization stats (20 features)")
    
    # Load model
    try:
        model = DQN.load(model_path)
        print(f"✅ Loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Initialize trading state
    position = 0.0
    equity = 1.0
    last_price = df['close'].iloc[0]
    max_equity = 1.0
    recent_pnl = []
    
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
            
            # Get observation (22 dimensions)
            obs = get_observation(df, i, position, equity, feat_mean, feat_std)
            obs = obs.reshape(1, -1)  # (1, 22) for model
            
            # Get model prediction
            action, _ = model.predict(obs, deterministic=True)
            action = int(action[0])
            
            # Map action to position
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
            
            recent_pnl.append(pnl)
            if len(recent_pnl) > 20:
                recent_pnl.pop(0)
            
            if equity > max_equity:
                max_equity = equity
            
            # Store prediction with ALL 20 features
            pred_row = {
                'date': trade_date,
                'symbol': symbol,
                'price': price,
                'high': float(df['high'].iloc[i]) if 'high' in df.columns else price,
                'low': float(df['low'].iloc[i]) if 'low' in df.columns else price,
                'volume': float(df['volume'].iloc[i]) if 'volume' in df.columns else 0,
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
            
            # Add all BASE FEATURES (8)
            for feat in BASE_FEATURES:
                pred_row[feat] = float(df[feat].iloc[i])
            
            # Add all ADVANCED FEATURES (12)
            for feat in ADVANCED_FEATURES:
                pred_row[feat] = float(df[feat].iloc[i])
            
            predictions.append(pred_row)
            
            # Progress update (quarterly)
            if (i + 1) % 63 == 0:  # ~quarterly for 252 trading days/year
                print(f"  Day {i+1:4d}: Equity=${equity:.4f} | Signal={['FLAT', 'LONG', 'SHORT'][action]} | Price=${price:.2f}")
        
        except Exception as e:
            print(f"⚠️  Day {i}: {str(e)[:50]}")
            continue
    
    # Create DataFrame
    df_preds = pd.DataFrame(predictions)
    
    print(f"\n✅ Backtest complete: {len(df_preds)} trading days")
    
    return df_preds

# ----------------------------
# 7. ANALYSIS FUNCTION
# ----------------------------

def analyze_backtest(df_preds: pd.DataFrame) -> dict:
    """Analyze backtest results comprehensively."""
    
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
    
    # Volatility
    daily_vol = daily_returns.std() * np.sqrt(252) * 100
    
    # Profit factor
    total_wins = df_preds[df_preds['daily_pnl'] > 0]['daily_pnl'].sum()
    total_losses = abs(df_preds[df_preds['daily_pnl'] < 0]['daily_pnl'].sum())
    profit_factor = total_wins / (total_losses + 1e-8)
    
    # By signal
    by_signal = df_preds.groupby('signal').agg({
        'daily_pnl': ['count', 'sum', 'mean'],
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
        'daily_volatility': daily_vol,
        'profit_factor': profit_factor,
        'daily_wins': wins,
        'daily_losses': losses
    }
    
    # Print
    print(f"\n💰 RETURNS:")
    print(f"   Final Equity:    ${final_equity:.4f}")
    print(f"   Total Return:    {total_return:+.2f}%")
    print(f"   Sharpe Ratio:    {sharpe:.2f}")
    print(f"   Annual Volatility: {daily_vol:.2f}%")
    print(f"   Profit Factor:   {profit_factor:.2f}")
    
    print(f"\n📈 TRADES:")
    print(f"   Total Trades:    {total_trades}")
    print(f"   Daily Wins:      {wins}")
    print(f"   Daily Losses:    {losses}")
    print(f"   Win Rate:        {win_rate:.1f}%")
    
    print(f"\n📉 RISK:")
    print(f"   Max Drawdown:    {max_drawdown:.2f}%")
    
    print(f"\n📊 BY SIGNAL:")
    print(by_signal)
    
    # Buy and hold comparison
    first_price = df_preds['price'].iloc[0]
    last_price = df_preds['price'].iloc[-1]
    buy_hold_return = (last_price - first_price) / first_price * 100
    
    print(f"\n📈 COMPARISON TO BUY & HOLD:")
    print(f"   Buy & Hold Return: {buy_hold_return:+.2f}%")
    print(f"   Model Return:      {total_return:+.2f}%")
    print(f"   Outperformance:    {total_return - buy_hold_return:+.2f}%")
    
    if total_return > buy_hold_return:
        print(f"   ✅ MODEL OUTPERFORMED BUY & HOLD")
    else:
        print(f"   ⚠️  Model underperformed buy & hold")
    
    print(f"\n{'='*60}")
    
    return results

# ----------------------------
# 8. SQL SAVE FUNCTION
# ----------------------------

def save_to_sql(conn_str: str, df_preds: pd.DataFrame, run_id: str):
    """Save backtest results to SQL."""
    
    print(f"\n💾 Saving to SQL...")
    
    engine = sa.create_engine(conn_str)
    
    # Add metadata
    df_preds['run_id'] = run_id
    df_preds['run_date'] = datetime.now()
    
    # Ensure date is DATE type
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
    
    running_max = df_preds['equity'].expanding().max()
    drawdown = (df_preds['equity'] - running_max) / running_max * 100
    max_dd = drawdown.min()
    
    trades = (df_preds['position_change'] > 0).sum()
    wins = (df_preds['daily_pnl'] > 0).sum()
    total_wins = df_preds[df_preds['daily_pnl'] > 0]['daily_pnl'].sum()
    total_losses = abs(df_preds[df_preds['daily_pnl'] < 0]['daily_pnl'].sum())
    profit_factor = total_wins / (total_losses + 1e-8)
    
    daily_vol = daily_returns.std() * np.sqrt(252) * 100
    
    summary = pd.DataFrame([{
        'run_id': run_id,
        'symbol': 'SPY',
        'period_start': df_preds['date'].min(),
        'period_end': df_preds['date'].max(),
        'days': len(df_preds),
        'total_return_pct': total_return,
        'final_equity': final_equity,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'total_trades': int(trades),
        'winning_days': int(wins),
        'losing_days': int(len(df_preds) - wins),
        'win_rate': (wins / len(df_preds) * 100) if len(df_preds) > 0 else 0,
        'profit_factor': profit_factor,
        'daily_volatility': daily_vol,
        'model_version': 'retrained_2015_2024',
        'feature_count': 20,
        'dqn_trades': int(trades),
        'random_trades': 0,
        'run_date': datetime.now()
    }])
    
    summary.to_sql('dqn_prediction_summary', engine, if_exists='append', index=False)
    
    print(f"\n✅ SAVED TO SQL")
    print(f"   Run ID: {run_id}")
    print(f"   Predictions: {len(df_preds):,} rows")
    print(f"   Summary: 1 row")

# ----------------------------
# 9. MAIN ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    
    SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    
    try:
        print(f"\n{'='*60}")
        print(f"🚀 DUAL-HORIZON BACKTEST - RETRAINED MODEL (2015-2024)")
        print(f"{'='*60}\n")
        
        # ===== STEP 1: Load Data =====
        print(f"STEP 1: Load backtest data (2024 - Unseen by model)")
        print(f"-" * 60)
        df_raw = load_data(SQL_CONN, SYMBOL, BACKTEST_START, BACKTEST_END)
        
        # ===== STEP 2: Compute Advanced Features =====
        print(f"\nSTEP 2: Compute 20 advanced features")
        print(f"-" * 60)
        df = compute_advanced_features(df_raw)
        
        if len(df) == 0:
            raise ValueError("No valid data after feature computation!")
        
        # ===== STEP 3: Run Backtest =====
        print(f"\nSTEP 3: Run backtest with RETRAINED model")
        print(f"-" * 60)
        df_preds = run_backtest(MODEL_NAME, df, SYMBOL)
        
        # ===== STEP 4: Analyze =====
        print(f"\nSTEP 4: Analyze results")
        print(f"-" * 60)
        results = analyze_backtest(df_preds)
        
        # ===== STEP 5: Save to SQL =====
        print(f"\nSTEP 5: Save to SQL")
        print(f"-" * 60)
        run_id = f"backtest_{SYMBOL}_retrained_2015_2024_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_to_sql(SQL_CONN, df_preds, run_id)
        
        # ===== FINAL SUMMARY =====
        print(f"\n{'='*60}")
        print(f"✅ BACKTEST COMPLETE - RETRAINED MODEL")
        print(f"{'='*60}")
        print(f"\n📊 KEY METRICS:")
        print(f"   Return:        {results['total_return_pct']:+.2f}%")
        print(f"   Final Equity:  ${results['final_equity']:.4f}")
        print(f"   Sharpe Ratio:  {results['sharpe_ratio']:.2f}")
        print(f"   Total Trades:  {results['total_trades']}")
        print(f"   Win Rate:      {results['win_rate']:.1f}%")
        print(f"   Max Drawdown:  {results['max_drawdown']:.2f}%")
        print(f"   Volatility:    {results['daily_volatility']:.2f}%")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"\n📋 Run ID: {run_id}")
        print(f"\n📈 Database Queries:")
        print(f"   SELECT * FROM dqn_daily_predictions")
        print(f"   WHERE run_id = '{run_id}'")
        print(f"\n   SELECT * FROM dqn_prediction_summary")
        print(f"   WHERE run_id = '{run_id}'")
        print(f"\n📊 Compare to Old Model:")
        print(f"   SELECT model_version, total_return_pct, sharpe_ratio")
        print(f"   FROM dqn_prediction_summary")
        print(f"   WHERE symbol = 'SPY'")
        print(f"   ORDER BY total_return_pct DESC")
        print(f"\n{'='*60}\n")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
