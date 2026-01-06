"""
🚀 DUAL-HORIZON DQN BACKTEST - LONG ONLY MODEL (COMPLETE - FIXED)
Full backtest script with confidence-based position sizing
All 20 features included
Complete analysis and SQL integration
✅ FIXED: Removed 'year' column that doesn't exist in SQL table
"""

import pandas as pd
import sqlalchemy as sa
import numpy as np
from stable_baselines3 import DQN
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🎯 DUAL-HORIZON DQN BACKTEST - LONG ONLY MODEL (FIXED)\n")

# ============================================================
# 1. CONFIGURATION
# ============================================================

SYMBOL = "SPY"

# All 20 Features
BASE_FEATURES = [
    'ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
    'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20'
]

ADVANCED_FEATURES = [
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'atr_14', 'vol_ratio', 'vol_trend',
    'adx_14', 'bb_position',
    'range_norm', 'ret_skew', 'obv_ratio'
]

ALL_FEATURES = BASE_FEATURES + ADVANCED_FEATURES

# Backtest period (unseen data)
BACKTEST_START = "2024-01-01"
BACKTEST_END = "2025-01-01"
MODEL_NAME = "dqn_spy_longonly_2015_2024"

# SQL Configuration
SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

# ============================================================
# 2. DATA LOADING
# ============================================================

def load_data(conn_str: str, symbol: str, start_date: str, end_date: str):
    """Load price data from SQL Server."""
    print(f"📊 Loading {symbol} ({start_date} to {end_date})...")
    
    engine = sa.create_engine(conn_str)
    
    query = f"""
    SELECT TradeDate, [close], [high], [low], volume
    FROM AI_ETF_Prices
    WHERE symbol = '{symbol}' 
        AND TradeDate >= '{start_date}' 
        AND TradeDate <= '{end_date}'
    ORDER BY TradeDate
    """
    
    try:
        df = pd.read_sql(query, engine)
        if len(df) == 0:
            raise ValueError(f"No data found for {symbol}")
        
        df['TradeDate'] = pd.to_datetime(df['TradeDate'])
        print(f"✅ Loaded {len(df)} days\n")
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

# ============================================================
# 3. TECHNICAL INDICATOR FUNCTIONS
# ============================================================

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
    
    obv_ratio = obv / (obv_ma + 1e-8)
    
    return obv_ratio.clip(-5, 5) / 5

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def compute_advanced_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 20 advanced features (8 base + 12 advanced)."""
    print("🔧 Computing 20 advanced features...")
    
    df = raw_df.copy()
    df = df.sort_values('TradeDate').reset_index(drop=True)
    
    if len(df) < 250:
        raise ValueError(f"Need at least 250 days, got {len(df)}")
    
    # ===== BASE FEATURES (8) =====
    
    print("   Computing return metrics...")
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    
    print("   Computing volatility metrics...")
    df['vol_10d'] = df['ret_1d'].rolling(10, min_periods=5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20, min_periods=10).std()
    
    print("   Computing moving average distances...")
    for win in [20, 50, 200]:
        sma = df['close'].rolling(win, min_periods=win//2).mean()
        df[f'dist_sma_{win}'] = (df['close'] - sma) / (sma + 1e-8)
    
    print("   Computing volume Z-score...")
    vol_ma = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z_20'] = (df['volume'] - vol_ma) / (vol_std + 1e-8)
    
    # ===== ADVANCED FEATURES (12) =====
    
    print("   Computing momentum indicators (RSI, MACD)...")
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    
    macd_dict = calculate_macd(df['close'])
    df['macd_line'] = macd_dict['macd_line']
    df['macd_signal'] = macd_dict['macd_signal']
    df['macd_hist'] = macd_dict['macd_hist']
    
    print("   Computing volatility indicators (ATR, Vol Ratio, Vol Trend)...")
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    df['vol_ratio'] = df['vol_20d'] / (df['vol_10d'] + 1e-8)
    df['vol_trend'] = df['vol_20d'].pct_change(5)
    
    print("   Computing trend indicators (ADX, Bollinger Bands)...")
    df['adx_14'] = calculate_adx(df['high'], df['low'], df['close'], 14)
    df['bb_position'] = calculate_bb_position(df['close'], 20)
    
    print("   Computing price action indicators...")
    df['high_20'] = df['close'].rolling(20).max()
    df['low_20'] = df['close'].rolling(20).min()
    df['range_norm'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
    
    print("   Computing return patterns (Skewness)...")
    df['ret_skew'] = df['ret_1d'].rolling(20, min_periods=10).skew()
    
    print("   Computing volume-price indicators (OBV)...")
    df['obv_ratio'] = calculate_obv(df['close'], df['volume'])
    
    # ===== FORWARD RETURNS (for analysis) =====
    df['ret_5d_fwd'] = df['close'].shift(-5) / df['close'] - 1.0
    df['ret_60d_fwd'] = df['close'].shift(-60) / df['close'] - 1.0
    
    # ===== CREATE FINAL DATASET =====
    result = df[ALL_FEATURES + ['TradeDate', 'close', 'high', 'low', 'volume', 
                                'ret_5d_fwd', 'ret_60d_fwd']].copy()
    
    result = result.fillna(method='bfill').fillna(method='ffill').dropna()
    result = result.reset_index(drop=True)
    
    print(f"✅ {len(result)} days with 20 features\n")
    
    return result

# ============================================================
# 5. OBSERVATION FUNCTION
# ============================================================

def get_observation(df: pd.DataFrame, i: int, position: float, 
                   feat_mean: np.ndarray, feat_std: np.ndarray):
    """
    Get normalized observation for model.
    Returns 21-dimensional vector: 20 features + 1 position
    """
    
    # Extract raw features (20)
    feat_row = df.iloc[i][ALL_FEATURES].values.astype(np.float32)
    
    # Normalize features
    normalized_features = (feat_row - feat_mean) / feat_std
    
    # Add position state (1 dimension)
    state = np.array([position], dtype=np.float32)
    
    # Concatenate to get 21 dimensions total
    obs = np.concatenate([normalized_features, state]).astype(np.float32)
    
    assert obs.shape == (21,), f"Observation shape {obs.shape} != (21,)"
    return obs

# ============================================================
# 6. MAIN BACKTEST FUNCTION
# ============================================================

def run_backtest(model_path: str, df: pd.DataFrame, symbol: str = "SPY"):
    """
    Run backtest on long-only model with confidence-based position sizing.
    
    Returns:
        DataFrame with daily predictions and all 20 features
    """
    
    print(f"🔮 Loading long-only model: {model_path}")
    
    # ===== LOAD NORMALIZATION STATS =====
    try:
        feat_mean = np.load("feat_mean_longonly.npy")
        feat_std = np.load("feat_std_longonly.npy")
        
        assert feat_mean.shape[0] == 20, f"Mean shape {feat_mean.shape[0]} != 20"
        assert feat_std.shape[0] == 20, f"Std shape {feat_std.shape[0]} != 20"
        
        print(f"✅ Loaded normalization stats (20 features)")
    except Exception as e:
        print(f"❌ Error loading normalization stats: {e}")
        raise
    
    # ===== LOAD MODEL =====
    try:
        model = DQN.load(model_path)
        print(f"✅ Loaded model: {model_path}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # ===== INITIALIZE TRADING STATE =====
    position = 0.0              # Current position (0 = FLAT, 1.0 = FULL LONG)
    equity = 1.0                # Starting equity
    last_price = df['close'].iloc[0]
    max_equity = 1.0
    recent_pnl = []
    
    predictions = []
    
    print(f"🔮 Running {len(df)} day backtest...")
    print(f"   Strategy: LONG ONLY (no SHORT)")
    print(f"   Position Sizing: Confidence-based (0-1)")
    print(f"   Observation Space: 21 dimensions (20 features + position)\n")
    
    # ===== MAIN BACKTEST LOOP =====
    for i in range(len(df)):
        try:
            # Get data for this day
            trade_date = pd.to_datetime(df['TradeDate'].iloc[i]).date()
            price = float(df['close'].iloc[i])
            high = float(df['high'].iloc[i]) if 'high' in df.columns else price
            low = float(df['low'].iloc[i]) if 'low' in df.columns else price
            volume = float(df['volume'].iloc[i]) if 'volume' in df.columns else 0
            ret_5d_actual = float(df['ret_5d_fwd'].iloc[i])
            ret_60d_actual = float(df['ret_60d_fwd'].iloc[i])
            
            # ===== GET MODEL OBSERVATION =====
            obs = get_observation(df, i, position, feat_mean, feat_std)
            obs = obs.reshape(1, -1)  # (1, 21) for model
            
            # ===== GET MODEL PREDICTION =====
            action, _ = model.predict(obs, deterministic=True)
            action = int(action[0])  # 0=FLAT, 1=LONG
            
            # ===== CONFIDENCE-BASED POSITION SIZING =====
            # Get Q-values from model to calculate confidence
            try:
                q_values = model.q_net(model.policy.obs_to_tensor(obs)[0])
                q_values_np = q_values.detach().cpu().numpy()[0]
                
                # Calculate confidence as difference between best and second-best actions
                sorted_q = np.sort(q_values_np)
                confidence = (sorted_q[-1] - sorted_q[-2]) / (abs(sorted_q[-1]) + 1e-8)
                confidence = np.clip(confidence, 0, 1)  # Normalize to 0-1
            except Exception:
                # Fallback if Q-values not available
                confidence = 0.5
            
            # ===== MAP ACTION TO POSITION WITH CONFIDENCE SCALING =====
            if action == 0:  # FLAT
                target_pos = 0.0
                signal = 'FLAT'
            else:  # LONG (action == 1)
                target_pos = 1.0 * confidence  # Scale by confidence (0 to 1)
                signal = 'LONG'
            
            # ===== TRANSACTION COST =====
            pos_change = abs(target_pos - position)
            cost = pos_change * equity * 0.00005  # 0.5 basis points
            
            # ===== CALCULATE DAILY P&L =====
            if i > 0:
                ret = (price - last_price) / last_price
            else:
                ret = 0.0
            
            # P&L = position * return * equity
            pnl = position * ret * equity
            
            # ===== UPDATE TRADING STATE =====
            equity += pnl - cost
            position = target_pos
            last_price = price
            
            # Track recent P&L for statistics
            recent_pnl.append(pnl)
            if len(recent_pnl) > 20:
                recent_pnl.pop(0)
            
            # Update maximum equity (for drawdown calculation)
            if equity > max_equity:
                max_equity = equity
            
            # ===== CREATE PREDICTION ROW WITH ALL DATA =====
            pred_row = {
                # Metadata
                'date': trade_date,
                'symbol': symbol,
                'price': price,
                'high': high,
                'low': low,
                'volume': volume,
                
                # Model prediction
                'action': action,
                'signal': signal,
                
                # Position management
                'position': position,
                'position_confidence': confidence,
                'position_change': pos_change,
                
                # P&L metrics
                'daily_pnl': pnl,
                'trade_cost': cost,
                
                # Portfolio state
                'equity': equity,
                'equity_pct': equity * 100,
                
                # Returns
                'daily_return': ret * 100,
                'ret_5d_forecast': ret_5d_actual * 100,
                'ret_60d_forecast': ret_60d_actual * 100,
            }
            
            # Add BASE FEATURES (8)
            for feat in BASE_FEATURES:
                pred_row[feat] = float(df[feat].iloc[i])
            
            # Add ADVANCED FEATURES (12)
            for feat in ADVANCED_FEATURES:
                pred_row[feat] = float(df[feat].iloc[i])
            
            predictions.append(pred_row)
            
            # ===== PROGRESS UPDATE (Quarterly) =====
            if (i + 1) % 63 == 0:  # ~quarterly for 252 trading days/year
                print(f"  Day {i+1:4d}: Equity=${equity:.4f} | Signal={signal:4s} | Pos={position:.2f} | Conf={confidence:.2f} | Price=${price:.2f}")
        
        except Exception as e:
            print(f"⚠️  Day {i}: Error - {str(e)[:50]}")
            continue
    
    # ===== CREATE RESULTS DATAFRAME =====
    df_preds = pd.DataFrame(predictions)
    
    print(f"\n✅ Backtest complete: {len(df_preds)} trading days processed\n")
    
    return df_preds

# ============================================================
# 7. ANALYSIS FUNCTION (FIXED - NO YEAR COLUMN)
# ============================================================

def analyze_backtest(df_preds: pd.DataFrame) -> dict:
    """Comprehensive backtest analysis (FIXED: no year column)."""
    
    print(f"{'='*70}")
    print(f"📊 BACKTEST ANALYSIS - LONG ONLY MODEL")
    print(f"{'='*70}\n")
    
    # ===== BASIC PERFORMANCE METRICS =====
    final_equity = df_preds['equity'].iloc[-1]
    total_return = (final_equity - 1.0) * 100
    
    # Sharpe ratio
    daily_returns = df_preds['daily_pnl'] / df_preds['equity'].shift(1)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    
    # Win/Loss statistics
    wins = (df_preds['daily_pnl'] > 0).sum()
    losses = (df_preds['daily_pnl'] < 0).sum()
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    # Drawdown
    running_max = df_preds['equity'].expanding().max()
    drawdown = (df_preds['equity'] - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Volatility
    daily_vol = daily_returns.std() * np.sqrt(252) * 100
    
    # Profit factor
    total_wins = df_preds[df_preds['daily_pnl'] > 0]['daily_pnl'].sum()
    total_losses = abs(df_preds[df_preds['daily_pnl'] < 0]['daily_pnl'].sum())
    profit_factor = total_wins / (total_losses + 1e-8)
    
    # Trade statistics
    total_trades = (df_preds['position_change'] > 0.01).sum()
    
    # Store results
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
    
    # ===== PRINT DETAILED ANALYSIS =====
    
    print(f"💰 RETURNS:")
    print(f"   Final Equity:       ${final_equity:.4f}")
    print(f"   Total Return:       {total_return:+.2f}%")
    print(f"   Annualized Sharpe:  {sharpe:.2f}")
    print(f"   Profit Factor:      {profit_factor:.2f}x")
    
    print(f"\n📈 TRADES & WIN RATE:")
    print(f"   Total Trades:       {total_trades}")
    print(f"   Daily Wins:         {wins}")
    print(f"   Daily Losses:       {losses}")
    print(f"   Win Rate:           {win_rate:.1f}%")
    print(f"   Avg Winning Day:    ${df_preds[df_preds['daily_pnl'] > 0]['daily_pnl'].mean():.6f}")
    print(f"   Avg Losing Day:     ${df_preds[df_preds['daily_pnl'] < 0]['daily_pnl'].mean():.6f}")
    
    print(f"\n📉 RISK METRICS:")
    print(f"   Max Drawdown:       {max_drawdown:.2f}%")
    print(f"   Daily Volatility:   {daily_vol:.2f}%")
    print(f"   Best Day P&L:       ${df_preds['daily_pnl'].max():.6f}")
    print(f"   Worst Day P&L:      ${df_preds['daily_pnl'].min():.6f}")
    
    # ===== SIGNAL ANALYSIS =====
    print(f"\n📊 SIGNAL DISTRIBUTION:")
    
    flat_days = (df_preds['signal'] == 'FLAT').sum()
    long_days = (df_preds['signal'] == 'LONG').sum()
    
    print(f"   FLAT days: {flat_days:4d} ({flat_days/len(df_preds)*100:5.1f}%)")
    print(f"   LONG days: {long_days:4d} ({long_days/len(df_preds)*100:5.1f}%)")
    
    # Performance by signal
    if flat_days > 0:
        flat_data = df_preds[df_preds['signal'] == 'FLAT']
        flat_pnl = flat_data['daily_pnl'].sum()
        flat_avg = flat_data['daily_pnl'].mean()
        flat_wins = (flat_data['daily_pnl'] > 0).sum()
        print(f"\n   FLAT Performance:")
        print(f"      Total P&L: ${flat_pnl:.4f}")
        print(f"      Avg P&L:   ${flat_avg:.6f}")
        print(f"      Win Rate:  {flat_wins/flat_days*100:.1f}%")
    
    if long_days > 0:
        long_data = df_preds[df_preds['signal'] == 'LONG']
        long_pnl = long_data['daily_pnl'].sum()
        long_avg = long_data['daily_pnl'].mean()
        long_wins = (long_data['daily_pnl'] > 0).sum()
        print(f"\n   LONG Performance:")
        print(f"      Total P&L: ${long_pnl:.4f}")
        print(f"      Avg P&L:   ${long_avg:.6f}")
        print(f"      Win Rate:  {long_wins/long_days*100:.1f}%")
    
    # ===== POSITION SIZING ANALYSIS =====
    print(f"\n📏 POSITION SIZING:")
    
    avg_position = df_preds['position'].mean()
    avg_confidence = df_preds['position_confidence'].mean()
    max_position = df_preds['position'].max()
    min_position = df_preds['position'].min()
    
    print(f"   Avg Position:       {avg_position:.2f}")
    print(f"   Max Position:       {max_position:.2f}")
    print(f"   Min Position:       {min_position:.2f}")
    print(f"   Avg Confidence:     {avg_confidence:.2f}")
    
    # ===== COMPARISON TO BUY & HOLD =====
    print(f"\n📈 COMPARISON TO BUY & HOLD:")
    
    first_price = df_preds['price'].iloc[0]
    last_price = df_preds['price'].iloc[-1]
    buy_hold_return = (last_price - first_price) / first_price * 100
    
    print(f"   Buy & Hold Return:  {buy_hold_return:+.2f}%")
    print(f"   Model Return:       {total_return:+.2f}%")
    print(f"   Outperformance:     {total_return - buy_hold_return:+.2f}%")
    
    if total_return > buy_hold_return:
        print(f"   ✅ MODEL OUTPERFORMED BUY & HOLD")
    elif abs(total_return - buy_hold_return) < 5:
        print(f"   ⚠️  MODEL UNDERPERFORMED BY {buy_hold_return - total_return:.2f}%")
    else:
        print(f"   ❌ MODEL SIGNIFICANTLY UNDERPERFORMED")
    
    # ===== BACKTEST PERIOD SUMMARY (NO YEAR COLUMN) =====
    print(f"\n📊 BACKTEST PERIOD:")
    
    date_min = pd.to_datetime(df_preds['date']).min()
    date_max = pd.to_datetime(df_preds['date']).max()
    
    print(f"   Start Date:         {date_min.date()}")
    print(f"   End Date:           {date_max.date()}")
    print(f"   Total Days:         {len(df_preds)}")
    
    print(f"\n{'='*70}\n")
    
    return results

# ============================================================
# 8. SQL SAVE FUNCTION (FIXED - NO YEAR COLUMN)
# ============================================================

def save_to_sql(conn_str: str, df_preds: pd.DataFrame, run_id: str):
    """Save backtest results and summary to SQL Server (FIXED)."""
    
    print(f"💾 Saving results to SQL Server...")
    
    engine = sa.create_engine(conn_str)
    
    # ===== ADD METADATA TO PREDICTIONS =====
    df_preds['run_id'] = run_id
    df_preds['run_date'] = datetime.now()
    
    # Ensure date is DATE type
    df_preds['date'] = pd.to_datetime(df_preds['date']).dt.date
    
    # ===== DO NOT ADD YEAR COLUMN =====
    # ❌ REMOVED: df_preds['year'] = pd.to_datetime(df_preds['date']).dt.year
    
    # ===== INSERT PREDICTIONS IN CHUNKS =====
    chunk_size = 500
    total_chunks = (len(df_preds) + chunk_size - 1) // chunk_size
    
    print(f"   Inserting {len(df_preds):,} daily predictions in {total_chunks} chunks...")
    
    for i in range(0, len(df_preds), chunk_size):
        chunk = df_preds.iloc[i:i+chunk_size].copy()
        chunk.to_sql('dqn_daily_predictions', engine, if_exists='append', index=False)
        chunk_num = (i // chunk_size) + 1
        print(f"   ✅ Chunk {chunk_num}/{total_chunks}")
    
    # ===== CREATE AND INSERT SUMMARY =====
    print(f"\n   Creating summary statistics...")
    
    final_equity = df_preds['equity'].iloc[-1]
    total_return = (final_equity - 1.0) * 100
    
    daily_returns = df_preds['daily_pnl'] / df_preds['equity'].shift(1)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    
    running_max = df_preds['equity'].expanding().max()
    drawdown = (df_preds['equity'] - running_max) / running_max * 100
    max_dd = drawdown.min()
    
    trades = (df_preds['position_change'] > 0.01).sum()
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
        'model_version': 'longonly_2015_2024',
        'feature_count': 20,
        'dqn_trades': int(trades),
        'random_trades': 0,
        'run_date': datetime.now()
    }])
    
    summary.to_sql('dqn_prediction_summary', engine, if_exists='append', index=False)
    
    print(f"   ✅ Summary inserted")
    print(f"\n✅ SAVED TO SQL")
    print(f"   Run ID: {run_id}")
    print(f"   Predictions: {len(df_preds):,} rows")
    print(f"   Summary: 1 row\n")

# ============================================================
# 9. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    try:
        print(f"\n{'='*70}")
        print(f"🚀 DUAL-HORIZON DQN BACKTEST - LONG ONLY MODEL (FIXED)")
        print(f"{'='*70}\n")
        
        # ===== STEP 1: LOAD DATA =====
        print(f"STEP 1: Load backtest data (2024 - Unseen by model)")
        print(f"-" * 70)
        df_raw = load_data(SQL_CONN, SYMBOL, BACKTEST_START, BACKTEST_END)
        
        # ===== STEP 2: COMPUTE FEATURES =====
        print(f"STEP 2: Compute 20 advanced features (8 base + 12 advanced)")
        print(f"-" * 70)
        df = compute_advanced_features(df_raw)
        
        if len(df) == 0:
            raise ValueError("No valid data after feature computation!")
        
        # ===== STEP 3: RUN BACKTEST =====
        print(f"STEP 3: Run backtest with LONG ONLY model")
        print(f"-" * 70)
        df_preds = run_backtest(MODEL_NAME, df, SYMBOL)
        
        # ===== STEP 4: ANALYZE RESULTS =====
        print(f"STEP 4: Analyze backtest results")
        print(f"-" * 70)
        results = analyze_backtest(df_preds)
        
        # ===== STEP 5: SAVE TO SQL =====
        print(f"STEP 5: Save results to SQL")
        print(f"-" * 70)
        run_id = f"backtest_{SYMBOL}_longonly_2015_2024_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_to_sql(SQL_CONN, df_preds, run_id)
        
        # ===== FINAL SUMMARY =====
        print(f"\n{'='*70}")
        print(f"✅ BACKTEST COMPLETE - LONG ONLY MODEL")
        print(f"{'='*70}\n")
        
        print(f"📊 KEY METRICS:")
        print(f"   Return:             {results['total_return_pct']:+.2f}%")
        print(f"   Final Equity:       ${results['final_equity']:.4f}")
        print(f"   Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
        print(f"   Total Trades:       {results['total_trades']}")
        print(f"   Win Rate:           {results['win_rate']:.1f}%")
        print(f"   Max Drawdown:       {results['max_drawdown']:.2f}%")
        print(f"   Volatility:         {results['daily_volatility']:.2f}%")
        print(f"   Profit Factor:      {results['profit_factor']:.2f}x")
        
        print(f"\n📋 DATABASE:")
        print(f"   Run ID: {run_id}")
        
        print(f"\n📈 QUERY TO VIEW RESULTS:")
        print(f"   SELECT * FROM dqn_daily_predictions")
        print(f"   WHERE run_id = '{run_id}'")
        
        print(f"\n📊 QUERY TO COMPARE MODELS:")
        print(f"   SELECT TOP 10 model_version, total_return_pct, sharpe_ratio")
        print(f"   FROM dqn_prediction_summary")
        print(f"   WHERE symbol = 'SPY'")
        print(f"   ORDER BY total_return_pct DESC")
        
        print(f"\n{'='*70}\n")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
