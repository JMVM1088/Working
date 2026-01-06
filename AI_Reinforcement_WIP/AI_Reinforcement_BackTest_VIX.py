"""
🚀 DUAL-HORIZON BACKTEST - DATE FIX
✅ Proper datetime handling ✅ SQL compatible
"""

import pandas as pd
import sqlalchemy as sa
import numpy as np
from stable_baselines3 import DQN
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🚀 DUAL-HORIZON BACKTEST - DATE FIXED")

BASE_FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
                'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']

VIX_FEATURES = ['vix_ret_1d', 'vix_vol_20d', 'vix_sma_20', 'vix_z_20', 
               'spy_vix_corr', 'vix_rsi']

ALL_FEATURES = BASE_FEATURES + VIX_FEATURES

def load_spy_vix_data(conn_str: str, start_date: str, end_date: str):
    engine = sa.create_engine(conn_str)
    
    spy_query = f"""
    SELECT TradeDate, [open], high, low, [close] as spy_close, volume as spy_volume
    FROM AI_ETF_Prices
    WHERE symbol = 'SPY' AND TradeDate >= '{start_date}' AND TradeDate <= '{end_date}'
    ORDER BY TradeDate
    """
    
    try:
        spy_df = pd.read_sql(spy_query, engine)
        if len(spy_df) == 0:
            raise ValueError("No SPY data")
        print(f"✅ SPY: {len(spy_df)} days")
    except Exception as e:
        print(f"❌ SPY query failed: {e}")
        raise
    
    try:
        vix_query = f"""
        SELECT TradeDate, [close] as vix_close
        FROM AI_ETF_Prices
        WHERE symbol = '^VIX' AND TradeDate >= '{start_date}' AND TradeDate <= '{end_date}'
        ORDER BY TradeDate
        """
        vix_df = pd.read_sql(vix_query, engine)
        if len(vix_df) > 0:
            print(f"✅ VIX (real): {len(vix_df)} days")
        else:
            raise ValueError("VIX empty")
    except:
        print("⚠️ Creating synthetic VIX...")
        spy_df_copy = spy_df.copy()
        spy_df_copy['ret'] = spy_df_copy['spy_close'].pct_change()
        spy_df_copy['vol_20d'] = spy_df_copy['ret'].rolling(20, min_periods=10).std()
        spy_df_copy['vix_close'] = (spy_df_copy['vol_20d'] * 200 + 15).fillna(18.0)
        spy_df_copy['vix_close'] = spy_df_copy['vix_close'].clip(10, 40)
        vix_df = spy_df_copy[['TradeDate', 'vix_close']].copy()
        print(f"✅ Synthetic VIX: {len(vix_df)} days")
    
    # Ensure TradeDate is datetime
    spy_df['TradeDate'] = pd.to_datetime(spy_df['TradeDate'])
    vix_df['TradeDate'] = pd.to_datetime(vix_df['TradeDate'])
    
    df = spy_df[['TradeDate', 'spy_close', 'spy_volume']].merge(vix_df, on='TradeDate', how='inner')
    print(f"✅ Merged: {len(df)} days")
    return df

def compute_features_with_vix(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df['TradeDate'] = pd.to_datetime(df['TradeDate'])
    df = df.sort_values('TradeDate').reset_index(drop=True)
    
    if len(df) < 50:
        print(f"⚠️ Only {len(df)} days")
        return pd.DataFrame()
    
    # SPY (8)
    df['ret_1d'] = df['spy_close'].pct_change()
    df['ret_5d'] = df['spy_close'].pct_change(5)
    df['vol_10d'] = df['ret_1d'].rolling(10, min_periods=5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20, min_periods=10).std()
    
    for win in [20, 50, 200]:
        sma = df['spy_close'].rolling(win, min_periods=win//2).mean()
        df[f'dist_sma_{win}'] = (df['spy_close'] - sma) / (sma + 1e-8)
    
    vol_ma = df['spy_volume'].rolling(20).mean()
    vol_std = df['spy_volume'].rolling(20).std()
    df['vol_z_20'] = (df['spy_volume'] - vol_ma) / (vol_std + 1e-8)
    
    # VIX (6)
    df['vix_ret_1d'] = df['vix_close'].pct_change()
    df['vix_vol_20d'] = df['vix_ret_1d'].rolling(20, min_periods=10).std()
    df['vix_sma_20'] = df['vix_close'].rolling(20).mean()
    df['vix_z_20'] = (df['vix_close'] - df['vix_sma_20']) / (df['vix_vol_20d'] + 1e-8)
    df['spy_vix_corr'] = df['ret_1d'].rolling(20).corr(df['vix_ret_1d'])
    
    # VIX RSI
    vix_delta = df['vix_close'].diff().fillna(0)
    vix_gain = vix_delta.clip(lower=0)
    vix_loss = -vix_delta.clip(upper=0)
    vix_avg_gain = vix_gain.rolling(14, min_periods=7).mean().fillna(0)
    vix_avg_loss = vix_loss.rolling(14, min_periods=7).mean().fillna(0)
    vix_rs = vix_avg_gain / (vix_avg_loss + 1e-8)
    df['vix_rsi'] = (100 - (100 / (1 + vix_rs))).fillna(50)
    
    result = df[ALL_FEATURES + ['TradeDate', 'spy_close', 'vix_close']].copy()
    result = result.fillna(method='bfill').fillna(method='ffill').dropna()
    
    print(f"✅ Features: {len(result)} clean days")
    return result.reset_index(drop=True)

def get_daily_observation(features_df: pd.DataFrame, i: int, position: float, equity: float, 
                         feat_mean: np.ndarray, feat_std: np.ndarray):
    feat_row = features_df.iloc[i][ALL_FEATURES].values.astype(np.float32)
    normalized = (feat_row - feat_mean) / feat_std
    state = np.array([position, equity], dtype=np.float32)
    obs = np.concatenate([normalized, state]).astype(np.float32)
    return obs

def run_backtest_dual_horizon(model_path: str, features_df: pd.DataFrame, symbol: str = "SPY"):
    if len(features_df) == 0:
        print("❌ Empty features!")
        return pd.DataFrame()
    
    feat_mean = np.load("feat_mean_vix.npy")
    feat_std = np.load("feat_std_vix.npy")
    
    model = DQN.load(model_path)
    
    position, equity, last_price = 0.0, 1.0, features_df['spy_close'].iloc[0]
    predictions = []
    
    print(f"🔮 Backtest: {len(features_df)} days")
    
    for i in range(len(features_df)):
        try:
            # CRITICAL: Proper datetime conversion
            trade_date = pd.to_datetime(features_df['TradeDate'].iloc[i])
            price = float(features_df['spy_close'].iloc[i])
            vix = float(features_df['vix_close'].iloc[i])
            
            obs = get_daily_observation(features_df, i, position, equity, feat_mean, feat_std)
            obs = obs.reshape(1, -1)
            
            if np.random.random() < 0.3:
                action = np.random.randint(0, 3)
                source = "RANDOM"
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action[0])
                source = "DQN"
            
            target_pos = [0.0, 1.0, -1.0][action]
            pos_change = abs(target_pos - position)
            cost = pos_change * equity * 0.00005
            
            ret = (price - last_price) / last_price if i > 0 else 0
            pnl = position * ret * equity
            equity += pnl - cost
            position = target_pos
            last_price = price
            
            # Build prediction row with proper types
            pred_row = {
                'date': trade_date,  # datetime.date
                'symbol': symbol,
                'price': float(price),
                'vix': float(vix),
                'action': int(action),
                'signal': ['FLAT','LONG','SHORT'][action],
                'position': float(position),
                'position_change': float(pos_change),
                'daily_pnl': float(pnl),
                'trade_cost': float(cost),
                'equity': float(equity),
                'equity_pct': float(equity * 100),
                'daily_return': float(ret * 100),
                'prediction_source': source,
                'run_id': None,  # Will be set later
                'run_date': None  # Will be set later
            }
            
            # Add all features with proper float conversion
            for feat in ALL_FEATURES:
                pred_row[feat] = float(features_df[feat].iloc[i])
            
            predictions.append(pred_row)
            
            if (i + 1) % 252 == 0:
                print(f"  Day {i+1}: Equity={equity:.3f}, VIX={vix:.1f}")
        
        except Exception as e:
            print(f"⚠️ Day {i}: {e}")
            continue
    
    df = pd.DataFrame(predictions)
    
    # Ensure date column is proper format
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    return df

def save_backtest(conn_str: str, df: pd.DataFrame, run_id: str):
    if len(df) == 0:
        print("❌ No predictions!")
        return
    
    engine = sa.create_engine(conn_str)
    
    # Set metadata
    df['run_id'] = run_id
    df['run_date'] = datetime.now()
    
    # Ensure date is DATE type (not datetime)
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Insert in chunks
    chunk_size = 500
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        chunk.to_sql('dqn_daily_predictions', engine, if_exists='append', index=False)
        print(f"💾 Chunk {(i//chunk_size)+1}/{total_chunks}")
    
    # Summary
    final_eq = df['equity'].iloc[-1]
    ret = (final_eq - 1) * 100
    sharpe = (df['daily_pnl'].mean() / df['daily_pnl'].std() * np.sqrt(252)) if df['daily_pnl'].std() > 0 else 0
    trades = (df['position_change'] > 0).sum()
    
    summary = pd.DataFrame([{
        'run_id': run_id,
        'symbol': 'SPY',
        'period_start': df['date'].min(),
        'period_end': df['date'].max(),
        'days': len(df),
        'total_return_pct': ret,
        'final_equity': final_eq,
        'sharpe_ratio': sharpe,
        'total_trades': int(trades),
        'dqn_trades': int((df[df['prediction_source']=='DQN']['position_change'] > 0).sum()),
        'random_trades': int((df[df['prediction_source']=='RANDOM']['position_change'] > 0).sum()),
        'run_date': datetime.now()
    }])
    
    summary.to_sql('dqn_prediction_summary', engine, if_exists='append', index=False)
    
    print(f"\n✅ {len(df)} predictions saved!")
    print(f"📈 Return: {ret:+.1f}% | Sharpe: {sharpe:.2f} | Trades: {int(trades)}")

if __name__ == "__main__":
    SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    
    try:
        print("📊 Loading data...")
        df = load_spy_vix_data(SQL_CONN, "2020-01-01", "2025-01-01")
        
        print("🔧 Computing features...")
        df = compute_features_with_vix(df)
        
        if len(df) > 0:
            print("\n🔮 Running backtest...")
            preds = run_backtest_dual_horizon("dqn_dual_horizon_vix.zip", df)
            
            if len(preds) > 0:
                run_id = f"dual_{datetime.now().strftime('%Y%m%d_%H%M')}"
                save_backtest(SQL_CONN, preds, run_id)
                print(f"🎉 COMPLETE! Run ID: {run_id}")
        
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
