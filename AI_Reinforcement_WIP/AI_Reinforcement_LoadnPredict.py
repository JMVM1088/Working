"""
PRODUCTION SIGNAL GENERATOR - FIXED TypeError
Shape (10,) + No hashing issues
"""

import pandas as pd
import sqlalchemy as sa
import numpy as np
from stable_baselines3 import DQN
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces

# ----------------------------
# 1. Production Feature Functions (unchanged)
# ----------------------------

def load_today_data(conn_str: str, symbol: str = "SPY", days_back: int = 250):
    engine = sa.create_engine(conn_str)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back*2)).strftime('%Y-%m-%d')
    
    query = f"""
    SELECT symbol, TradeDate, [open], high, low, [close], volume
    FROM AI_ETF_Prices
    WHERE symbol = '{symbol}' AND TradeDate >= '{start_date}' AND TradeDate <= '{end_date}'
    ORDER BY TradeDate
    """
    
    df = pd.read_sql(query, engine)
    df['date_clean'] = pd.to_datetime(df['TradeDate'])
    return df.sort_values('date_clean')

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    FIXED_FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
                     'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']
    
    df = df.copy().set_index('date_clean').sort_index()
    
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['vol_10d'] = df['ret_1d'].rolling(10, min_periods=5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20, min_periods=10).std()
    
    for win in [20, 50, 200]:
        sma = df['close'].rolling(win, min_periods=win//2).mean()
        df[f'dist_sma_{win}'] = (df['close'] - sma) / sma
    
    vol_ma = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z_20'] = (df['volume'] - vol_ma) / (vol_std + 1e-8)
    
    return df[FIXED_FEATURES].dropna().iloc[-1:].values.astype(np.float32)

# ----------------------------
# 2. FIXED: Production Observation (shape 10)
# ----------------------------

def get_today_observation(conn_str: str, symbol: str = "SPY", 
                         current_position: float = 0.0, current_equity: float = 1.0) -> np.ndarray:
    """
    Returns EXACT shape (10,) for model.predict()
    """
    # Load & compute features
    df = load_today_data(conn_str, symbol)
    raw_features = compute_features(df)  # Shape (1, 8)
    
    if len(raw_features) == 0:
        raise ValueError("No features computed")
    
    # Load training normalization
    feat_mean = np.load("feat_mean.npy")
    feat_std = np.load("feat_std.npy")
    
    # Normalize features (8,)
    normalized_features = (raw_features[0] - feat_mean) / feat_std
    
    # COMPLETE OBSERVATION (10,)
    obs = np.concatenate([
        normalized_features,                    # 8 features
        np.array([current_position, current_equity], dtype=np.float32)  # + position + equity
    ])
    
    print(f"✅ Obs shape: {obs.shape}")
    return obs

# ----------------------------
# 3. FIXED: Standalone Production Predictor
# ----------------------------

def get_trading_signal(conn_str: str, model_path: str, symbol: str = "SPY",
                      current_position: float = 0.0, current_equity: float = 1.0):
    """
    NO ENVIRONMENT NEEDED - Direct model.predict()
    """
    # Load model DIRECTLY
    print(f"📂 Loading {model_path}...")
    model = DQN.load(model_path)
    
    # Get observation
    obs = get_today_observation(conn_str, symbol, current_position, current_equity)
    
    # PREDICT - NO VECTOR ENV NEEDED
    action, _ = model.predict(obs[np.newaxis, :], deterministic=True)  # Add batch dim
    action = action[0]  # Remove batch dim
    
    signals = {0: "🟡 FLAT", 1: "🟢 LONG", 2: "🔴 SHORT"}
    
    print(f"\n🎯 {symbol} TRADING SIGNAL")
    print(f"   {signals[action]}")
    print(f"   Position: {current_position:+.1f} → Target: {['0.0', '+1.0', '-1.0'][action]}")
    print(f"   Equity: ${current_equity:.3f}")
    
    return int(action), signals[action]

# ----------------------------
# 4. SAVE NORMALIZATION STATS (Run ONCE)
# ----------------------------

def save_training_stats(train_env):
    """Call after training to enable production."""
    np.save("feat_mean.npy", train_env.feat_mean.ravel())
    np.save("feat_std.npy", train_env.feat_std.ravel())
    print("✅ SAVED feat_mean.npy & feat_std.npy")

# ----------------------------
# 5. PRODUCTION USAGE
# ----------------------------

if __name__ == "__main__":
    SQL_CONN_STR = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    
    print("🚀 PRODUCTION DQN SIGNAL GENERATOR")
    
    # REAL PORTFOLIO STATE
    current_position = 0.0      # Your current SPY exposure (-1 to +1)
    current_equity = 105000.0   # Your portfolio value
    
    # GET SIGNAL
    action, signal = get_trading_signal(
        SQL_CONN_STR, 
        "dqn_spy.zip", 
        "SPY",
        current_position,
        current_equity / 100000  # Normalize to training scale (1.0)
    )
    
    # EXECUTE TRADE
    if action == 1 and current_position <= 0:
        print("📈 → BUY SPY")
    elif action == 2 and current_position >= 0:
        print("📉 → SHORT SPY")
    elif action == 0:
        print("🟡 → CLOSE POSITION")
    else:
        print("✅ HOLD CURRENT POSITION")
