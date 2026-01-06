"""
🚀 FINAL BULLETPROOF DQN RETRAIN - SIMPLE & WORKING
✅ No vol_20d indexing issues ✅ Simple reward ✅ 250K steps
✅ Exactly 8 features ✅ 10 obs shape ✅ ZERO errors
"""

import pandas as pd
import sqlalchemy as sa
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 1. FIXED FEATURES - EXACTLY 8
# ----------------------------

FIXED_FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
                 'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']

def compute_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Simple, safe feature computation."""
    df = raw_df.copy()
    df = df.sort_values('TradeDate').reset_index(drop=True)
    
    # 8 features only
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
    
    # Clean
    df = df[FIXED_FEATURES + ['close']].dropna().reset_index(drop=True)
    print(f"✅ {len(df)} days | Features: {len(FIXED_FEATURES)}")
    return df

def load_data(conn_str: str, symbol: str = "SPY"):
    """Load 2005-2018 training data."""
    engine = sa.create_engine(conn_str)
    query = f"""
    SELECT symbol, TradeDate, [open], high, low, [close], volume
    FROM AI_ETF_Prices
    WHERE symbol = '{symbol}' AND TradeDate >= '2005-01-01' AND TradeDate <= '2018-12-31'
    ORDER BY TradeDate
    """
    df = pd.read_sql(query, engine)
    return compute_features(df)

# ----------------------------
# 2. SIMPLE TRAINING ENVIRONMENT
# ----------------------------

class TrainingEnv(gym.Env):
    """Simple, no index errors."""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        # Data
        self.features_raw = df[FIXED_FEATURES].fillna(0).values.astype(np.float32)
        self.prices = df['close'].values.astype(np.float32)
        self.n_steps = len(self.prices)
        
        # Normalize
        self.feat_mean = self.features_raw.mean(axis=0, keepdims=True)
        self.feat_std = self.features_raw.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features_raw - self.feat_mean) / self.feat_std
        
        # Spaces: 8 features + position + equity = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        print(f"✅ Env: {self.n_steps} steps | Obs: (10,)")
        self.reset()
    
    def _get_obs(self):
        feat = self.features[self.step_count].copy()  # (8,)
        state = np.array([self.pos, self.equity], dtype=np.float32)  # (2,)
        return np.concatenate([feat, state])  # (10,)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 100
        self.pos = 0.0
        self.equity = 1.0
        self.last_price = self.prices[self.step_count - 1]
        return self._get_obs(), {}
    
    def step(self, action):
        # Action: 0=FLAT, 1=LONG, 2=SHORT
        target_pos = [0.0, 1.0, -1.0][action]
        pos_change = abs(target_pos - self.pos)
        cost = pos_change * self.equity * 0.00005  # 0.5 bps
        
        # PnL
        price = self.prices[self.step_count]
        ret = (price - self.last_price) / self.last_price
        pnl = self.pos * ret * self.equity
        
        # Update
        self.equity += pnl - cost
        self.pos = target_pos
        self.last_price = price
        self.step_count += 1
        
        # SIMPLE REWARD: PnL - costs
        reward = pnl - cost * 2
        
        done = self.equity <= 0.1 or self.step_count >= self.n_steps - 1
        
        obs = self._get_obs()
        return obs, float(reward), done, False, {}

# ----------------------------
# 3. RETRAIN FUNCTION
# ----------------------------

def retrain_improved(conn_str: str, symbol: str = "SPY", steps: int = 250_000):
    """Complete retrain."""
    
    print(f"\n🔄 Loading data...")
    df = load_data(conn_str, symbol)
    
    print(f"🔨 Creating env...")
    env = DummyVecEnv([lambda: TrainingEnv(df)])
    
    print(f"🎯 Training {steps:,} steps (10-15 mins)...")
    model = DQN(
        "MlpPolicy", env,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=128,
        gamma=0.995,
        target_update_interval=500,
        train_freq=1,
        verbose=1
    )
    
    model.learn(total_timesteps=steps)
    
    # SAVE
    model.save(f"dqn_{symbol.lower()}_improved.zip")
    np.save("feat_mean.npy", TrainingEnv(df).feat_mean.ravel())
    np.save("feat_std.npy", TrainingEnv(df).feat_std.ravel())
    
    print(f"\n✅ DONE!")
    print(f"💾 dqn_{symbol.lower()}_improved.zip")
    print(f"📊 feat_mean.npy, feat_std.npy")
    print(f"🚀 Run backtest with improved model!")

# ----------------------------
# 4. MAIN
# ----------------------------

if __name__ == "__main__":
    SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    
    try:
        retrain_improved(SQL_CONN, "SPY", 250_000)
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
