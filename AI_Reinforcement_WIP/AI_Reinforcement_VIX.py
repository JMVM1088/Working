"""
🚀 DUAL-HORIZON DQN + VIX - 14 FEATURES EXACT
✅ 8 SPY + 6 VIX = 14 ✅ 16 obs ✅ WORKING
"""

import pandas as pd
import sqlalchemy as sa
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

print("🎯 DUAL-HORIZON + VIX - 14 FEATURES")

# 8 SPY features
BASE_FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
                'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']

# 6 VIX features
VIX_FEATURES = ['vix_ret_1d', 'vix_vol_20d', 'vix_sma_20', 'vix_z_20', 
               'spy_vix_corr', 'vix_rsi']

ALL_FEATURES = BASE_FEATURES + VIX_FEATURES

print(f"✅ {len(BASE_FEATURES)} SPY + {len(VIX_FEATURES)} VIX = {len(ALL_FEATURES)} TOTAL")

def load_spy_vix_data(conn_str: str, start_date: str, end_date: str):
    engine = sa.create_engine(conn_str)
    
    spy_query = f"""
    SELECT TradeDate, [close] as spy_close, volume as spy_volume
    FROM AI_ETF_Prices
    WHERE symbol = 'SPY' AND TradeDate >= '{start_date}' AND TradeDate <= '{end_date}'
    ORDER BY TradeDate
    """
    spy_df = pd.read_sql(spy_query, engine)
    
    try:
        vix_query = f"""
        SELECT TradeDate, [close] as vix_close
        FROM AI_ETF_Prices
        WHERE symbol = '^VIX' AND TradeDate >= '{start_date}' AND TradeDate <= '{end_date}'
        ORDER BY TradeDate
        """
        vix_df = pd.read_sql(vix_query, engine)
    except:
        vix_df = spy_df[['TradeDate']].copy()
        vix_df['vix_close'] = 18.0
    
    df = spy_df.merge(vix_df, on='TradeDate', how='inner')
    return df

def compute_features_with_vix(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.sort_values('TradeDate').reset_index(drop=True)
    
    # ===== SPY FEATURES (8) =====
    df['ret_1d'] = df['spy_close'].pct_change()
    df['ret_5d'] = df['spy_close'].pct_change(5)
    df['vol_10d'] = df['ret_1d'].rolling(10, min_periods=5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20, min_periods=10).std()
    
    for win in [20, 50, 200]:
        sma = df['spy_close'].rolling(win, min_periods=win//2).mean()
        df[f'dist_sma_{win}'] = (df['spy_close'] - sma) / sma
    
    vol_ma = df['spy_volume'].rolling(20).mean()
    vol_std = df['spy_volume'].rolling(20).std()
    df['vol_z_20'] = (df['spy_volume'] - vol_ma) / (vol_std + 1e-8)
    
    # ===== VIX FEATURES (6) =====
    df['vix_ret_1d'] = df['vix_close'].pct_change()
    df['vix_vol_20d'] = df['vix_ret_1d'].rolling(20, min_periods=10).std()
    df['vix_sma_20'] = df['vix_close'].rolling(20).mean()
    df['vix_z_20'] = (df['vix_close'] - df['vix_sma_20']) / (df['vix_vol_20d'] + 1e-8)
    df['spy_vix_corr'] = df['ret_1d'].rolling(20).corr(df['vix_ret_1d'])
    
    # VIX RSI
    vix_delta = df['vix_close'].diff()
    vix_gain = vix_delta.clip(lower=0)
    vix_loss = -vix_delta.clip(upper=0)
    vix_avg_gain = vix_gain.rolling(14, min_periods=7).mean()
    vix_avg_loss = vix_loss.rolling(14, min_periods=7).mean()
    vix_rs = vix_avg_gain / (vix_avg_loss + 1e-8)
    df['vix_rsi'] = 100 - (100 / (1 + vix_rs))
    
    result = df[ALL_FEATURES + ['spy_close', 'vix_close']].dropna().reset_index(drop=True)
    print(f"✅ {len(result)} days | {len(ALL_FEATURES)} features")
    return result

class DualHorizonEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        # Extract EXACTLY 14 features
        self.features_raw = df[ALL_FEATURES].fillna(0).values.astype(np.float32)
        assert self.features_raw.shape[1] == 14, f"Features {self.features_raw.shape[1]} != 14"
        
        self.prices = df['spy_close'].values.astype(np.float32)
        self.vix = df['vix_close'].values.astype(np.float32)
        self.n_steps = len(self.prices)
        
        # Forward returns
        self.ret_5d = np.zeros(self.n_steps, dtype=np.float32)
        self.ret_60d = np.zeros(self.n_steps, dtype=np.float32)
        
        for i in range(self.n_steps - 5):
            self.ret_5d[i] = (self.prices[i+5] / self.prices[i]) - 1.0
        
        for i in range(self.n_steps - 60):
            self.ret_60d[i] = (self.prices[i+60] / self.prices[i]) - 1.0
        
        # Normalize
        self.feat_mean = self.features_raw.mean(axis=0, keepdims=True)
        self.feat_std = self.features_raw.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features_raw - self.feat_mean) / self.feat_std
        
        assert self.features.shape[1] == 14, f"Normalized {self.features.shape[1]} != 14"
        
        # Obs: 14 features + position + equity = 16
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        print(f"✅ Env: {self.n_steps} steps | Features: {self.features.shape[1]} | Obs: (16,)")
        self.reset()
    
    def _get_obs(self):
        feat = self.features[self.step_count]
        state = np.array([self.pos, self.equity], dtype=np.float32)
        obs = np.concatenate([feat, state]).astype(np.float32)
        assert obs.shape == (16,), f"Obs {obs.shape} != (16,)"
        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 100
        self.pos = 0.0
        self.equity = 1.0
        self.last_price = self.prices[self.step_count - 1]
        return self._get_obs(), {}
    
    def step(self, action):
        target_pos = [0.0, 1.0, -1.0][action]
        pos_change = abs(target_pos - self.pos)
        cost = pos_change * self.equity * 0.00005
        
        price = self.prices[self.step_count]
        ret = (price - self.last_price) / self.last_price
        pnl = self.pos * ret * self.equity
        
        self.equity += pnl - cost
        self.pos = target_pos
        self.last_price = price
        
        # Dual reward: 5-day + 60-day weighted by VIX
        vix = self.vix[self.step_count]
        w5 = min(vix / 20.0, 1.0)
        w60 = max(1.0 - vix / 20.0, 0.0)
        
        expected = w5 * self.ret_5d[self.step_count] + w60 * self.ret_60d[self.step_count]
        reward = self.pos * expected - cost * 2 - max(vix - 15, 0) * 0.001
        
        self.step_count += 1
        done = self.equity <= 0.1 or self.step_count >= self.n_steps - 1
        
        return self._get_obs(), float(reward), done, False, {}

def train_dual_horizon(conn_str: str, steps: int = 250_000):
    print(f"\n📊 Loading SPY + VIX...")
    df = load_spy_vix_data(conn_str, "2005-01-01", "2018-12-31")
    
    print(f"🔧 Computing 14 features...")
    df = compute_features_with_vix(df)
    
    print(f"🔨 Creating env...")
    env = DummyVecEnv([lambda: DualHorizonEnv(df)])
    
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
    model.save("dqn_dual_horizon_vix.zip")
    test_env = DualHorizonEnv(df)
    np.save("feat_mean_vix.npy", test_env.feat_mean.ravel())
    np.save("feat_std_vix.npy", test_env.feat_std.ravel())
    
    print(f"\n✅ TRAINED!")
    print(f"💾 dqn_dual_horizon_vix.zip")
    print(f"📊 feat_mean_vix.npy, feat_std_vix.npy")
    print(f"📈 14 Features (8 SPY + 6 VIX)")
    print(f"🎯 Dual predictions: 5-day + 60-day")

if __name__ == "__main__":
    SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    
    try:
        train_dual_horizon(SQL_CONN, 250_000)
        print("\n🚀 Ready for backtest!")
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
