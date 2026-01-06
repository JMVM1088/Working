"""
COMPLETE RL TRADING SCRIPT - FIXED UnboundLocalError
AI_ETF_Prices + TradeDate + NO tensorboard + CORRECT test_env
"""

import numpy as np
import pandas as pd
import sqlalchemy as sa
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')

print("✅ All imports loaded successfully!")

# ----------------------------
# 1. SQL Data Loading
# ----------------------------

def load_ohlcv_from_sql(conn_str: str, table: str = "AI_ETF_Prices", 
                       symbols: tuple = ("SPY",), start_date: str = None, 
                       end_date: str = None) -> pd.DataFrame:
    print("🔍 Loading from AI_ETF_Prices...")
    engine = sa.create_engine(conn_str)
    
    sym_list = ",".join(f"'{s}'" for s in symbols)
    date_filter = ""
    if start_date:
        date_filter += f" AND TradeDate >= '{start_date}'"
    if end_date:
        date_filter += f" AND TradeDate <= '{end_date}'"
    
    query = f"""
    SELECT symbol, TradeDate, [open], high, low, [close], volume
    FROM {table}
    WHERE symbol IN ({sym_list}){date_filter}
    ORDER BY TradeDate, symbol
    """
    
    df = pd.read_sql(query, engine)
    print(f"✅ Loaded {len(df)} rows")
    print(f"📅 Range: {df['TradeDate'].min()} to {df['TradeDate'].max()}")
    return df

# ----------------------------
# 2. Feature Engineering
# ----------------------------

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty data")
    
    df = df.copy()
    df['date_clean'] = pd.to_datetime(df['TradeDate'])
    df = df.sort_values(['date_clean', 'symbol']).reset_index(drop=True)
    
    print(f"📊 {len(df)} rows for {df['symbol'].nunique()} symbols")
    
    FIXED_FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
                     'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']
    
    feature_groups = []
    for symbol in df['symbol'].unique():
        sym_df = df[df['symbol'] == symbol].copy()
        sym_df = sym_df.set_index('date_clean').sort_index()
        
        if len(sym_df) < 200: continue
        
        sym_df['ret_1d'] = sym_df['close'].pct_change()
        sym_df['ret_5d'] = sym_df['close'].pct_change(5)
        sym_df['vol_10d'] = sym_df['ret_1d'].rolling(10, min_periods=5).std()
        sym_df['vol_20d'] = sym_df['ret_1d'].rolling(20, min_periods=10).std()
        
        for win in [20, 50, 200]:
            sma = sym_df['close'].rolling(win, min_periods=win//2).mean()
            sym_df[f'dist_sma_{win}'] = (sym_df['close'] - sma) / sma
        
        vol_ma = sym_df['volume'].rolling(20).mean()
        vol_std = sym_df['volume'].rolling(20).std()
        sym_df['vol_z_20'] = (sym_df['volume'] - vol_ma) / (vol_std + 1e-8)
        
        sym_df['symbol'] = symbol
        feature_groups.append(sym_df.reset_index())
    
    df_features = pd.concat(feature_groups, axis=0)
    df_features = df_features[['date_clean', 'symbol', 'close'] + FIXED_FEATURES].dropna()
    
    print(f"✅ {len(df_features)} valid rows")
    return df_features

# ----------------------------
# 3. Gym Environment
# ----------------------------

class DailyTradingEnv(gym.Env):
    FIXED_FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
                     'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']
    
    def __init__(self, df: pd.DataFrame, symbol: str, trading_cost_bps: float = 1.0):
        super().__init__()
        self.symbol = symbol
        self.trading_cost_bps = trading_cost_bps / 10000.0
        
        self.df = df[df['symbol'] == symbol].copy()
        self.df = self.df.sort_values('date_clean').set_index('date_clean')
        
        self.feature_cols = [f for f in self.FIXED_FEATURES if f in self.df.columns]
        self.df = self.df[['close'] + self.feature_cols].dropna()
        
        self.features = self.df[self.feature_cols].fillna(0).values.astype(np.float32)
        self.prices = self.df['close'].values.astype(np.float32)
        
        self.feat_mean = self.features.mean(axis=0, keepdims=True)
        self.feat_std = self.features.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features - self.feat_mean) / self.feat_std
        
        obs_dim = len(self.feature_cols) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        print(f"✅ {symbol}: {len(self.prices)} days")
        self.reset()
    
    def _get_obs(self):
        feat = self.features[self.current_step]
        return np.concatenate([feat.ravel(), [self.position, self.equity]], axis=0)
    
    def reset(self, seed=None, options=None):
        self.current_step = max(50, len(self.features) // 4)
        self.position = 0.0
        self.equity = 1.0
        self.last_price = self.prices[self.current_step - 1]
        return self._get_obs(), {}
    
    def step(self, action):
        target_pos = [0.0, 1.0, -1.0][action]
        pos_change = abs(target_pos - self.position)
        trade_cost = pos_change * self.equity * self.trading_cost_bps
        
        price = self.prices[self.current_step]
        ret = (price - self.last_price) / self.last_price
        pnl = self.position * ret * self.equity
        
        self.equity += pnl - trade_cost
        self.position = target_pos
        self.last_price = price
        self.current_step += 1
        
        reward = pnl - trade_cost
        terminated = self.equity <= 0.1
        truncated = self.current_step >= len(self.prices) - 1
        
        obs = self._get_obs()
        info = {'equity': self.equity, 'position': self.position}
        return obs, float(reward), terminated, truncated, info

# ----------------------------
# 4. FIXED Training Pipeline
# ----------------------------

def create_envs(conn_str: str, symbol: str, split_date: str = "2018-01-01"):
    """FIXED: test_df → test_env"""
    raw_data = load_ohlcv_from_sql(conn_str, symbols=(symbol,), start_date="2000-01-01")
    features_df = add_basic_features(raw_data)
    
    split_dt = pd.to_datetime(split_date)
    train_df = features_df[features_df['date_clean'] < split_dt]
    test_df = features_df[features_df['date_clean'] >= split_dt]  # ← FIXED
    
    train_env = DailyTradingEnv(train_df, symbol)
    test_env = DailyTradingEnv(test_df, symbol)  # ← FIXED: was test_env
    
    print(f"✅ Train: {len(train_df)} days, Test: {len(test_df)} days")
    return train_env, test_env

def train_dqn(env: gym.Env, timesteps: int = 25_000):
    vec_env = DummyVecEnv([lambda: env])
    model = DQN(
        "MlpPolicy", vec_env,
        learning_rate=1e-3, buffer_size=50000, learning_starts=1000,
        batch_size=32, gamma=0.99, target_update_interval=500, train_freq=4,
        verbose=1
    )
    print(f"🎯 Training {timesteps:,} steps...")
    model.learn(total_timesteps=timesteps)
    return model

def evaluate_model(model, test_env, n_episodes=3):
    print("\n📊 TEST RESULTS:")
    for ep in range(n_episodes):
        obs, _ = test_env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
        print(f"  Episode {ep+1}: Equity={info['equity']:.3f}")

# ----------------------------
# 5. MAIN
# ----------------------------

if __name__ == "__main__":
    # UPDATE ONLY THIS LINE
    SQL_CONN_STR = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    SYMBOL = "SPY"
    
    print(f"🚀 DQN {SYMBOL} Trading Bot - AI_ETF_Prices")
    
    try:
        train_env, test_env = create_envs(SQL_CONN_STR, SYMBOL)
        model = train_dqn(train_env, timesteps=25_000)
        
        model.save(f"dqn_{SYMBOL.lower()}")
        print(f"💾 Saved: dqn_{SYMBOL.lower()}.zip")
        
        evaluate_model(model, test_env)
        print("\n🎉 SUCCESS! Model ready.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
