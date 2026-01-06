"""
🚀 RETRAIN DQN WITH RECENT DATA (2015-2024)
Includes COVID crash, volatility regimes, and recent market conditions
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

print("🎯 RETRAINING DQN - RECENT DATA (2015-2024)")

# ----------------------------
# CONFIGURATION - UPDATED DATES
# ----------------------------

SYMBOL = "SPY"

BASE_FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
                 'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']

ADVANCED_FEATURES = [
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'atr_14', 'vol_ratio', 'vol_trend',
    'adx_14', 'bb_position',
    'range_norm', 'ret_skew', 'obv_ratio'
]

ALL_FEATURES = BASE_FEATURES + ADVANCED_FEATURES

# ===== KEY CHANGE: Updated training dates =====
TRAIN_START = "2015-01-01"      # Changed from 2005
TRAIN_END = "2024-01-01"        # Changed from 2018
TRAINING_STEPS = 500_000        # Increased from 300K

print(f"📊 Training Configuration:")
print(f"   Period: {TRAIN_START} to {TRAIN_END}")
print(f"   Features: {len(ALL_FEATURES)} (8 base + 12 advanced)")
print(f"   Steps: {TRAINING_STEPS:,}")
print(f"   Includes: COVID crash, volatility regimes\n")

# ----------------------------
# TECHNICAL INDICATORS (same as before)
# ----------------------------

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return (rsi - 50) / 50

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    macd_line_norm = macd_line / prices * 100
    macd_signal_norm = macd_signal / prices * 100
    macd_hist_norm = macd_hist / prices * 100
    return {'macd_line': macd_line_norm, 'macd_signal': macd_signal_norm, 'macd_hist': macd_hist_norm}

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr / close

def calculate_adx(high, low, close, period=14):
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
    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    bb_high = sma + (std * num_std)
    bb_low = sma - (std * num_std)
    bb_pos = (close - bb_low) / (bb_high - bb_low + 1e-8)
    return bb_pos.clip(0, 1)

def calculate_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ma = obv.rolling(window=20, min_periods=20).mean()
    obv_std = obv.rolling(window=20, min_periods=20).std()
    obv_ratio = obv / (obv_ma + 1e-8)
    return obv_ratio.clip(-5, 5) / 5

# ----------------------------
# DATA LOADING & FEATURES
# ----------------------------

def load_and_compute_features(conn_str: str):
    """Load data and compute features."""
    print("📥 Loading data from SQL...")
    engine = sa.create_engine(conn_str)
    
    query = f"""
    SELECT TradeDate, [close], [high], [low], volume
    FROM AI_ETF_Prices
    WHERE symbol = '{SYMBOL}' AND TradeDate >= '{TRAIN_START}' AND TradeDate <= '{TRAIN_END}'
    ORDER BY TradeDate
    """
    
    df = pd.read_sql(query, engine)
    df['TradeDate'] = pd.to_datetime(df['TradeDate'])
    print(f"✅ Loaded {len(df)} days\n")
    
    # Compute features
    print("🔧 Computing features...")
    df = df.sort_values('TradeDate').reset_index(drop=True)
    
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
    
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    macd_dict = calculate_macd(df['close'])
    df['macd_line'] = macd_dict['macd_line']
    df['macd_signal'] = macd_dict['macd_signal']
    df['macd_hist'] = macd_dict['macd_hist']
    
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    df['vol_ratio'] = df['vol_20d'] / (df['vol_10d'] + 1e-8)
    df['vol_trend'] = df['vol_20d'].pct_change(5)
    
    df['adx_14'] = calculate_adx(df['high'], df['low'], df['close'], 14)
    df['bb_position'] = calculate_bb_position(df['close'], 20)
    
    df['high_20'] = df['close'].rolling(20).max()
    df['low_20'] = df['close'].rolling(20).min()
    df['range_norm'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
    
    df['ret_skew'] = df['ret_1d'].rolling(20, min_periods=10).skew()
    df['obv_ratio'] = calculate_obv(df['close'], df['volume'])
    
    df['ret_5d_fwd'] = df['close'].shift(-5) / df['close'] - 1.0
    df['ret_60d_fwd'] = df['close'].shift(-60) / df['close'] - 1.0
    
    result = df[ALL_FEATURES + ['TradeDate', 'close', 'ret_5d_fwd', 'ret_60d_fwd']].copy()
    result = result.fillna(method='bfill').fillna(method='ffill').dropna()
    result = result.reset_index(drop=True)
    
    print(f"✅ {len(result)} days with features\n")
    return result

# ----------------------------
# TRAINING ENVIRONMENT (same as before)
# ----------------------------

class DualHorizonAdvancedEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        self.features_raw = df[ALL_FEATURES].fillna(0).values.astype(np.float32)
        self.prices = df['close'].values.astype(np.float32)
        self.ret_5d = df['ret_5d_fwd'].values.astype(np.float32)
        self.ret_60d = df['ret_60d_fwd'].values.astype(np.float32)
        
        self.n_steps = len(self.prices)
        
        self.feat_mean = self.features_raw.mean(axis=0, keepdims=True)
        self.feat_std = self.features_raw.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features_raw - self.feat_mean) / self.feat_std
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        self.max_equity = 1.0
        self.recent_pnl = []
        
        self.reset()
    
    def _get_obs(self):
        feat = self.features[self.step_count]
        state = np.array([self.pos, self.equity], dtype=np.float32)
        obs = np.concatenate([feat, state]).astype(np.float32)
        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 100
        self.pos = 0.0
        self.equity = 1.0
        self.last_price = self.prices[self.step_count - 1]
        self.max_equity = 1.0
        self.recent_pnl = []
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
        
        self.recent_pnl.append(pnl)
        if len(self.recent_pnl) > 20:
            self.recent_pnl.pop(0)
        
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        ret_5d = self.ret_5d[self.step_count]
        ret_60d = self.ret_60d[self.step_count]
        
        vol_current = abs(self.features[self.step_count][3]) + 0.1
        vol_normalized = min(1.0, 0.025 / vol_current)
        expected_return = (0.6 * ret_60d + 0.4 * ret_5d) * vol_normalized
        
        position_reward = self.pos * expected_return
        
        current_dd = (self.equity - self.max_equity) / (self.max_equity + 1e-8)
        dd_penalty = max(0, -current_dd * 0.1)
        
        transaction_cost = pos_change * self.equity * 0.00005 * 2
        
        if len(self.recent_pnl) >= 20:
            recent_wins = sum([1 for p in self.recent_pnl[-20:] if p > 0])
            win_bonus = (recent_wins / 20 - 0.5) * 0.01
        else:
            win_bonus = 0
        
        vol_regime = vol_current / 0.015
        vol_multiplier = 1.0 / (1 + vol_regime)
        
        total_reward = (
            position_reward * vol_multiplier
            - dd_penalty
            - transaction_cost
            + win_bonus
        )
        
        self.step_count += 1
        done = self.equity <= 0.1 or self.step_count >= self.n_steps - 1
        
        obs = self._get_obs()
        return obs, float(total_reward), done, False, {}

# ----------------------------
# TRAINING
# ----------------------------

def train():
    """Main training function."""
    
    print(f"{'='*60}")
    print(f"🚀 RETRAINING DQN - RECENT DATA (2015-2024)")
    print(f"{'='*60}\n")
    
    # Load data
    SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    df = load_and_compute_features(SQL_CONN)
    
    # Create environment
    print("🏗️  Creating training environment...")
    train_env = DummyVecEnv([lambda: DualHorizonAdvancedEnv(df)])
    print(f"✅ Environment ready\n")
    
    # Create model
    print("🤖 Creating DQN model...")
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=128,
        gamma=0.99,
        target_update_interval=500,
        train_freq=1,
        exploration_fraction=0.15,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[512, 512, 512]),
        verbose=1,
        device='auto'
    )
    print(f"✅ Model created\n")
    
    # Train
    print(f"🎓 Training for {TRAINING_STEPS:,} steps...")
    print(f"⏱️  Estimated time: 20-25 minutes\n")
    
    start_time = datetime.now()
    model.learn(total_timesteps=TRAINING_STEPS, progress_bar=True)
    elapsed = datetime.now() - start_time
    
    print(f"\n✅ Training complete in {elapsed}")
    
    # Save
    print(f"\n💾 Saving model...")
    model_name = f"dqn_{SYMBOL.lower()}_retrained_2015_2024"
    model.save(model_name)
    
    test_env = DualHorizonAdvancedEnv(df)
    np.save("feat_mean_retrained.npy", test_env.feat_mean.ravel())
    np.save("feat_std_retrained.npy", test_env.feat_std.ravel())
    
    print(f"✅ Model: {model_name}.zip")
    print(f"✅ Stats: feat_mean_retrained.npy, feat_std_retrained.npy")
    
    print(f"\n{'='*60}")
    print(f"🎉 RETRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n📊 Model Details:")
    print(f"   Training Period: {TRAIN_START} to {TRAIN_END}")
    print(f"   Training Days: {len(df):,}")
    print(f"   Includes: COVID crash, volatility regimes, recent data")
    print(f"   Training Steps: {TRAINING_STEPS:,}")
    print(f"   Time: {elapsed}")
    print(f"\n📋 Next: Run backtest")
    print(f"   python dual_horizon_backtest_retrained.py")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    train()
