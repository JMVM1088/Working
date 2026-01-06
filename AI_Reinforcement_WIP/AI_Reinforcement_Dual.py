"""
🚀 DUAL-HORIZON DQN TRAINING - COMPLETE SCRIPT
✅ Single stock (SPY) ✅ 8 features ✅ 5-day + 60-day predictions
✅ 300K steps training ✅ Production ready
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

print("🎯 DUAL-HORIZON DQN TRAINING")

# ----------------------------
# 1. CONFIGURATION
# ----------------------------

SYMBOL = "SPY"
FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
           'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']
TRAINING_STEPS = 300_000
TRAIN_START = "2005-01-01"
TRAIN_END = "2018-12-31"

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
    print(f"   Features: {FEATURES}")
    print(f"   Labels: ret_5d_fwd, ret_60d_fwd")
    
    return result

# ----------------------------
# 4. TRAINING ENVIRONMENT
# ----------------------------

class DualHorizonTrainingEnv(gym.Env):
    """
    Custom Gym environment for dual-horizon trading.
    
    State: 8 price features + position + equity = 10 values
    Action: 0=FLAT, 1=LONG, 2=SHORT
    Reward: Combined 5-day and 60-day return prediction reward
    """
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        print("   Initializing environment...")
        
        # Extract features
        self.features_raw = df[FEATURES].fillna(0).values.astype(np.float32)
        
        # Validate shape
        if self.features_raw.shape[1] != 8:
            raise ValueError(f"Expected 8 features, got {self.features_raw.shape[1]}")
        
        # Price data
        self.prices = df['close'].values.astype(np.float32)
        
        # Forward returns (labels for reward)
        self.ret_5d = df['ret_5d_fwd'].values.astype(np.float32)
        self.ret_60d = df['ret_60d_fwd'].values.astype(np.float32)
        
        self.n_steps = len(self.prices)
        
        # Normalize features
        self.feat_mean = self.features_raw.mean(axis=0, keepdims=True)
        self.feat_std = self.features_raw.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features_raw - self.feat_mean) / self.feat_std
        
        # Define spaces
        # Observation: 8 normalized features + position + equity = 10 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(10,), 
            dtype=np.float32
        )
        
        # Action: 0=FLAT, 1=LONG, 2=SHORT
        self.action_space = spaces.Discrete(3)
        
        print(f"   ✅ Env ready: {self.n_steps} steps")
        self.reset()
    
    def _get_obs(self):
        """Get current observation (normalized features + state)."""
        feat = self.features[self.step_count]  # (8,)
        state = np.array([self.pos, self.equity], dtype=np.float32)  # (2,)
        obs = np.concatenate([feat, state]).astype(np.float32)  # (10,)
        
        assert obs.shape == (10,), f"Observation shape {obs.shape} != (10,)"
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Start training from day 100 (allow warmup)
        self.step_count = 100
        self.pos = 0.0  # No position
        self.equity = 1.0  # Start with $1
        self.last_price = self.prices[self.step_count - 1]
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one trading step."""
        
        # Map action to position: 0=FLAT, 1=LONG, 2=SHORT
        target_pos = [0.0, 1.0, -1.0][action]
        
        # Calculate position change and transaction cost
        pos_change = abs(target_pos - self.pos)
        cost = pos_change * self.equity * 0.00005  # 0.5 basis points
        
        # Get current price and calculate return
        price = self.prices[self.step_count]
        ret = (price - self.last_price) / self.last_price
        
        # Calculate P&L
        pnl = self.pos * ret * self.equity
        
        # Update equity and position
        self.equity += pnl - cost
        self.pos = target_pos
        self.last_price = price
        
        # ===== DUAL-HORIZON REWARD =====
        # Get forward returns for this step
        ret_5d = self.ret_5d[self.step_count]
        ret_60d = self.ret_60d[self.step_count]
        
        # Combine 5-day and 60-day signals
        # 60% weight on long-term trend, 40% on short-term
        expected_return = 0.6 * ret_60d + 0.4 * ret_5d
        
        # Position alignment reward
        # Reward when position matches expected direction
        position_reward = self.pos * expected_return
        
        # Total reward
        reward = position_reward - cost * 2  # Higher penalty for trading costs
        
        # Step counter
        self.step_count += 1
        
        # Episode ends when equity depleted or max steps reached
        done = self.equity <= 0.1 or self.step_count >= self.n_steps - 1
        
        obs = self._get_obs()
        
        return obs, float(reward), done, False, {}

# ----------------------------
# 5. TRAINING FUNCTION
# ----------------------------

def train_dual_horizon(conn_str: str):
    """Complete training pipeline."""
    
    print(f"\n{'='*60}")
    print(f"🎯 DUAL-HORIZON DQN TRAINING")
    print(f"{'='*60}\n")
    
    # ===== STEP 1: Load Data =====
    print(f"STEP 1: Load training data")
    print(f"-" * 40)
    df_raw = load_data(conn_str, SYMBOL, TRAIN_START, TRAIN_END)
    
    # ===== STEP 2: Compute Features =====
    print(f"\nSTEP 2: Engineer features")
    print(f"-" * 40)
    df = compute_features(df_raw)
    
    # ===== STEP 3: Create Environment =====
    print(f"\nSTEP 3: Create training environment")
    print(f"-" * 40)
    train_env = DummyVecEnv([lambda: DualHorizonTrainingEnv(df)])
    
    # ===== STEP 4: Create DQN Model =====
    print(f"\nSTEP 4: Create DQN model")
    print(f"-" * 40)
    print(f"   Hyperparameters:")
    print(f"   - Learning rate: 5e-4")
    print(f"   - Buffer size: 100,000")
    print(f"   - Batch size: 128")
    print(f"   - Gamma: 0.99")
    print(f"   - Total steps: {TRAINING_STEPS:,}")
    
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=5e-4,           # Learning rate
        buffer_size=100_000,          # Replay buffer size
        learning_starts=2_000,        # Steps before training starts
        batch_size=128,               # Batch size for training
        gamma=0.99,                   # Discount factor
        target_update_interval=500,   # Update target network every 500 steps
        train_freq=1,                 # Train every step
        exploration_fraction=0.15,    # Exploration over 15% of training
        verbose=1,
        device='auto'
    )
    
    # ===== STEP 5: Train Model =====
    print(f"\nSTEP 5: Train model ({TRAINING_STEPS:,} steps)")
    print(f"-" * 40)
    print(f"⏱️  Estimated time: 15-20 minutes\n")
    
    start_time = datetime.now()
    
    model.learn(total_timesteps=TRAINING_STEPS, progress_bar=True)
    
    elapsed = datetime.now() - start_time
    print(f"\n✅ Training complete in {elapsed}")
    
    # ===== STEP 6: Save Model =====
    print(f"\nSTEP 6: Save model and statistics")
    print(f"-" * 40)
    
    model_name = f"dqn_{SYMBOL.lower()}_dual_horizon"
    model.save(model_name)
    print(f"   ✅ Model: {model_name}.zip")
    
    # Save normalization statistics
    test_env = DualHorizonTrainingEnv(df)
    np.save("feat_mean.npy", test_env.feat_mean.ravel())
    np.save("feat_std.npy", test_env.feat_std.ravel())
    print(f"   ✅ Stats: feat_mean.npy, feat_std.npy")
    
    # ===== STEP 7: Validation =====
    print(f"\nSTEP 7: Quick validation")
    print(f"-" * 40)
    
    obs, _ = test_env.reset()
    print(f"   Observation shape: {obs.shape} (should be (10,))")
    
    action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
    action_name = ['FLAT', 'LONG', 'SHORT'][int(action[0])]
    print(f"   Test prediction: {action_name}")
    
    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Model: {model_name}.zip")
    print(f"📊 Normalization: feat_mean.npy, feat_std.npy")
    print(f"📚 Training data: {len(df):,} days ({TRAIN_START} to {TRAIN_END})")
    print(f"🎯 Features: {len(FEATURES)} technical indicators")
    print(f"🎓 Steps: {TRAINING_STEPS:,}")
    print(f"⏱️  Time: {elapsed}")
    print(f"\n📋 Next step:")
    print(f"   python dual_horizon.py")
    print(f"   (Run backtest on 2020-2025 data)")
    print(f"\n" + "="*60)

# ----------------------------
# 6. MAIN ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    # SQL Connection
    SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    
    try:
        train_dual_horizon(SQL_CONN)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n📋 Troubleshooting:")
        print(f"   - Check SQL connection string")
        print(f"   - Verify SPY data exists in AI_ETF_Prices (2005-2018)")
        print(f"   - Ensure sufficient disk space")
