"""
🚀 DUAL-HORIZON DQN WITH 12 ADVANCED FEATURES
✅ 20 total features ✅ Enhanced technical indicators ✅ Production ready
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

print("🎯 DUAL-HORIZON DQN WITH ADVANCED FEATURES")

# ----------------------------
# 1. CONFIGURATION
# ----------------------------

SYMBOL = "SPY"

# ORIGINAL 8 FEATURES
BASE_FEATURES = ['ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
                 'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20']

# NEW 12 ADVANCED FEATURES
ADVANCED_FEATURES = [
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'atr_14', 'vol_ratio', 'vol_trend',
    'adx_14', 'bb_position',
    'range_norm', 'ret_skew', 'obv_ratio'
]

# ALL 20 FEATURES
ALL_FEATURES = BASE_FEATURES + ADVANCED_FEATURES

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
    return (rsi - 50) / 50  # Normalize to (-1, 1)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    
    # Normalize
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
    
    return atr / close  # Normalize

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
    
    return adx.clip(0, 100) / 100  # Normalize to 0-1

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
    
    return obv_ratio.clip(-5, 5) / 5  # Normalize

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
    
    # ===== NEW 12 ADVANCED FEATURES =====
    
    print("   Computing momentum indicators...")
    # 1. MOMENTUM INDICATORS
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    
    macd_dict = calculate_macd(df['close'])
    df['macd_line'] = macd_dict['macd_line']
    df['macd_signal'] = macd_dict['macd_signal']
    df['macd_hist'] = macd_dict['macd_hist']
    
    print("   Computing volatility indicators...")
    # 2. VOLATILITY INDICATORS
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    df['vol_ratio'] = df['vol_20d'] / (df['vol_10d'] + 1e-8)
    df['vol_trend'] = df['vol_20d'].pct_change(5)
    
    print("   Computing trend indicators...")
    # 3. TREND INDICATORS
    df['adx_14'] = calculate_adx(df['high'], df['low'], df['close'], 14)
    df['bb_position'] = calculate_bb_position(df['close'], 20)
    
    print("   Computing price action indicators...")
    # 4. PRICE ACTION
    df['high_20'] = df['close'].rolling(20).max()
    df['low_20'] = df['close'].rolling(20).min()
    df['range_norm'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
    
    print("   Computing return patterns...")
    # 5. RETURN PATTERNS
    df['ret_skew'] = df['ret_1d'].rolling(20, min_periods=10).skew()
    
    print("   Computing volume-price indicators...")
    # 6. VOLUME-PRICE
    df['obv_ratio'] = calculate_obv(df['close'], df['volume'])
    
    # ===== FORWARD LABELS =====
    
    # 5-day forward return
    df['ret_5d_fwd'] = df['close'].shift(-5) / df['close'] - 1.0
    
    # 60-day forward return
    df['ret_60d_fwd'] = df['close'].shift(-60) / df['close'] - 1.0
    
    # Clean NaNs
    result = df[ALL_FEATURES + ['TradeDate', 'close', 'ret_5d_fwd', 'ret_60d_fwd']].copy()
    result = result.fillna(method='bfill').fillna(method='ffill').dropna()
    result = result.reset_index(drop=True)
    
    print(f"✅ {len(result)} days with 20 features")
    print(f"   Base features (8): {BASE_FEATURES}")
    print(f"   Advanced features (12): {ADVANCED_FEATURES}")
    
    return result

# ----------------------------
# 5. TRAINING ENVIRONMENT
# ----------------------------

class DualHorizonAdvancedEnv(gym.Env):
    """
    Custom Gym environment with 20 advanced features.
    
    State: 20 features + position + equity = 22 values
    Action: 0=FLAT, 1=LONG, 2=SHORT
    Reward: Enhanced dual-horizon reward
    """
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        print("   Initializing environment with 20 features...")
        
        # Extract features
        self.features_raw = df[ALL_FEATURES].fillna(0).values.astype(np.float32)
        
        # Validate shape
        if self.features_raw.shape[1] != 20:
            raise ValueError(f"Expected 20 features, got {self.features_raw.shape[1]}")
        
        # Price data
        self.prices = df['close'].values.astype(np.float32)
        self.highs = df['high'].values.astype(np.float32) if 'high' in df else self.prices
        self.lows = df['low'].values.astype(np.float32) if 'low' in df else self.prices
        
        # Forward returns (labels for reward)
        self.ret_5d = df['ret_5d_fwd'].values.astype(np.float32)
        self.ret_60d = df['ret_60d_fwd'].values.astype(np.float32)
        
        self.n_steps = len(self.prices)
        
        # Normalize features
        self.feat_mean = self.features_raw.mean(axis=0, keepdims=True)
        self.feat_std = self.features_raw.std(axis=0, keepdims=True) + 1e-8
        self.features = (self.features_raw - self.feat_mean) / self.feat_std
        
        # Define spaces
        # Observation: 20 normalized features + position + equity = 22 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(22,),  # CHANGED FROM 10 TO 22
            dtype=np.float32
        )
        
        # Action: 0=FLAT, 1=LONG, 2=SHORT
        self.action_space = spaces.Discrete(3)
        
        # Track metrics
        self.max_equity = 1.0
        self.recent_pnl = []
        
        print(f"   ✅ Env ready: {self.n_steps} steps, 20 features")
        self.reset()
    
    def _get_obs(self):
        """Get current observation (20 normalized features + state)."""
        feat = self.features[self.step_count]  # (20,)
        state = np.array([self.pos, self.equity], dtype=np.float32)  # (2,)
        obs = np.concatenate([feat, state]).astype(np.float32)  # (22,)
        
        assert obs.shape == (22,), f"Observation shape {obs.shape} != (22,)"
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Start training from day 100 (allow warmup)
        self.step_count = 100
        self.pos = 0.0
        self.equity = 1.0
        self.last_price = self.prices[self.step_count - 1]
        self.max_equity = 1.0
        self.recent_pnl = []
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one trading step with enhanced reward."""
        
        # Map action to position
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
        
        # Track for bonus
        self.recent_pnl.append(pnl)
        if len(self.recent_pnl) > 20:
            self.recent_pnl.pop(0)
        
        # Update max equity for drawdown penalty
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        # ===== ENHANCED DUAL-HORIZON REWARD =====
        
        # Get forward returns
        ret_5d = self.ret_5d[self.step_count]
        ret_60d = self.ret_60d[self.step_count]
        
        # Get current volatility (from vol_20d feature, index 3)
        vol_current = abs(self.features[self.step_count][3]) + 0.1
        
        # 1. EXPECTED RETURN REWARD
        # Weight by volatility (higher vol = lower confidence)
        vol_normalized = min(1.0, 0.025 / vol_current)
        expected_return = (0.6 * ret_60d + 0.4 * ret_5d) * vol_normalized
        
        # 2. POSITION ALIGNMENT REWARD
        position_reward = self.pos * expected_return
        
        # 3. DRAWDOWN PENALTY
        current_dd = (self.equity - self.max_equity) / (self.max_equity + 1e-8)
        dd_penalty = max(0, -current_dd * 0.1)
        
        # 4. TRADING COST
        transaction_cost = pos_change * self.equity * 0.00005 * 2
        
        # 5. WIN RATE BONUS
        if len(self.recent_pnl) >= 20:
            recent_wins = sum([1 for p in self.recent_pnl[-20:] if p > 0])
            win_bonus = (recent_wins / 20 - 0.5) * 0.01
        else:
            win_bonus = 0
        
        # 6. VOLATILITY REGIME ADJUSTMENT
        # In high vol, prefer FLAT. In low vol, prefer trades
        vol_regime = vol_current / 0.015  # Compare to baseline
        vol_multiplier = 1.0 / (1 + vol_regime)
        
        # COMBINE REWARDS
        total_reward = (
            position_reward * vol_multiplier
            - dd_penalty
            - transaction_cost
            + win_bonus
        )
        
        # Step counter
        self.step_count += 1
        
        # Episode ends when equity depleted or max steps reached
        done = self.equity <= 0.1 or self.step_count >= self.n_steps - 1
        
        obs = self._get_obs()
        
        return obs, float(total_reward), done, False, {}

# ----------------------------
# 6. TRAINING FUNCTION
# ----------------------------

def train_dual_horizon_advanced(conn_str: str):
    """Complete training pipeline with advanced features."""
    
    print(f"\n{'='*60}")
    print(f"🎯 DUAL-HORIZON DQN - ADVANCED FEATURES")
    print(f"{'='*60}\n")
    
    # ===== STEP 1: Load Data =====
    print(f"STEP 1: Load training data")
    print(f"-" * 40)
    df_raw = load_data(conn_str, SYMBOL, TRAIN_START, TRAIN_END)
    
    # ===== STEP 2: Compute Advanced Features =====
    print(f"\nSTEP 2: Engineer 20 advanced features")
    print(f"-" * 40)
    df = compute_advanced_features(df_raw)
    
    # ===== STEP 3: Create Environment =====
    print(f"\nSTEP 3: Create training environment")
    print(f"-" * 40)
    train_env = DummyVecEnv([lambda: DualHorizonAdvancedEnv(df)])
    
    # ===== STEP 4: Create DQN Model =====
    print(f"\nSTEP 4: Create DQN model")
    print(f"-" * 40)
    print(f"   Model Architecture:")
    print(f"   - Input size: 22 (20 features + 2 state)")
    print(f"   - Hidden layers: [512, 512, 512]")
    print(f"   - Dueling: Yes")
    print(f"\n   Hyperparameters:")
    print(f"   - Learning rate: 5e-4")
    print(f"   - Buffer size: 100,000")
    print(f"   - Batch size: 128")
    print(f"   - Gamma: 0.99")
    print(f"   - Total steps: {TRAINING_STEPS:,}")
    
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
    policy_kwargs=dict(
        net_arch=[512, 512, 512]   # ✅ CORRECT - simple architecture
    ),
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
    
    model_name = f"dqn_{SYMBOL.lower()}_advanced"
    model.save(model_name)
    print(f"   ✅ Model: {model_name}.zip")
    
    # Save normalization statistics
    test_env = DualHorizonAdvancedEnv(df)
    np.save("feat_mean_advanced.npy", test_env.feat_mean.ravel())
    np.save("feat_std_advanced.npy", test_env.feat_std.ravel())
    print(f"   ✅ Stats: feat_mean_advanced.npy, feat_std_advanced.npy")
    
    # ===== STEP 7: Validation =====
    print(f"\nSTEP 7: Quick validation")
    print(f"-" * 40)
    
    obs, _ = test_env.reset()
    print(f"   Observation shape: {obs.shape} (should be (22,))")
    
    action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
    action_name = ['FLAT', 'LONG', 'SHORT'][int(action[0])]
    print(f"   Test prediction: {action_name}")
    
    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Model: {model_name}.zip")
    print(f"📊 Normalization: feat_mean_advanced.npy, feat_std_advanced.npy")
    print(f"📚 Training data: {len(df):,} days ({TRAIN_START} to {TRAIN_END})")
    print(f"🎯 Features: 20 advanced technical indicators")
    print(f"   - Base (8): {BASE_FEATURES}")
    print(f"   - Advanced (12): {ADVANCED_FEATURES}")
    print(f"🎓 Steps: {TRAINING_STEPS:,}")
    print(f"⏱️  Time: {elapsed}")
    print(f"\n📋 Next step:")
    print(f"   python dual_horizon_backtest_advanced.py")
    print(f"   (Run backtest on 2020-2025 data)")
    print(f"\n" + "="*60)

# ----------------------------
# 7. MAIN ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    # SQL Connection
    SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    
    try:
        train_dual_horizon_advanced(SQL_CONN)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n📋 Troubleshooting:")
        print(f"   - Check SQL connection string")
        print(f"   - Verify SPY data exists with high/low columns")
        print(f"   - Ensure sufficient disk space")
