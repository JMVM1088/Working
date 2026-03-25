"""
🚀 FVG-DQN INTEGRATED TRADING SYSTEM
✅ FVG detection integrated into DQN state
✅ FVG-based reward shaping
✅ FVG signal as trading trigger
✅ Clean, production-ready code
"""

import pandas as pd
import sqlalchemy as sa
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import warnings
import urllib
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FVGConfig:
    """FVG detection parameters"""
    threshold_per: float = 0.5  # FVG threshold percentage
    auto_threshold: bool = False  # Auto-calculate threshold
    rr_target: float = 2.0  # Risk/reward target
    sl_buffer_mult: float = 1.0  # Stop loss buffer multiplier
    lookback_days: int = 90  # Lookback for FVG detection

    # FVG state features
    max_active_fvgs: int = 3  # Number of active FVGs to track
    fvg_ttl_days: int = 20  # FVG time-to-live in days

# =============================================================================
# FVG DETECTION LOGIC (From FVG_Daily.py)
# =============================================================================

@dataclass
class FVG:
    symbol: str
    max_val: float
    min_val: float
    isbull: bool
    t_index: int
    t_time: pd.Timestamp

    def gap_height(self) -> float:
        return abs(self.max_val - self.min_val)

    def mid_price(self) -> float:
        return (self.max_val + self.min_val) / 2


def detect_fvg_for_symbol(
    df: pd.DataFrame,
    threshold_per: float = 0.0,
    auto: bool = False
) -> List[FVG]:
    """
    FVG detection logic (translated from Pine Script).
    Detects bullish and bearish Fair Value Gaps.
    """
    df = df.sort_values("date").reset_index(drop=True)

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    n = len(df)
    if n < 3:
        return []

    # Calculate threshold
    if auto:
        rel_range = (high - low) / np.where(low == 0, np.nan, low)
        cum = np.nancumsum(rel_range)
        idx = np.arange(1, n + 1, dtype=float)
        threshold = cum / idx
    else:
        threshold = np.full(n, threshold_per / 100.0)

    # Shift arrays for vectorized comparison
    high_2 = np.concatenate(([np.nan, np.nan], high[:-2]))
    low_2 = np.concatenate(([np.nan, np.nan], low[:-2]))
    close_1 = np.concatenate(([np.nan], close[:-1]))

    # Bullish FVG: Low[i] > High[i-2] and Close[i-1] > High[i-2]
    cond_bull = (
        (low > high_2) &
        (close_1 > high_2) &
        ((low - high_2) / np.maximum(high_2, 1e-8) > threshold)
    )

    # Bearish FVG: High[i] < Low[i-2] and Close[i-1] < Low[i-2]
    cond_bear = (
        (high < low_2) &
        (close_1 < low_2) &
        ((low_2 - high) / np.maximum(high, 1e-8) > threshold)
    )

    fvg_records: List[FVG] = []
    last_t_index: Optional[int] = None

    for i in range(n):
        if not cond_bull[i] and not cond_bear[i]:
            continue

        if cond_bull[i]:
            max_val = low[i]
            min_val = high[i - 2]
            isbull = True
        else:
            max_val = high[i]
            min_val = low[i - 2]
            isbull = False

        # Avoid duplicates at same index
        if last_t_index is not None and last_t_index == i:
            continue

        fvg_records.append(FVG(
            symbol=str(df.loc[i, "symbol"]) if "symbol" in df.columns else "UNK",
            max_val=max_val,
            min_val=min_val,
            isbull=isbull,
            t_index=i,
            t_time=pd.Timestamp(df.loc[i, "date"])
        ))
        last_t_index = i

    return fvg_records


def get_fvg_features(
    df: pd.DataFrame,
    current_idx: int,
    config: FVGConfig
) -> Tuple[np.ndarray, List[FVG]]:
    """
    Extract FVG features for the current time step.
    Returns feature vector and list of active FVGs.
    """
    # Detect all FVGs up to current point
    lookback_start = max(0, current_idx - config.lookback_days)
    df_window = df.iloc[lookback_start:current_idx+1].copy()
    df_window = df_window.reset_index(drop=True)

    # Adjust indices
    all_fvgs = detect_fvg_for_symbol(
        df_window, 
        threshold_per=config.threshold_per,
        auto=config.auto_threshold
    )

    # Filter active FVGs (not too old)
    active_fvgs = []
    current_date = df.iloc[current_idx]['date']

    for fvg in all_fvgs:
        # Check if FVG is still valid (not expired)
        days_since = (current_date - fvg.t_time).days
        if days_since <= config.fvg_ttl_days and fvg.t_index < len(df_window) - 1:
            active_fvgs.append(fvg)

    # Sort by recency (most recent first)
    active_fvgs.sort(key=lambda x: x.t_index, reverse=True)
    active_fvgs = active_fvgs[:config.max_active_fvgs]

    # Build feature vector
    features = []
    current_price = df.iloc[current_idx]['close']

    for i in range(config.max_active_fvgs):
        if i < len(active_fvgs):
            fvg = active_fvgs[i]
            # Normalize features
            gap_height_norm = fvg.gap_height() / current_price
            distance_to_mid = (current_price - fvg.mid_price()) / current_price
            age_days = (current_date - fvg.t_time).days / config.fvg_ttl_days
            direction = 1.0 if fvg.isbull else -1.0

            features.extend([
                gap_height_norm,
                distance_to_mid,
                age_days,
                direction,
                1.0  # Active flag
            ])
        else:
            # Pad with zeros if no active FVG
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32), active_fvgs


def calculate_fvg_reward(
    action: int,
    active_fvgs: List[FVG],
    current_price: float,
    next_price: float,
    position: float
) -> float:
    """
    Calculate reward based on FVG alignment.
    Rewards actions that align with FVG signals.
    """
    if not active_fvgs:
        return 0.0

    # Get most recent FVG
    recent_fvg = active_fvgs[0]
    price_change = (next_price - current_price) / current_price

    # Determine target position based on FVG
    if recent_fvg.isbull:
        fvg_signal = 1.0  # Long signal
    else:
        fvg_signal = -1.0  # Short signal

    # Convert action to position
    action_pos = [0.0, 1.0, -1.0][action]

    # Reward alignment with FVG
    alignment = action_pos * fvg_signal

    # Reward profitable moves in FVG direction
    profit_alignment = position * price_change * 100  # Scale up

    # Bonus for entering in FVG direction
    entry_bonus = 0.0
    if action_pos != 0 and position == 0:  # Opening position
        entry_bonus = alignment * 0.1

    return profit_alignment + entry_bonus + (alignment * 0.05)


# =============================================================================
# TECHNICAL FEATURES (From Train.py)
# =============================================================================

TECHNICAL_FEATURES = [
    'ret_1d', 'ret_5d', 'vol_10d', 'vol_20d',
    'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20'
]


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators."""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Returns
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)

    # Volatility
    df['vol_10d'] = df['ret_1d'].rolling(10, min_periods=5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20, min_periods=10).std()

    # Distance from SMAs
    for win in [20, 50, 200]:
        sma = df['close'].rolling(win, min_periods=win//2).mean()
        df[f'dist_sma_{win}'] = (df['close'] - sma) / sma

    # Volume z-score
    vol_ma = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z_20'] = (df['volume'] - vol_ma) / (vol_std + 1e-8)

    return df


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(
    conn_str: str, 
    symbol: str = "SPY",
    start_date: str = "2005-01-01",
    end_date: str = "2018-12-31"
) -> pd.DataFrame:
    """Load price data from SQL database."""
    engine = sa.create_engine(conn_str)

    # Adjust query based on your table structure
    query = f"""
    SELECT 
        symbol, 
        TradeDate as date, 
        [open], 
        high, 
        low, 
        [close], 
        volume
    FROM AI_ETF_Prices
    WHERE symbol = '{symbol}' 
      AND TradeDate >= '{start_date}' 
      AND TradeDate <= '{end_date}'
    ORDER BY TradeDate
    """

    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])

    print(f"✅ Loaded {len(df)} rows for {symbol}")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with all features."""
    df = compute_technical_features(df)

    # Keep necessary columns
    cols = ['date', 'open', 'high', 'low', 'close', 'volume'] + TECHNICAL_FEATURES
    df = df[cols].dropna().reset_index(drop=True)

    print(f"✅ Prepared {len(df)} rows with {len(TECHNICAL_FEATURES)} technical features")
    return df


# =============================================================================
# FVG-DQN ENVIRONMENT
# =============================================================================

class FVG_DQN_Env(gym.Env):
    """
    DQN Environment with FVG integration.
    State: Technical features + FVG features + Position + Equity
    """

    def __init__(
        self, 
        df: pd.DataFrame,
        fvg_config: Optional[FVGConfig] = None,
        use_fvg_reward: bool = True,
        transaction_cost: float = 0.0005
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.fvg_config = fvg_config or FVGConfig()
        self.use_fvg_reward = use_fvg_reward
        self.transaction_cost = transaction_cost

        # Data arrays
        self.tech_features = self.df[TECHNICAL_FEATURES].fillna(0).values.astype(np.float32)
        self.prices = self.df['close'].values.astype(np.float32)
        self.highs = self.df['high'].values.astype(np.float32)
        self.lows = self.df['low'].values.astype(np.float32)
        self.dates = self.df['date'].values

        self.n_steps = len(self.prices)

        # Normalize technical features
        self.tech_mean = self.tech_features.mean(axis=0, keepdims=True)
        self.tech_std = self.tech_features.std(axis=0, keepdims=True) + 1e-8
        self.tech_features = (self.tech_features - self.tech_mean) / self.tech_std

        # Calculate FVG feature dimension
        self.fvg_feature_dim = self.fvg_config.max_active_fvgs * 5  # 5 features per FVG

        # Observation space: Tech features + FVG features + Position + Equity
        obs_dim = len(TECHNICAL_FEATURES) + self.fvg_feature_dim + 2
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

        # Action space: 0=FLAT, 1=LONG, 2=SHORT
        self.action_space = spaces.Discrete(3)

        # State
        self.current_idx = 0
        self.position = 0.0
        self.equity = 1.0
        self.last_price = 0.0
        self.active_fvgs: List[FVG] = []

        print(f"✅ FVG-DQN Env initialized:")
        print(f"   - Steps: {self.n_steps}")
        print(f"   - Tech features: {len(TECHNICAL_FEATURES)}")
        print(f"   - FVG features: {self.fvg_feature_dim}")
        print(f"   - Total obs dim: {obs_dim}")

        self.reset()

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        # Technical features
        tech = self.tech_features[self.current_idx].copy()

        # FVG features
        fvg_feats, self.active_fvgs = get_fvg_features(
            self.df, 
            self.current_idx, 
            self.fvg_config
        )

        # State features
        state = np.array([self.position, self.equity], dtype=np.float32)

        # Concatenate
        obs = np.concatenate([tech, fvg_feats, state])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start after enough history for indicators
        self.current_idx = max(200, self.fvg_config.lookback_days)
        self.position = 0.0
        self.equity = 1.0
        self.last_price = self.prices[self.current_idx - 1]
        self.active_fvgs = []

        return self._get_observation(), {}

    def step(self, action):
        # Execute action
        target_pos = [0.0, 1.0, -1.0][action]

        # Transaction cost
        pos_change = abs(target_pos - self.position)
        cost = pos_change * self.equity * self.transaction_cost

        # Get current and next price
        current_price = self.prices[self.current_idx]

        # Move to next step
        self.current_idx += 1
        next_price = self.prices[self.current_idx]

        # Calculate return
        price_ret = (next_price - current_price) / current_price

        # PnL
        pnl = self.position * price_ret * self.equity
        self.equity += pnl - cost

        # FVG-based reward
        if self.use_fvg_reward:
            fvg_reward = calculate_fvg_reward(
                action, 
                self.active_fvgs,
                current_price,
                next_price,
                self.position
            )
            reward = pnl * 100 + fvg_reward - cost * 10  # Scale rewards
        else:
            reward = pnl * 100 - cost * 10

        # Update position
        self.position = target_pos

        # Check termination
        done = self.equity <= 0.1 or self.current_idx >= self.n_steps - 1
        truncated = False

        obs = self._get_observation()
        info = {
            'equity': self.equity,
            'position': self.position,
            'active_fvgs': len(self.active_fvgs)
        }

        return obs, float(reward), done, truncated, info


# =============================================================================
# TRAINING
# =============================================================================

def train_fvg_dqn(
    conn_str: str,
    symbol: str = "SPY",
    total_timesteps: int = 250_000,
    fvg_config: Optional[FVGConfig] = None,
    model_name: str = "fvg_dqn"
):
    """Train FVG-DQN model."""

    print("\n" + "="*60)
    print("🚀 FVG-DQN TRAINING STARTED")
    print("="*60)

    # Load data
    print(f"\n📊 Loading data for {symbol}...")
    df = load_data(conn_str, symbol)
    df = prepare_data(df)

    # Create environment
    print(f"\n🔨 Creating FVG-DQN environment...")
    fvg_config = fvg_config or FVGConfig()

    def make_env():
        return FVG_DQN_Env(df, fvg_config, use_fvg_reward=True)

    env = DummyVecEnv([make_env])

    # Create model
    print(f"\n🎯 Training for {total_timesteps:,} steps...")
    print(f"   This may take 15-30 minutes depending on hardware...")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        gamma=0.99,
        target_update_interval=1_000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=f"./fvg_dqn_logs/{symbol}/"
    )

    # Train
    model.learn(total_timesteps=total_timesteps)

    # Save model
    model_filename = f"{model_name}_{symbol.lower()}.zip"
    model.save(model_filename)

    # Save normalization stats
    env_instance = make_env()
    np.save(f"{model_name}_{symbol.lower()}_tech_mean.npy", env_instance.tech_mean.ravel())
    np.save(f"{model_name}_{symbol.lower()}_tech_std.npy", env_instance.tech_std.ravel())

    print(f"\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"💾 Model saved: {model_filename}")
    print(f"📊 Stats saved: {model_name}_{symbol.lower()}_tech_mean.npy")
    print(f"📊 Stats saved: {model_name}_{symbol.lower()}_tech_std.npy")
    print(f"\n🚀 Ready for backtesting!")
    print(f"   Use the zip file with your backtest framework.")

    return model, env_instance


# =============================================================================
# BACKTEST INTERFACE
# =============================================================================

class FVG_DQN_Backtester:
    """Helper class for backtesting the trained FVG-DQN model."""

    def __init__(
        self,
        model_path: str,
        tech_mean_path: str,
        tech_std_path: str,
        fvg_config: Optional[FVGConfig] = None
    ):
        self.model = DQN.load(model_path)
        self.tech_mean = np.load(tech_mean_path)
        self.tech_std = np.load(tech_std_path)
        self.fvg_config = fvg_config or FVGConfig()

    def predict(self, obs: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
        """Get action from model."""
        action, _states = self.model.predict(obs, deterministic=True)
        return int(action), _states


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Database connection - adjust to your setup
    SQL_CONN = (
        "mssql+pyodbc://localhost/Stock?"
        "driver=ODBC+Driver+17+for+SQL+Server&"
        "trusted_connection=yes"
    )

    # Alternative connection string format
    # SQL_CONN = (
    #     "mssql+pyodbc:///?odbc_connect=" + 
    #     urllib.parse.quote_plus(
    #         "DRIVER={ODBC Driver 17 for SQL Server};"
    #         "SERVER=BEELINK;"
    #         "DATABASE=Stock;"
    #         "Trusted_Connection=yes;"
    #     )
    # )

    # FVG Configuration
    fvg_config = FVGConfig(
        threshold_per=0.5,      # 0.5% FVG threshold
        rr_target=2.0,          # 1:2 risk/reward
        max_active_fvgs=3,      # Track 3 most recent FVGs
        fvg_ttl_days=20         # FVGs valid for 20 days
    )

    try:
        # Train model
        model, env = train_fvg_dqn(
            conn_str=SQL_CONN,
            symbol="SPY",
            total_timesteps=250_000,
            fvg_config=fvg_config,
            model_name="fvg_dqn"
        )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()