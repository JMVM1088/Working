"""
RL skeleton for daily SPY/QQQ trading using SQL Server OHLCV data.

Requirements:
    pip install pandas numpy sqlalchemy pyodbc gymnasium stable-baselines3

Assumptions:
    - SQL Server table 'daily_prices' with columns:
        symbol (e.g. 'SPY', 'QQQ')
        time (DATE or DATETIME)
        open, high, low, close, volume (floats)
    - Daily data with no gaps per symbol (or at least tradable days only).pi
"""

import numpy as np
import pandas as pd
import sqlalchemy as sa
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# ----------------------------
# 1. Data loading from SQL Server
# ----------------------------

def load_ohlcv_from_sql(
    conn_str: str,
    table: str = "AI_ETF_prices",
    symbols=("SPY"),
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    engine = sa.create_engine(conn_str)
    sym_list = ",".join(f"'{s}'" for s in symbols)

    date_filter = ""
    if start_date:
        date_filter += f" AND [Time] >= '{start_date}'"
    if end_date:
        date_filter += f" AND [Time] <= '{end_date}'"

    query = f"""
    SELECT symbol, [Time], [open], high, low, [close], volume
    FROM {table}
    WHERE symbol IN ({sym_list}) {date_filter}
    ORDER BY [Time], symbol;
    """
    df = pd.read_sql(query, engine)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["Time", "symbol"]).reset_index(drop=True)
    return df


# ----------------------------
# 2. Feature engineering for state
# ----------------------------

def add_basic_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, interpretable features for RL state.
    panel: multi-index (time, symbol) or flat with those columns.
    """
    df = panel.copy()

    # Ensure multi-index
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(["Time", "symbol"]).sort_index()

    # Daily returns
    df["ret_1d"] = df.groupby("symbol")["close"].pct_change()
    df["ret_5d"] = df.groupby("symbol")["close"].pct_change(5)
    df["ret_10d"] = df.groupby("symbol")["close"].pct_change(10)

    # Rolling volatility
    df["vol_10d"] = (
        df.groupby("symbol")["ret_1d"].rolling(10).std().reset_index(level=0, drop=True)
    )
    df["vol_20d"] = (
        df.groupby("symbol")["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    )

    # Trend features: distance from 20/50/200-day SMA
    for win in (20, 50, 200):
        sma = (
            df.groupby("symbol")["close"]
            .rolling(win)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"sma_{win}"] = sma
        df[f"dist_sma_{win}"] = (df["close"] - sma) / sma

    # Volume features
    df["vol_z_20"] = (
        df.groupby("symbol")["volume"]
        .transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std())
    )

    # Market context: SPY vs QQQ
    pivot_close = df["close"].unstack("symbol")
    for sym in pivot_close.columns:
        df[f"{sym}_ret_1d"] = pivot_close[sym].pct_change().reindex(
            pivot_close.index
        ).repeat(len(pivot_close.columns))
    df = df.sort_index()

    # Drop initial NaNs
    df = df.dropna()
    return df


# ----------------------------
# 3. Gym-style environment
# ----------------------------

class DailyTradingEnv(gym.Env):
    """
    Single-asset RL environment (trade SPY or QQQ) using daily OHLCV + features.

    - Observation: feature vector at time t.
    - Action space: {0: flat, 1: long, 2: short}.
    - Reward: change in equity net of simple transaction cost.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str = "SPY",
        trading_cost_bps: float = 1.0,
        max_position: float = 1.0,
        init_cash: float = 1.0,
    ):
        super().__init__()
        self.symbol = symbol
        self.trading_cost_bps = trading_cost_bps / 10000.0
        self.max_position = max_position  # position in [-1,1] scaled via actions
        self.init_cash = init_cash

        # Filter to single symbol timeseries
        self.df = df.xs(symbol, level="symbol").sort_index()
        self.dates = self.df.index.get_level_values("time").unique()
        self.n_steps = len(self.dates)

        # Build feature matrix and price series
        feature_cols = [
            c
            for c in self.df.columns
            if c
            not in ["open", "high", "low", "close", "volume"]
        ]
        self.feature_cols = feature_cols
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.prices = self.df["close"].values.astype(np.float32)

        # Normalize features (simple scaling)
        self.feat_mean = self.features.mean(axis=0)
        self.feat_std = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.feat_mean) / self.feat_std

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(feature_cols) + 2,), dtype=np.float32
        )
        # 0: flat, 1: long, 2: short
        self.action_space = spaces.Discrete(3)

        self.reset(seed=None)

    def _get_obs(self):
        # Current feature vector + position + normalized equity
        feat = self.features[self.t].copy()
        obs = np.concatenate(
            [
                feat,
                np.array(
                    [self.position, self.equity / self.init_cash],
                    dtype=np.float32,
                ),
            ]
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start somewhere after initial warmup
        self.t = 1
        self.position = 0.0  # -1, 0, 1
        self.equity = self.init_cash
        self.last_price = self.prices[self.t - 1]
        info = {}
        return self._get_obs(), info

    def step(self, action: int):
        assert self.action_space.contains(action)

        # Map action to target position
        if action == 0:
            target_pos = 0.0
        elif action == 1:
            target_pos = +self.max_position
        else:
            target_pos = -self.max_position

        # Transaction cost on position change
        pos_change = target_pos - self.position
        trade_cost = abs(pos_change) * self.equity * self.trading_cost_bps

        # Price move
        price = self.prices[self.t]
        ret = (price - self.last_price) / self.last_price

        # PnL from position
        pnl = self.position * ret * self.equity

        # Update equity
        self.equity += pnl - trade_cost

        # Update state
        self.position = target_pos
        self.last_price = price

        # Reward: equity change (can also scale or add risk penalty)
        reward = pnl - trade_cost

        # Advance time
        self.t += 1
        terminated = False
        truncated = False

        if self.t >= self.n_steps - 1:
            truncated = True

        if self.equity <= 0.0:
            terminated = True

        obs = self._get_obs()
        info = {
            "equity": self.equity,
            "position": self.position,
            "price": price,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            f"t={self.t}, price={self.last_price:.2f}, equity={self.equity:.4f}, pos={self.position}"
        )


# ----------------------------
# 4. Utility to create train/test envs for SPY/QQQ
# ----------------------------

def prepare_data_and_envs(
    conn_str: str,
    table: str = "AI_ETF_Prices",
    start_date: str = "2000-01-01",
    split_date: str = "2018-01-01",
    end_date: str = "2025-01-01",
):
    # Load data
    raw = load_ohlcv_from_sql(
        conn_str=conn_str,
        table=table,
        symbols=("SPY", "QQQ"),
        start_date=start_date,
        end_date=end_date,
    )

    # Build multi-index panel
    panel = raw.set_index(["Time", "symbol"]).sort_index()

    # Add features
    panel_feat = add_basic_features(panel)

    # Train / test split by date
    train_panel = panel_feat.loc[
        (panel_feat.index.get_level_values("time") < pd.to_datetime(split_date))
    ]
    test_panel = panel_feat.loc[
        (panel_feat.index.get_level_values("time") >= pd.to_datetime(split_date))
    ]

    # Create envs for SPY and QQQ separately; you can also merge or randomize symbol per episode.
    train_env_spy = DailyTradingEnv(train_panel, symbol="SPY")
    test_env_spy = DailyTradingEnv(test_panel, symbol="SPY")

    train_env_qqq = DailyTradingEnv(train_panel, symbol="QQQ")
    test_env_qqq = DailyTradingEnv(test_panel, symbol="QQQ")

    return (train_env_spy, test_env_spy, train_env_qqq, test_env_qqq)


# ----------------------------
# 5. Training loop with DQN
# ----------------------------

def train_dqn_on_env(env: gym.Env, timesteps: int = 200_000) -> DQN:
    """
    Wrap env in DummyVecEnv and train a basic DQN agent.
    """
    vec_env = DummyVecEnv([lambda: env])
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        train_freq=1,
        verbose=1,
    )
    model.learn(total_timesteps=timesteps)
    return model


def evaluate_model(model: DQN, env: gym.Env, n_episodes: int = 1):
    """
    Simple test evaluation: run episodes and log final equity.
    """
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        print(f"[Episode {ep}] Final equity: {info['equity']:.4f}")


# ----------------------------
# 6. Main entry point
# ----------------------------

if __name__ == "__main__":
    # Example SQL Server connection string; adjust to your setup.
    # For SQL Server + pyodbc:
    # DRIVER={ODBC Driver 17 for SQL Server};SERVER=your_server;DATABASE=your_db;UID=your_user;PWD=your_pwd
    sql_conn_str = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    
    # sql_conn_str = (
    #     r'DRIVER={ODBC Driver 17 for SQL Server};'
    #     r'SERVER=BEELINK;'  # Replace with your server name
    #     r'DATABASE=Stock;' # Replace with your database name
    #     r'Trusted_Connection=yes;'
    # )


    train_env_spy, test_env_spy, train_env_qqq, test_env_qqq = prepare_data_and_envs(
        conn_str=sql_conn_str,
        table="AI_ETF_prices",
        start_date="2000-01-01",
        split_date="2018-01-01",
        end_date="2025-01-01",
    )

    # Train on SPY as an example
    print("Training DQN on SPY...")
    dqn_spy = train_dqn_on_env(train_env_spy, timesteps=100_000)

    print("Evaluating DQN on SPY test period...")
    evaluate_model(dqn_spy, test_env_spy, n_episodes=1)

    # Optionally: train a separate model on QQQ or reuse the same model
    # print("Training DQN on QQQ...")
    # dqn_qqq = train_dqn_on_env(train_env_qqq, timesteps=100_000)
    # print("Evaluating DQN on QQQ test period...")
    # evaluate_model(dqn_qqq, test_env_qqq, n_episodes=1)
