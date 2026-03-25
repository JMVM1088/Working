
"""
🚀 FVG-DQN HYBRID BACKTEST SYSTEM
✅ Integrates Fair Value Gap detection with Deep Q-Network trading
✅ Uses FVG for entry/exit levels and DQN for position sizing
✅ Enhanced risk management with FVG-based stop losses
✅ Multi-timeframe analysis: FVG pattern + DQN trend confirmation
"""

import pandas as pd
import sqlalchemy as sa
import numpy as np
from stable_baselines3 import DQN
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import torch
import urllib
import warnings
warnings.filterwarnings('ignore')

print("🎯 FVG-DQN HYBRID BACKTEST SYSTEM\n")

# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class FVG:
    """Fair Value Gap data structure"""
    symbol: str
    max: float
    min: float
    isbull: bool
    t_index: int
    t_time: pd.Timestamp

class Config:
    """Configuration settings"""
    # Data settings
    SYMBOL = "SPY"
    BACKTEST_START = "2025-01-01"
    BACKTEST_END = "2025-12-31"

    # Model settings
    MODEL_PATH = "dqn_spy_improved"

    # FVG Settings
    FVG_THRESHOLD_PER = 0.5  # Minimum gap size as percentage
    FVG_AUTO_THRESHOLD = False  # Use dynamic threshold
    FVG_LOOKBACK_DAYS = 90  # Days to look back for FVG detection

    # Risk Management
    RR_TARGET = 2.0  # Risk:Reward ratio
    SL_BUFFER_MULT = 1.0  # Stop loss buffer multiplier
    POSITION_SCALE = 1.2  # DQN confidence scaler
    MIN_CONFIDENCE_THRESHOLD = 0.05
    MAX_POSITION_SIZE = 1.0

    # Volatility scaling
    VOLATILITY_SCALING = True
    MIN_VOLATILITY = 0.008
    MAX_VOLATILITY = 0.030

    # Hybrid settings
    FVG_CONFIRMATION_BOOST = 0.2  # Boost to confidence when FVG aligns
    FVG_CONTRARIAN_PENALTY = 0.3  # Penalty when FVG contradicts DQN

    # Database
    SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

# ============================================================
# 2. DATABASE CONNECTION
# ============================================================

def get_engine(conn_str: str = None):
    """Create SQL Server engine"""
    if conn_str is None:
        conn_str = Config.SQL_CONN
    return sa.create_engine(conn_str)

def load_price_data(conn_str: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load OHLCV data from SQL Server"""
    print(f"📊 Loading {symbol} data ({start_date} to {end_date})...")

    engine = get_engine(conn_str)

    # Extended lookback for FVG detection
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    extended_start = (start_dt - timedelta(days=Config.FVG_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    query = f"""
    SELECT TradeDate as date, [open], high, low, [close], volume
    FROM AI_ETF_Prices
    WHERE symbol = '{symbol}' 
        AND TradeDate >= '{extended_start}' 
        AND TradeDate <= '{end_date}'
    ORDER BY TradeDate
    """

    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    df['symbol'] = symbol

    print(f"✅ Loaded {len(df)} days (extended from {extended_start})\n")
    return df

# ============================================================
# 3. FVG DETECTION ENGINE
# ============================================================

def detect_fvg_for_symbol(
    df: pd.DataFrame,
    threshold_per: float = Config.FVG_THRESHOLD_PER,
    auto: bool = Config.FVG_AUTO_THRESHOLD
) -> List[FVG]:
    """
    Detect Fair Value Gaps in price data.
    Bullish FVG: Low[i] > High[i-2] (gap up)
    Bearish FVG: High[i] < Low[i-2] (gap down)
    """
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 3:
        return []

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

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

    # Bullish FVG conditions
    cond_bull = (
        (low > high_2) &
        (close_1 > high_2) &
        ((low - high_2) / high_2 > threshold)
    )

    # Bearish FVG conditions
    cond_bear = (
        (high < low_2) &
        (close_1 < low_2) &
        ((low_2 - high) / high > threshold)
    )

    fvg_records = []
    for i in range(n):
        if cond_bull[i]:
            fvg_records.append(FVG(
                symbol=str(df.loc[i, "symbol"]),
                max=float(low[i]),
                min=float(high[i-2]),
                isbull=True,
                t_index=i,
                t_time=df.loc[i, "date"]
            ))
        elif cond_bear[i]:
            fvg_records.append(FVG(
                symbol=str(df.loc[i, "symbol"]),
                max=float(high[i]),
                min=float(low[i-2]),
                isbull=False,
                t_index=i,
                t_time=df.loc[i, "date"]
            ))

    return fvg_records

def build_trade_from_fvg(
    df: pd.DataFrame,
    fvg: FVG,
    rr_target: float = Config.RR_TARGET,
    sl_buffer_mult: float = Config.SL_BUFFER_MULT
) -> Dict[str, Any]:
    """Build trade parameters from FVG pattern"""

    entry_idx = fvg.t_index
    if entry_idx >= len(df):
        return {}

    entry_price = float(df["close"].iloc[entry_idx])
    direction = "long" if fvg.isbull else "short"

    # Calculate gap height for SL/TP
    gap_height = abs(fvg.max - fvg.min)
    if gap_height <= 0:
        gap_height = entry_price * 0.01

    gap_height *= sl_buffer_mult

    if direction == "long":
        stop_loss = entry_price - gap_height
        take_profit = entry_price + rr_target * gap_height
    else:
        stop_loss = entry_price + gap_height
        take_profit = entry_price - rr_target * gap_height

    return {
        "direction": direction,
        "entry_price": entry_price,
        "fvg_max": fvg.max,
        "fvg_min": fvg.min,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "gap_height": gap_height,
        "creation_time": fvg.t_time,
        "creation_index": fvg.t_index
    }

def get_active_fvgs(df: pd.DataFrame, current_idx: int, 
                    lookback: int = 20) -> Tuple[Optional[FVG], Optional[FVG]]:
    """
    Get most recent bullish and bearish FVGs that are still active.
    Returns: (bullish_fvg, bearish_fvg)
    """
    # Detect all FVGs up to current point
    df_slice = df.iloc[:current_idx+1].copy()
    fvgs = detect_fvg_for_symbol(df_slice)

    bullish_fvg = None
    bearish_fvg = None

    # Find most recent active FVGs within lookback
    for fvg in reversed(fvgs):
        if current_idx - fvg.t_index > lookback:
            break

        if fvg.isbull and bullish_fvg is None:
            bullish_fvg = fvg
        elif not fvg.isbull and bearish_fvg is None:
            bearish_fvg = fvg

        if bullish_fvg is not None and bearish_fvg is not None:
            break

    return bullish_fvg, bearish_fvg

# ============================================================
# 4. TECHNICAL INDICATORS (from backtest_improved)
# ============================================================

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

    return {
        'macd_line': macd_line / prices * 100,
        'macd_signal': macd_signal / prices * 100,
        'macd_hist': macd_hist / prices * 100
    }

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
    return (obv / (obv_ma + 1e-8)).clip(-5, 5) / 5

# ============================================================
# 5. ENHANCED FEATURE ENGINEERING
# ============================================================

BASE_FEATURES = [
    'ret_1d', 'ret_5d', 'vol_10d', 'vol_20d', 
    'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'vol_z_20'
]

ADVANCED_FEATURES = [
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'atr_14', 'vol_ratio', 'vol_trend',
    'adx_14', 'bb_position',
    'range_norm', 'ret_skew', 'obv_ratio'
]

FVG_FEATURES = [
    'fvg_bull_active', 'fvg_bear_active',
    'fvg_bull_age', 'fvg_bear_age',
    'fvg_bull_gap', 'fvg_bear_gap',
    'fvg_signal_aligned'
]

ALL_FEATURES = BASE_FEATURES + ADVANCED_FEATURES + FVG_FEATURES

def compute_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features including FVG indicators"""
    print("🔧 Computing features (20 technical + 7 FVG)...")

    df = raw_df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    if len(df) < 250:
        raise ValueError(f"Need at least 250 days, got {len(df)}")

    # Basic returns
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)

    # Volatility
    df['vol_10d'] = df['ret_1d'].rolling(10, min_periods=5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20, min_periods=10).std()

    # Moving averages
    for win in [20, 50, 200]:
        sma = df['close'].rolling(win, min_periods=win//2).mean()
        df[f'dist_sma_{win}'] = (df['close'] - sma) / (sma + 1e-8)

    # Volume
    vol_ma = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z_20'] = (df['volume'] - vol_ma) / (vol_std + 1e-8)

    # Momentum
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    macd_dict = calculate_macd(df['close'])
    df['macd_line'] = macd_dict['macd_line']
    df['macd_signal'] = macd_dict['macd_signal']
    df['macd_hist'] = macd_dict['macd_hist']

    # Volatility indicators
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    df['vol_ratio'] = df['vol_20d'] / (df['vol_10d'] + 1e-8)
    df['vol_trend'] = df['vol_20d'].pct_change(5)

    # Trend
    df['adx_14'] = calculate_adx(df['high'], df['low'], df['close'], 14)
    df['bb_position'] = calculate_bb_position(df['close'], 20)

    # Price action
    df['high_20'] = df['close'].rolling(20).max()
    df['low_20'] = df['close'].rolling(20).min()
    df['range_norm'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
    df['ret_skew'] = df['ret_1d'].rolling(20, min_periods=10).skew()
    df['obv_ratio'] = calculate_obv(df['close'], df['volume'])

    # FVG Features - calculated for each bar
    print("   Computing FVG features...")
    fvg_bull_active = []
    fvg_bear_active = []
    fvg_bull_age = []
    fvg_bear_age = []
    fvg_bull_gap = []
    fvg_bear_gap = []

    for i in range(len(df)):
        bull_fvg, bear_fvg = get_active_fvgs(df, i, lookback=20)

        fvg_bull_active.append(1.0 if bull_fvg else 0.0)
        fvg_bear_active.append(1.0 if bear_fvg else 0.0)

        fvg_bull_age.append(i - bull_fvg.t_index if bull_fvg else -1)
        fvg_bear_age.append(i - bear_fvg.t_index if bear_fvg else -1)

        fvg_bull_gap.append(abs(bull_fvg.max - bull_fvg.min) / df['close'].iloc[i] if bull_fvg else 0.0)
        fvg_bear_gap.append(abs(bear_fvg.max - bear_fvg.min) / df['close'].iloc[i] if bear_fvg else 0.0)

    df['fvg_bull_active'] = fvg_bull_active
    df['fvg_bear_active'] = fvg_bear_active
    df['fvg_bull_age'] = np.array(fvg_bull_age) / 20.0  # Normalize
    df['fvg_bear_age'] = np.array(fvg_bear_age) / 20.0
    df['fvg_bull_gap'] = fvg_bull_gap
    df['fvg_bear_gap'] = fvg_bear_gap
    df['fvg_signal_aligned'] = df['fvg_bull_active'] - df['fvg_bear_active']
    # Forward returns for analysis
    df['ret_5d_fwd'] = df['close'].shift(-5) / df['close'] - 1.0
    df['ret_60d_fwd'] = df['close'].shift(-60) / df['close'] - 1.0

    # Select and clean
    result = df[ALL_FEATURES + ['date', 'close', 'high', 'low', 'volume', 
                                'ret_5d_fwd', 'ret_60d_fwd', 'symbol']].copy()
    result = result.fillna(method='bfill').fillna(method='ffill').dropna()
    result = result.reset_index(drop=True)

    print(f"✅ {len(result)} days with {len(ALL_FEATURES)} features\n")
    return result

# ============================================================
# 6. HYBRID SIGNAL GENERATION
# ============================================================

def calculate_hybrid_confidence(
    dqn_confidence: float,
    dqn_action: int,
    fvg_bull_active: float,
    fvg_bear_active: float,
    position: float
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Combine DQN signal with FVG confirmation.
    Returns: (adjusted_confidence, signal_type, trade_params)
    """

    # Base signal from DQN
    if dqn_action == 0:
        base_signal = 'FLAT'
        base_direction = None
    else:
        base_signal = 'LONG'
        base_direction = 'long'

    # FVG alignment check
    fvg_bullish = fvg_bull_active > 0.5
    fvg_bearish = fvg_bear_active > 0.5

    trade_params = {
        'dqn_raw_confidence': dqn_confidence,
        'fvg_bullish': fvg_bullish,
        'fvg_bearish': fvg_bearish,
        'alignment': 'neutral'
    }

    # Adjust confidence based on FVG alignment
    adjusted_confidence = dqn_confidence

    if base_signal == 'LONG':
        if fvg_bullish:
            # Strong alignment: DQN wants long + Bullish FVG present
            adjusted_confidence = min(1.0, dqn_confidence + Config.FVG_CONFIRMATION_BOOST)
            trade_params['alignment'] = 'strong_long'
        elif fvg_bearish:
            # Contrarian: DQN wants long but Bearish FVG present
            adjusted_confidence = max(0.0, dqn_confidence - Config.FVG_CONTRARIAN_PENALTY)
            trade_params['alignment'] = 'weak_long'

    # For short positions (if implemented in future)
    # elif base_signal == 'SHORT':
    #     if fvg_bearish:
    #         adjusted_confidence = min(1.0, dqn_confidence + Config.FVG_CONFIRMATION_BOOST)
    #     elif fvg_bullish:
    #         adjusted_confidence = max(0.0, dqn_confidence - Config.FVG_CONTRARIAN_PENALTY)

    return adjusted_confidence, base_signal, trade_params

# ============================================================
# 7. BACKTEST ENGINE
# ============================================================

def run_fvg_dqn_backtest(
    model_path: str, 
    df: pd.DataFrame, 
    symbol: str = Config.SYMBOL
) -> pd.DataFrame:
    """Run hybrid FVG-DQN backtest"""

    print(f"🔮 Loading DQN model: {model_path}")

    # Load normalization stats
    try:
        feat_mean = np.load("feat_mean_longonly.npy")
        feat_std = np.load("feat_std_longonly.npy")
        # Extend stats for FVG features (7 new features)
        if len(feat_mean) < len(ALL_FEATURES):
            feat_mean = np.concatenate([feat_mean, np.zeros(7)])
            feat_std = np.concatenate([feat_std, np.ones(7)])
        print(f"✅ Loaded normalization stats ({len(feat_mean)} features)")
    except Exception as e:
        print(f"⚠️  Using default normalization: {e}")
        feat_mean = np.zeros(len(ALL_FEATURES))
        feat_std = np.ones(len(ALL_FEATURES))

    # Load model
    try:
        model = DQN.load(model_path)
        print(f"✅ Loaded DQN model\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

    # Backtest state
    position = 0.0
    equity = 1.0
    last_price = df['close'].iloc[0]
    max_equity = 1.0

    predictions = []
    active_fvg_trade = None  # Track FVG-based trade

    # Find start index (after lookback period)
    start_idx = len(df) - len(df[df['date'] >= Config.BACKTEST_START])
    if start_idx < 50:
        start_idx = 50

    print(f"🔮 Running backtest from index {start_idx}...")
    print(f"   Hybrid mode: DQN + FVG confirmation\n")

    for i in range(start_idx, len(df)):
        try:
            trade_date = pd.to_datetime(df['date'].iloc[i]).date()
            price = float(df['close'].iloc[i])
            high = float(df['high'].iloc[i])
            low = float(df['low'].iloc[i])
            volume = float(df['volume'].iloc[i])

            # Get features
            feat_row = df.iloc[i][ALL_FEATURES].values.astype(np.float32)
            normalized_features = (feat_row - feat_mean) / feat_std
            state = np.array([position], dtype=np.float32)
            obs = np.concatenate([normalized_features, state]).astype(np.float32)

            # DQN prediction
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # Get Q-values for confidence
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
                q_values = model.q_net(obs_tensor)
                q_values_np = q_values.cpu().numpy()[0]
                q_max = q_values_np.max()
                q_exp = np.exp(q_values_np - q_max)
                q_softmax = q_exp / q_exp.sum()
                dqn_confidence = float(q_softmax[1])

            # Get FVG features
            fvg_bull = float(df['fvg_bull_active'].iloc[i])
            fvg_bear = float(df['fvg_bear_active'].iloc[i])

            # Calculate hybrid signal
            adjusted_conf, signal, trade_info = calculate_hybrid_confidence(
                dqn_confidence, action, fvg_bull, fvg_bear, position
            )

            # Position sizing with volatility adjustment
            vol_20d = float(df['vol_20d'].iloc[i]) if 'vol_20d' in df.columns else 0.015

            if signal == 'FLAT':
                target_pos = 0.0
            else:
                # Volatility scaling
                if Config.VOLATILITY_SCALING:
                    if vol_20d < Config.MIN_VOLATILITY:
                        vol_mult = 1.0
                    elif vol_20d > Config.MAX_VOLATILITY:
                        vol_mult = 0.8
                    else:
                        vol_mult = 1.0 - (vol_20d - Config.MIN_VOLATILITY) /                                   (Config.MAX_VOLATILITY - Config.MIN_VOLATILITY) * 0.2
                else:
                    vol_mult = 1.0

                target_pos = adjusted_conf * Config.POSITION_SCALE * vol_mult
                target_pos = np.clip(target_pos, 0, Config.MAX_POSITION_SIZE)

            # Check for FVG-based exit if in position
            exit_reason = None
            if position > 0 and active_fvg_trade:
                # Check if price hit FVG-based SL or TP
                if low <= active_fvg_trade['stop_loss']:
                    target_pos = 0.0
                    exit_reason = 'fvg_sl'
                elif high >= active_fvg_trade['take_profit']:
                    target_pos = 0.0
                    exit_reason = 'fvg_tp'

            # Calculate P&L
            pos_change = abs(target_pos - position)
            cost = pos_change * equity * 0.00005

            ret = (price - last_price) / last_price if i > start_idx else 0.0
            pnl = position * ret * equity
            equity += pnl - cost

            # Update FVG trade tracking
            if target_pos > 0 and position == 0:
                # New entry - check for fresh FVG
                bull_fvg, bear_fvg = get_active_fvgs(df, i, lookback=5)
                if bull_fvg:
                    active_fvg_trade = build_trade_from_fvg(df, bull_fvg)
                else:
                    active_fvg_trade = None
            elif target_pos == 0 and position > 0:
                # Exit
                active_fvg_trade = None

            position = target_pos
            last_price = price

            if equity > max_equity:
                max_equity = equity

            # Record prediction
            pred_row = {
                'date': trade_date,
                'symbol': symbol,
                'price': price,
                'high': high,
                'low': low,
                'volume': volume,
                'action': action,
                'signal': signal,
                'position': position,
                'dqn_confidence': dqn_confidence,
                'adjusted_confidence': adjusted_conf,
                'fvg_bull_active': fvg_bull,
                'fvg_bear_active': fvg_bear,
                'alignment': trade_info['alignment'],
                'position_change': pos_change,
                'daily_pnl': pnl,
                'trade_cost': cost,
                'equity': equity,
                'exit_reason': exit_reason,
                'fvg_sl': active_fvg_trade['stop_loss'] if active_fvg_trade else None,
                'fvg_tp': active_fvg_trade['take_profit'] if active_fvg_trade else None,
            }

            # Add all features
            for feat in ALL_FEATURES:
                pred_row[feat] = float(df[feat].iloc[i])

            predictions.append(pred_row)

            # Progress update
            if (i - start_idx + 1) % 63 == 0:
                long_pct = 100 * sum(1 for p in predictions if p['signal'] == 'LONG') / len(predictions)
                print(f"  Day {i-start_idx+1:4d}: Equity=${equity:.4f} | "
                      f"Signal={signal:4s} | Pos={position:.2f} | "
                      f"Conf={adjusted_conf:.2f} | LONG%={long_pct:.1f}%")

        except Exception as e:
            print(f"⚠️  Day {i}: Error - {str(e)[:80]}")
            continue

    df_preds = pd.DataFrame(predictions)

    # Summary
    flat_days = (df_preds['signal'] == 'FLAT').sum()
    long_days = (df_preds['signal'] == 'LONG').sum()

    print(f"\n✅ Backtest complete!")
    print(f"   Days processed: {len(df_preds)}")
    print(f"   FLAT: {flat_days} ({flat_days/len(df_preds)*100:.1f}%)")
    print(f"   LONG: {long_days} ({long_days/len(df_preds)*100:.1f}%)")
    print(f"   Final Equity: ${df_preds['equity'].iloc[-1]:.4f}\n")

    return df_preds

# ============================================================
# 8. ANALYSIS & REPORTING
# ============================================================

def analyze_backtest(df_preds: pd.DataFrame) -> dict:
    """Comprehensive backtest analysis"""

    print(f"{'='*70}")
    print(f"📊 FVG-DQN HYBRID BACKTEST ANALYSIS")
    print(f"{'='*70}\n")

    final_equity = df_preds['equity'].iloc[-1]
    total_return = (final_equity - 1.0) * 100

    daily_returns = df_preds['daily_pnl'] / df_preds['equity'].shift(1)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252))              if daily_returns.std() > 0 else 0

    wins = (df_preds['daily_pnl'] > 0).sum()
    losses = (df_preds['daily_pnl'] < 0).sum()
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    running_max = df_preds['equity'].expanding().max()
    drawdown = (df_preds['equity'] - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    daily_vol = daily_returns.std() * np.sqrt(252) * 100

    total_wins = df_preds[df_preds['daily_pnl'] > 0]['daily_pnl'].sum()
    total_losses = abs(df_preds[df_preds['daily_pnl'] < 0]['daily_pnl'].sum())
    profit_factor = total_wins / (total_losses + 1e-8)

    total_trades = (df_preds['position_change'] > 0.01).sum()

    # FVG-specific analysis
    aligned_trades = df_preds[df_preds['alignment'] == 'strong_long']
    contrarian_trades = df_preds[df_preds['alignment'] == 'weak_long']

    results = {
        'final_equity': final_equity,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'daily_volatility': daily_vol,
        'profit_factor': profit_factor,
        'aligned_trades': len(aligned_trades),
        'contrarian_trades': len(contrarian_trades),
    }

    print(f"💰 RETURNS:")
    print(f"   Final Equity:       ${final_equity:.4f}")
    print(f"   Total Return:       {total_return:+.2f}%")
    print(f"   Annualized Sharpe:  {sharpe:.2f}")

    print(f"\n📈 TRADES:")
    print(f"   Total Trades:       {total_trades}")
    print(f"   Win Rate:           {win_rate:.1f}%")
    print(f"   Profit Factor:      {profit_factor:.2f}")

    print(f"\n🎯 FVG SIGNAL ANALYSIS:")
    print(f"   Strong Aligned:     {len(aligned_trades)} ({len(aligned_trades)/len(df_preds)*100:.1f}%)")
    print(f"   Weak/Contrarian:    {len(contrarian_trades)} ({len(contrarian_trades)/len(df_preds)*100:.1f}%)")
    if len(aligned_trades) > 0:
        aligned_pnl = aligned_trades['daily_pnl'].sum()
        print(f"   Avg PnL (Aligned):  {aligned_pnl/len(aligned_trades)*100:.4f}%")

    print(f"\n📉 RISK:")
    print(f"   Max Drawdown:       {max_drawdown:.2f}%")
    print(f"   Daily Volatility:   {daily_vol:.2f}%")

    print(f"\n{'='*70}\n")

    return results

def save_results(conn_str: str, df_preds: pd.DataFrame, run_id: str):
    """Save backtest results to SQL"""

    print(f"💾 Saving to SQL Server...")

    engine = get_engine(conn_str)

    df_preds['run_id'] = run_id
    df_preds['run_date'] = datetime.now()
    df_preds['date'] = pd.to_datetime(df_preds['date']).dt.date

    # Chunk insert
    chunk_size = 500
    total_chunks = (len(df_preds) + chunk_size - 1) // chunk_size

    print(f"   Inserting {len(df_preds):,} rows...")

    for i in range(0, len(df_preds), chunk_size):
        chunk = df_preds.iloc[i:i+chunk_size].copy()
        chunk.to_sql('fvg_dqn_backtest_results', engine, if_exists='append', index=False)
        print(f"   ✅ Chunk {(i // chunk_size) + 1}/{total_chunks}")

    print(f"\n✅ Saved with Run ID: {run_id}\n")

# ============================================================
# 9. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    try:
        print(f"\n{'='*70}")
        print(f"🚀 FVG-DQN HYBRID BACKTEST")
        print(f"{'='*70}\n")

        # Step 1: Load data
        print("STEP 1: Load price data")
        print("-" * 70)
        df_raw = load_price_data(Config.SQL_CONN, Config.SYMBOL, 
                                 Config.BACKTEST_START, Config.BACKTEST_END)

        # Step 2: Compute features
        print("STEP 2: Compute features (Technical + FVG)")
        print("-" * 70)
        df = compute_features(df_raw)

        # Step 3: Run backtest
        print("STEP 3: Run FVG-DQN hybrid backtest")
        print("-" * 70)
        df_preds = run_fvg_dqn_backtest(Config.MODEL_PATH, df, Config.SYMBOL)

        # Step 4: Analyze
        print("STEP 4: Analyze results")
        print("-" * 70)
        results = analyze_backtest(df_preds)

        # Step 5: Save
        print("STEP 5: Save to SQL")
        print("-" * 70)
        run_id = f"fvg_dqn_{Config.SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_results(Config.SQL_CONN, df_preds, run_id)

        # Final summary
        print(f"\n{'='*70}")
        print(f"✅ FVG-DQN BACKTEST COMPLETE")
        print(f"{'='*70}\n")

        print(f"📊 FINAL RESULTS:")
        print(f"   Return:        {results['total_return_pct']:+.2f}%")
        print(f"   Sharpe:        {results['sharpe_ratio']:.2f}")
        print(f"   Max DD:        {results['max_drawdown']:.2f}%")
        print(f"   Win Rate:      {results['win_rate']:.1f}%")
        print(f"\n🎯 FVG Enhancement:")
        print(f"   Aligned signals improved confidence by {Config.FVG_CONFIRMATION_BOOST*100:.0f}%")
        print(f"   Contrarian signals reduced by {Config.FVG_CONTRARIAN_PENALTY*100:.0f}%")
        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()