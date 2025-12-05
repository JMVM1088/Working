from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import math
import numpy as np
import pandas as pd
import sqlite3
import Util


class AlgorithmType:
    LEGACY = 'LEGACY'
    ENHANCED = 'ENHANCED'


def linear_regression(points: List[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    if len(points) < 2:
        return None
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {'slope': float(slope), 'intercept': float(intercept)}


def calculate_rolling_stats(values: List[float], period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kept for compatibility; currently not used to filter pivots.
    """
    s = pd.Series(values)
    sma = np.empty(len(s))
    std = np.empty(len(s))
    for i in range(len(s)):
        if i < period - 1:
            window = s.iloc[0 : i + 1]
        else:
            window = s.iloc[i - period + 1 : i + 1]
        mean = window.mean()
        var = ((window - mean) ** 2).mean()
        sma[i] = float(mean) if not math.isnan(mean) else 0.0
        std[i] = float(math.sqrt(var)) if not math.isnan(var) else 0.0
    return sma, std


def analyze_pattern(
    data: pd.DataFrame,
    anchor_date: str,
    lookback_days: int = 60,          # focus on recent structure
    algo_type: str = AlgorithmType.ENHANCED,
    pivot_window: int = 2,
    std_dev_multiplier: float = 0.0,  # not used in this version
    sma_period: int = 20,
) -> Dict:
    """
    Analyze price series and detect converging pattern at anchor_date.

    Pattern detection uses only data up to anchor_date.
    Breakout detection on anchor_date is optional and can be ignored
    when you are scanning for pre-breakout setups.
    """
    df = data.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    else:
        df.index = pd.to_datetime(df.index)

    anchor = pd.to_datetime(anchor_date)
    if anchor not in df.index:
        return {
            'found': False,
            'pivotsHigh': [],
            'pivotsLow': [],
            'maturity': 0.0,
            'widthStart': 0.0,
            'widthEnd': 0.0,
            'breakout': False,
            'breakoutType': None,
            'resistanceLine': None,
            'supportLine': None,
            'resistanceAtAnchor': None,
            'supportAtAnchor': None,
            'breakoutVolumeFactor': None,
        }

    anchor_idx = df.index.get_loc(anchor)
    start_idx = max(0, anchor_idx - lookback_days + 1)
    window = df.iloc[start_idx : anchor_idx + 1].copy()
    if window.empty or len(window) < 2 * pivot_window + 2:
        return {
            'found': False,
            'pivotsHigh': [],
            'pivotsLow': [],
            'maturity': 0.0,
            'widthStart': 0.0,
            'widthEnd': 0.0,
            'breakout': False,
            'breakoutType': None,
            'resistanceLine': None,
            'supportLine': None,
            'resistanceAtAnchor': None,
            'supportAtAnchor': None,
            'breakoutVolumeFactor': None,
        }

    closes = window['Close'].tolist()
    sma, std = calculate_rolling_stats(closes, sma_period)

    pivots_high: List[Dict] = []
    pivots_low: List[Dict] = []

    n = len(window)
    highs = window['High'].values
    lows = window['Low'].values

    # ---- pure price-based pivot detection ----
    for i in range(pivot_window, n - pivot_window):
        current_high = float(highs[i])
        current_low = float(lows[i])

        # swing high
        is_high = True
        for j in range(1, pivot_window + 1):
            if highs[i - j] > current_high or highs[i + j] > current_high:
                is_high = False
                break
        if is_high:
            pivots_high.append(
                {'index': i, 'date': window.index[i], 'value': current_high, 'type': 'high'}
            )

        # swing low
        is_low = True
        for j in range(1, pivot_window + 1):
            if lows[i - j] < current_low or lows[i + j] < current_low:
                is_low = False
                break
        if is_low:
            pivots_low.append(
                {'index': i, 'date': window.index[i], 'value': current_low, 'type': 'low'}
            )

    if len(pivots_high) < 2 or len(pivots_low) < 2:
        return {
            'found': False,
            'pivotsHigh': pivots_high,
            'pivotsLow': pivots_low,
            'maturity': 0.0,
            'widthStart': 0.0,
            'widthEnd': 0.0,
            'breakout': False,
            'breakoutType': None,
            'resistanceLine': None,
            'supportLine': None,
            'resistanceAtAnchor': None,
            'supportAtAnchor': None,
            'breakoutVolumeFactor': None,
        }

    m_res = 0.0
    c_res = 0.0
    m_sup = 0.0
    c_sup = 0.0

    # ---- resistance/support line estimation ----
    if algo_type == AlgorithmType.LEGACY:
        # connect extreme high to last high
        max_val = -float('inf')
        max_idx = -1
        for p in pivots_high:
            if p['value'] > max_val:
                max_val = p['value']
                max_idx = p['index']
        last_pivot_high = pivots_high[-1]
        p2_high = last_pivot_high
        if last_pivot_high['index'] == max_idx and len(pivots_high) > 1:
            p2_high = pivots_high[-2]
        if max_idx != -1 and p2_high['index'] != max_idx:
            m_res = (p2_high['value'] - max_val) / (p2_high['index'] - max_idx)
            c_res = max_val - m_res * max_idx

        # connect extreme low to last low
        min_val = float('inf')
        min_idx = -1
        for p in pivots_low:
            if p['value'] < min_val:
                min_val = p['value']
                min_idx = p['index']
        last_pivot_low = pivots_low[-1]
        p2_low = last_pivot_low
        if last_pivot_low['index'] == min_idx and len(pivots_low) > 1:
            p2_low = pivots_low[-2]
        if min_idx != -1 and p2_low['index'] != min_idx:
            m_sup = (p2_low['value'] - min_val) / (p2_low['index'] - min_idx)
            c_sup = min_val - m_sup * min_idx

    else:
        # Enhanced: regression on most recent pivots
        MAX_PIVOTS = 6
        high_points = [(p['index'], p['value']) for p in pivots_high[-MAX_PIVOTS:]]
        low_points = [(p['index'], p['value']) for p in pivots_low[-MAX_PIVOTS:]]
        res_line = linear_regression(high_points)
        sup_line = linear_regression(low_points)
        if res_line:
            m_res = res_line['slope']
            c_res = res_line['intercept']
        if sup_line:
            m_sup = sup_line['slope']
            c_sup = sup_line['intercept']

    current_day_idx = n - 1
    mid_idx = n // 2

    # ---- convergence over recent half of window ----
    y_res_mid = m_res * mid_idx + c_res
    y_sup_mid = m_sup * mid_idx + c_sup
    y_res_end = m_res * current_day_idx + c_res
    y_sup_end = m_sup * current_day_idx + c_sup

    width_start = y_res_mid - y_sup_mid
    width_end = y_res_end - y_sup_end

    is_converging = (width_start > 0) and (width_end > 0) and (width_end < width_start)
    apex_x = None
    if abs(m_res - m_sup) > 1e-9:
        apex_x = (c_sup - c_res) / (m_res - m_sup)

    maturity = 0.0
    if apex_x and apex_x != 0:
        maturity = float(current_day_idx / apex_x)

    if algo_type == AlgorithmType.ENHANCED:
        # slightly looser so pattern is recognized earlier
        is_converging = (
            width_start > 0
            and width_end > 0
            and width_end < 0.9 * width_start
        )

    # ---- breakout detection on anchor bar (optional) ----
    anchor_local_idx = anchor_idx - start_idx  # index inside window
    anchor_price_high = float(window['High'].iloc[anchor_local_idx])
    anchor_price_low = float(window['Low'].iloc[anchor_local_idx])
    anchor_price_close = float(window['Close'].iloc[anchor_local_idx])

    res_at_anchor = m_res * anchor_local_idx + c_res
    sup_at_anchor = m_sup * anchor_local_idx + c_sup

    breakout = False
    breakout_type: Optional[str] = None

    breakout_volume_factor: Optional[float] = None
    high_volume = True
    if 'Volume' in window.columns:
        vol_series = window['Volume'].astype(float)
        vol_anchor = float(vol_series.iloc[anchor_local_idx])
        vol_ma = float(vol_series.rolling(20).mean().iloc[anchor_local_idx])
        if not math.isnan(vol_ma) and vol_ma > 0:
            breakout_volume_factor = vol_anchor / vol_ma
            high_volume = breakout_volume_factor >= 1.2
        else:
            breakout_volume_factor = None
            high_volume = True

    if is_converging:
        # Upward breakout on anchor bar (for diagnostics)
        if (
            anchor_price_high > res_at_anchor * 1.005
            and anchor_price_close > res_at_anchor
            and high_volume
        ):
            breakout = True
            breakout_type = 'up'
        # Downward breakout on anchor bar
        elif (
            anchor_price_low < sup_at_anchor * 0.995
            and anchor_price_close < sup_at_anchor
            and high_volume
        ):
            breakout = True
            breakout_type = 'down'

    return {
        'found': bool(is_converging),
        'resistanceLine': {
            'slope': m_res,
            'intercept': c_res,
            'startIdx': start_idx,
            'endIdx': anchor_idx,
        },
        'supportLine': {
            'slope': m_sup,
            'intercept': c_sup,
            'startIdx': start_idx,
            'endIdx': anchor_idx,
        },
        'pivotsHigh': pivots_high,
        'pivotsLow': pivots_low,
        'maturity': maturity,
        'widthStart': float(width_start),
        'widthEnd': float(width_end),
        'breakout': breakout,               # same-bar breakout (optional)
        'breakoutType': breakout_type,
        'resistanceAtAnchor': float(res_at_anchor),
        'supportAtAnchor': float(sup_at_anchor),
        'breakoutVolumeFactor': breakout_volume_factor,
    }


def get_dates_from_sqlite(
    db_file: str,
    table_name: str = 'HistoricalPrices',
    ticker: Optional[str] = None
) -> List[str]:
    """Return list of available dates (YYYY-MM-DD) for a ticker from an SQLite table."""
    db = db_file or getattr(Util, 'DB_FILE', None)
    if ticker:
        query = f"SELECT Date FROM {table_name} WHERE Ticker = ? ORDER BY Date"
        rows = Util.get_data_from_sqlite(db, query, params=(ticker,))
    else:
        query = f"SELECT DISTINCT Date FROM {table_name} ORDER BY Date"
        rows = Util.get_data_from_sqlite(db, query)

    dates: List[str] = []
    for r in rows:
        d = r[0] if isinstance(r, (list, tuple)) else r
        try:
            dt = pd.to_datetime(d)
            dates.append(dt.strftime('%Y-%m-%d'))
        except Exception:
            continue
    return dates


def get_ohlcv_from_sqlite(
    db_file: str,
    ticker: str,
    table_name: str = 'HistoricalPrices'
) -> pd.DataFrame:
    """Load OHLCV for a given ticker from SQLite and return a DataFrame indexed by Date."""
    db = db_file or getattr(Util, 'DB_FILE', None)
    query = (
        f"SELECT Date, Open, High, Low, Close, Volume "
        f"FROM {table_name} WHERE Ticker = ? ORDER BY Date"
    )
    rows = Util.get_data_from_sqlite(db, query, params=(ticker,))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
        if df['Date'].isna().any():
            df['Date'] = pd.to_datetime(df['Date'].astype(str), errors='coerce')
    except Exception:
        df['Date'] = pd.to_datetime(df['Date'].astype(str), errors='coerce')

    df = df.dropna(subset=['Date']).set_index('Date')
    df = df[['High', 'Low', 'Close', 'Volume']].copy()
    return df


def load_stock_and_market_from_sqlite(
    db_file: str,
    ticker: str,
    table_name: str = 'HistoricalPrices'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stock_df, market_df) where market_df is SPY with SMA_200 computed."""
    stock_df = get_ohlcv_from_sqlite(db_file, ticker, table_name)

    db = db_file or getattr(Util, 'DB_FILE', None)
    query_spy = f"SELECT Date, Close FROM {table_name} WHERE Ticker = 'SPY' ORDER BY Date"
    spy_rows = Util.get_data_from_sqlite(db, query_spy)

    if not spy_rows:
        market_df = pd.DataFrame(index=stock_df.index)
        market_df['Close'] = stock_df['Close']
        market_df['SMA_200'] = market_df['Close'].rolling(window=200).mean()
    else:
        spy_df = pd.DataFrame(spy_rows, columns=['Date', 'Close'])
        try:
            spy_df['Date'] = pd.to_datetime(spy_df['Date'], format='%Y%m%d', errors='coerce')
            if spy_df['Date'].isna().any():
                spy_df['Date'] = pd.to_datetime(spy_df['Date'].astype(str), errors='coerce')
        except Exception:
            spy_df['Date'] = pd.to_datetime(spy_df['Date'].astype(str), errors='coerce')
        spy_df = spy_df.dropna(subset=['Date']).set_index('Date')
        market_df = spy_df[['Close']].copy()
        market_df['SMA_200'] = market_df['Close'].rolling(window=200).mean()

    return stock_df, market_df


if __name__ == '__main__':
    # Example usage on ORCL
    db_file = r"C:\Users\jv2mk\OneDrive\Stock\Screener\DB\stage2"
    table_name = 'Historical_SP500'
    ticker = 'ORCL'
    anchor_date = '2025-09-05'  # a few days BEFORE breakout
    lookback_days = 60
    pivot_window = 2
    algo_type = AlgorithmType.ENHANCED

    stock_df, market_df = load_stock_and_market_from_sqlite(db_file, ticker, table_name=table_name)
    if stock_df.empty:
        print(f"No data found for ticker {ticker} in table {table_name} at DB {db_file}")
    else:
        df_in = stock_df.reset_index().rename(columns={'index': 'Date'})
        res = analyze_pattern(
            df_in,
            anchor_date,
            lookback_days=lookback_days,
            algo_type=algo_type,
            pivot_window=pivot_window,
        )
        print('Anchor date:', anchor_date)
        print('Pattern found:', res.get('found'))
        print('Maturity:', res.get('maturity'))
        print('Width start / end:', res.get('widthStart'), res.get('widthEnd'))
        print('Same-bar breakout:', res.get('breakout'), 'Type:', res.get('breakoutType'))
        print('Pivots High:', len(res.get('pivotsHigh', [])), 'Pivots Low:', len(res.get('pivotsLow', [])))
        print('Resistance:', res.get('resistanceLine'))
        print('Support:', res.get('supportLine'))
