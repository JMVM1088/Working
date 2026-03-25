import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import sqlalchemy as sa
import urllib
from datetime import datetime, timedelta, date

# =============================================================================
# DB CONNECTION
# =============================================================================

def get_engine():
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=BEELINK;"
        "DATABASE=Stock;"
        "Trusted_Connection=yes;"
    )
    quoted = urllib.parse.quote_plus(connection_string)
    engine = sa.create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}")
    return engine

# =============================================================================
# FVG + TRADE PARAMS
# =============================================================================

@dataclass
class FVG:
    symbol: str
    max: float
    min: float
    isbull: bool
    t_index: int
    t_time: pd.Timestamp


def detect_fvg_for_symbol(df: pd.DataFrame,
                          threshold_per: float = 0.0,
                          auto: bool = False) -> List[FVG]:
    df = df.sort_values("date").reset_index(drop=True)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    if n < 3:
        return []

    if auto:
        rel_range = (high - low) / np.where(low == 0, np.nan, low)
        cum = np.nancumsum(rel_range)
        idx = np.arange(1, n + 1, dtype=float)
        threshold = cum / idx
    else:
        threshold = np.full(n, threshold_per / 100.0)

    bull_fvg = np.full(n, False)
    bear_fvg = np.full(n, False)

    high_2 = np.concatenate(([np.nan, np.nan], high[:-2]))
    low_2 = np.concatenate(([np.nan, np.nan], low[:-2]))
    close_1 = np.concatenate(([np.nan], close[:-1]))

    cond_bull = (
        (low > high_2) &
        (close_1 > high_2) &
        ((low - high_2) / high_2 > threshold)
    )

    cond_bear = (
        (high < low_2) &
        (close_1 < low_2) &
        ((low_2 - high) / high > threshold)
    )

    bull_fvg[cond_bull] = True
    bear_fvg[cond_bear] = True

    fvg_records: List[FVG] = []
    last_t_index: Optional[int] = None

    for i in range(n):
        if not bull_fvg[i] and not bear_fvg[i]:
            continue

        if bull_fvg[i]:
            max_val = low[i]
            min_val = high[i - 2]
            isbull = True
        else:
            max_val = high[i]
            min_val = low[i - 2]
            isbull = False

        if last_t_index is not None and last_t_index == i:
            continue

        fvg_records.append(
            FVG(
                symbol=str(df.loc[i, "symbol"]),
                max=max_val,
                min=min_val,
                isbull=isbull,
                t_index=i,
                t_time=df.loc[i, "date"],
            )
        )
        last_t_index = i

    return fvg_records


def build_trade_from_fvg(
    df_sym: pd.DataFrame,
    fvg: FVG,
    rr_target: float = 2.0,
    sl_buffer_mult: float = 1.0,
) -> Dict[str, Any]:
    """
    Build a single trade recommendation (entry, SL, TP) from one FVG. [web:44][web:47][web:50]
    """
    df_sym = df_sym.sort_values("date").reset_index(drop=True)
    close = df_sym["close"].values
    high = df_sym["high"].values
    low = df_sym["low"].values
    dates = df_sym["date"].values

    entry_idx = fvg.t_index
    if entry_idx >= len(df_sym):
        return {}

    entry_time = dates[entry_idx]
    entry_price = close[entry_idx]
    direction = "long" if fvg.isbull else "short"

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
        "symbol": fvg.symbol,
        "direction": direction,
        "creation_time": fvg.t_time,
        "creation_index": fvg.t_index,
        "max": fvg.max,
        "min": fvg.min,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "status": "open",               # open / closed
        "exit_time": None,
        "exit_price": None,
        "exit_reason": None,            # sl / tp / time / manual
        "as_of_date": dates[-1],        # last bar used for generation
    }


# =============================================================================
# UPDATE EXISTING TRADES WITH SL/TP
# =============================================================================

def update_existing_trades(engine, as_of_date: Optional[str] = None):
    """
    Load all open trades from AI_FVG_Recommendations, check if SL/TP is hit
    with the available OHLC data, and update them in-place. [web:44][web:50]
    """
    conn = engine.connect()

    if as_of_date is None:
        # as_of_date = datetime.utcnow().date().isoformat()
        as_of_date = date.today().isoformat()

    trades_sql = """
        SELECT *
        FROM AI_FVG_Recommendations
        WHERE status = 'open'
    """
    open_trades = pd.read_sql(trades_sql, conn)

    if open_trades.empty:
        conn.close()
        return

    symbols = open_trades["symbol"].unique().tolist()
    sym_list = ",".join(f"'{s}'" for s in symbols)
    lookback_days = 90  # 60–90 is a good range
    start_date = date.today() - timedelta(days=lookback_days)
    prices_sql = f"""
        SELECT [time] as date, symbol, [open], high, low, [close]
        FROM AI_stock_Prices
        WHERE symbol IN ({sym_list})
          AND [time] <= '{as_of_date}'
          AND [time] >= '{start_date:%Y-%m-%d}'
        ORDER BY symbol, date
    """
    prices = pd.read_sql(prices_sql, conn)

    updates = []

    for symbol, df_sym in prices.groupby("symbol"):
        df_sym = df_sym.sort_values("date").reset_index(drop=True)
        high = df_sym["high"].values
        low = df_sym["low"].values
        dates = df_sym["date"].values

        trades_sym = open_trades[open_trades["symbol"] == symbol].copy()

        for idx, tr in trades_sym.iterrows():
            entry_time = tr["entry_time"]
            try:
                entry_idx = int(np.where(dates == entry_time)[0][0])
            except IndexError:
                continue

            stop_loss = tr["stop_loss"]
            take_profit = tr["take_profit"]
            direction = tr["direction"]

            exit_idx = None
            exit_price = None
            exit_reason = None

            for i in range(entry_idx + 1, len(df_sym)):
                bar_high = high[i]
                bar_low = low[i]

                if direction == "long":
                    if bar_low <= stop_loss:
                        exit_idx = i
                        exit_price = stop_loss
                        exit_reason = "sl"
                        break
                    if bar_high >= take_profit:
                        exit_idx = i
                        exit_price = take_profit
                        exit_reason = "tp"
                        break
                else:
                    if bar_high >= stop_loss:
                        exit_idx = i
                        exit_price = stop_loss
                        exit_reason = "sl"
                        break
                    if bar_low <= take_profit:
                        exit_idx = i
                        exit_price = take_profit
                        exit_reason = "tp"
                        break

            if exit_idx is not None:
                updates.append({
                    "id": tr["id"],                  # assumes primary key 'id'
                    "exit_time": dates[exit_idx],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "status": "closed",
                })

    for u in updates:
        update_sql = """
            UPDATE AI_FVG_Recommendations
            SET exit_time = :exit_time,
                exit_price = :exit_price,
                exit_reason = :exit_reason,
                status = :status
            WHERE id = :id
        """
        conn.execute(sa.text(update_sql), **u)

    conn.close()


# =============================================================================
# GENERATE NEW DAILY RECOMMENDATIONS
# =============================================================================

def generate_daily_recommendations(
    threshold_per: float = 0.0,
    auto: bool = False,
    rr_target: float = 2.0,
    sl_buffer_mult: float = 1.0,
    as_of_date: Optional[str] = None,
):
    """
    - Optionally first update existing open trades for SL/TP.
    - Then detect FVGs using all data up to as_of_date.
    - For FVGs created on the latest bar, create new trade recommendations. [web:44][web:47][web:50]
    """
    engine = get_engine()
    conn = engine.connect()

    if as_of_date is None:
        # as_of_date = datetime.utcnow().date().isoformat()
        as_of_date = date.today().isoformat()

    update_existing_trades(engine, as_of_date=as_of_date)

    # as_of_date = datetime(2026, 2, 6).date()  # for backtesting with fixed date
    lookback_days = 90  # 60–90 is a good range
    start_date = date.today() - timedelta(days=lookback_days)

    prices_sql = f"""
    SELECT [time] as date, symbol, [open], high, low, [close]
    FROM AI_stock_Prices
    WHERE [time] >= '{as_of_date}'
    ORDER BY symbol, date
    """
    prices = pd.read_sql(prices_sql, conn)

    last_date = prices["date"].max()

    all_recos = []

    for symbol, df_sym in prices.groupby("symbol"):
        df_sym = df_sym.sort_values("date").reset_index(drop=True)
        fvg_list = detect_fvg_for_symbol(df_sym, threshold_per=threshold_per, auto=auto)

        if not fvg_list:
            continue

        last_idx = len(df_sym) - 1
        for fvg in fvg_list:
            if fvg.t_index == last_idx:
                trade = build_trade_from_fvg(
                    df_sym,
                    fvg,
                    rr_target=rr_target,
                    sl_buffer_mult=sl_buffer_mult,
                )
                if trade:
                    all_recos.append(trade)

    if all_recos:
        recos_df = pd.DataFrame(all_recos)

        recos_df.to_sql(
            "AI_FVG_Recommendations",
            engine,
            if_exists="append",
            index=False
        )

    conn.close()


if __name__ == "__main__":
    # Typical daily run after market close (schedule externally via cron/Task Scheduler). [web:39][web:41]
    generate_daily_recommendations(
        threshold_per=0.5,     # adjust threshold
        auto=False,            # or True if you want auto threshold
        rr_target=2.0,         # 2R target
        sl_buffer_mult=1.0,    # SL just beyond gap
    )
