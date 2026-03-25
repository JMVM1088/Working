import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import sqlalchemy as sa
import urllib
from itertools import product

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
# FVG LOGIC
# =============================================================================

@dataclass
class FVG:
    symbol: str
    max: float
    min: float
    isbull: bool
    t_index: int
    t_time: pd.Timestamp


def detect_fvg_for_symbol(
    df: pd.DataFrame,
    threshold_per: float = 0.0,
    auto: bool = False
) -> List[FVG]:
    """
    FVG detection logic (approximation of LuxAlgo script).
    threshold_per: minimum gap height as percent if auto=False.
    auto: if True, use cumulative mean of relative ranges as threshold. [web:1]
    """
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


def mark_mitigation(df: pd.DataFrame, fvg_list: List[FVG]) -> pd.DataFrame:
    """
    For each FVG, mark if/when it is mitigated (FVG filled/invalidated).
    """
    if not fvg_list:
        return pd.DataFrame(columns=[
            "symbol", "creation_time", "creation_index",
            "isbull", "max", "min",
            "mitigated", "mitigation_time", "mitigation_index"
        ])

    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].values
    dates = df["date"].values

    records = []
    for fvg in fvg_list:
        mitigated = False
        mit_idx = None
        mit_time = None

        for i in range(fvg.t_index + 1, len(df)):
            if fvg.isbull:
                if close[i] < fvg.min:
                    mitigated = True
                    mit_idx = i
                    mit_time = dates[i]
                    break
            else:
                if close[i] > fvg.max:
                    mitigated = True
                    mit_idx = i
                    mit_time = dates[i]
                    break

        records.append({
            "symbol": fvg.symbol,
            "creation_time": fvg.t_time,
            "creation_index": fvg.t_index,
            "isbull": fvg.isbull,
            "max": fvg.max,
            "min": fvg.min,
            "mitigated": mitigated,
            "mitigation_time": mit_time,
            "mitigation_index": mit_idx,
        })

    return pd.DataFrame(records)


# =============================================================================
# BACKTEST WITH SL/TP
# =============================================================================

def backtest_fvg_strategy(
    df: pd.DataFrame,
    fvg_df: pd.DataFrame,
    risk_per_trade: float = 0.01,
    initial_capital: float = 10000.0,
    rr_target: float = 2.0,
    sl_buffer_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Backtest FVG trades with stop loss and take profit. [web:21][web:24][web:29]

    - Entry at close of creation bar.
    - Stop loss just outside the gap boundary (using gap height * sl_buffer_mult).
    - Take profit at rr_target * risk (R:R target).
    - Uses gap height as risk proxy.

    sl_buffer_mult ~1 means SL just beyond FVG, >1 makes it wider.
    """
    if fvg_df.empty:
        return pd.DataFrame(columns=[
            "symbol", "entry_time", "exit_time",
            "direction", "entry_price", "exit_price",
            "pnl", "return_pct", "exit_reason"
        ])

    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    dates = df["date"].values

    trades = []
    equity = initial_capital

    for _, row in fvg_df.iterrows():
        s = row["symbol"]
        isbull = bool(row["isbull"])
        entry_idx = int(row["creation_index"])

        if entry_idx >= len(df) - 1:
            continue

        entry_price = close[entry_idx]
        direction = "long" if isbull else "short"

        gap_height = abs(row["max"] - row["min"])
        if gap_height <= 0:
            gap_height = entry_price * 0.01

        gap_height *= sl_buffer_mult
        risk_dollars = equity * risk_per_trade
        qty = risk_dollars / gap_height

        if direction == "long":
            sl = entry_price - gap_height
            tp = entry_price + rr_target * gap_height
        else:
            sl = entry_price + gap_height
            tp = entry_price - rr_target * gap_height

        exit_idx = None
        exit_price = None
        exit_reason = None

        for i in range(entry_idx + 1, len(df)):
            bar_high = high[i]
            bar_low = low[i]

            if direction == "long":
                if bar_low <= sl:
                    exit_idx = i
                    exit_price = sl
                    exit_reason = "stop"
                    break
                if bar_high >= tp:
                    exit_idx = i
                    exit_price = tp
                    exit_reason = "tp"
                    break
            else:
                if bar_high >= sl:
                    exit_idx = i
                    exit_price = sl
                    exit_reason = "stop"
                    break
                if bar_low <= tp:
                    exit_idx = i
                    exit_price = tp
                    exit_reason = "tp"
                    break

        if exit_idx is None:
            exit_idx = len(df) - 1
            exit_price = close[exit_idx]
            exit_reason = "time"

        if direction == "long":
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty

        ret_pct = pnl / equity
        equity += pnl

        trades.append({
            "symbol": s,
            "entry_time": dates[entry_idx],
            "exit_time": dates[exit_idx],
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "return_pct": ret_pct,
            "exit_reason": exit_reason,
        })

    return pd.DataFrame(trades)


# =============================================================================
# PARAMETER OPTIMIZATION
# =============================================================================

def evaluate_param_set(
    prices: pd.DataFrame,
    threshold_per: float,
    auto: bool,
    risk_per_trade: float,
    rr_target: float,
    sl_buffer_mult: float,
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run FVG detection + backtest on all symbols for a given parameter set,
    return aggregate metrics to optimize on. [web:16][web:19][web:26]
    """
    totals = {
        "total_pnl": 0.0,
        "total_trades": 0,
        "sharpe_like": 0.0,
    }
    all_trades = []

    for symbol, df_sym in prices.groupby("symbol"):
        fvg_list = detect_fvg_for_symbol(
            df_sym,
            threshold_per=threshold_per,
            auto=auto
        )
        fvg_df = mark_mitigation(df_sym, fvg_list)
        trades_df = backtest_fvg_strategy(
            df_sym,
            fvg_df,
            risk_per_trade=risk_per_trade,
            initial_capital=initial_capital,
            rr_target=rr_target,
            sl_buffer_mult=sl_buffer_mult,
        )
        if trades_df.empty:
            continue
        all_trades.append(trades_df)

    if not all_trades:
        return {
            "total_pnl": -np.inf,
            "total_trades": 0,
            "sharpe_like": -np.inf,
        }

    trades_all = pd.concat(all_trades, ignore_index=True)
    totals["total_pnl"] = trades_all["pnl"].sum()
    totals["total_trades"] = len(trades_all)

    ret_series = trades_all["return_pct"]
    if ret_series.std() > 0:
        totals["sharpe_like"] = ret_series.mean() / ret_series.std()
    else:
        totals["sharpe_like"] = -np.inf

    return totals


def optimize_parameters(
    prices: pd.DataFrame,
    threshold_grid: List[float],
    auto_grid: List[bool],
    risk_grid: List[float],
    rr_grid: List[float],
    sl_buffer_grid: List[float],
    objective: str = "sharpe_like"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Simple grid search optimization for parameters. [web:19][web:20][web:26]

    objective: 'sharpe_like' or 'total_pnl'
    """
    best_params: Dict[str, Any] = {}
    best_metrics: Dict[str, Any] = {}
    best_score = -np.inf

    for threshold_per, auto, risk_per_trade, rr_target, sl_buffer_mult in product(
        threshold_grid, auto_grid, risk_grid, rr_grid, sl_buffer_grid
    ):
        metrics = evaluate_param_set(
            prices,
            threshold_per=threshold_per,
            auto=auto,
            risk_per_trade=risk_per_trade,
            rr_target=rr_target,
            sl_buffer_mult=sl_buffer_mult,
        )

        if objective == "total_pnl":
            score = metrics["total_pnl"]
        else:
            score = metrics["sharpe_like"]

        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_params = {
                "threshold_per": threshold_per,
                "auto": auto,
                "risk_per_trade": risk_per_trade,
                "rr_target": rr_target,
                "sl_buffer_mult": sl_buffer_mult,
            }

    return best_params, best_metrics


# =============================================================================
# MAIN SCAN + SAVE RESULTS
# =============================================================================

def run_daily_scan_with_optimization(
    threshold_grid: List[float],
    auto_grid: List[bool],
    risk_grid: List[float],
    rr_grid: List[float],
    sl_buffer_grid: List[float],
    objective: str = "sharpe_like",
    save_table_signals: str = "AI_FVG_Signals",
    save_table_backtest: str = "AI_FVG_Backtest",
    save_table_params: str = "AI_FVG_Params",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    engine = get_engine()

    where_clauses = []
    if start_date:
        where_clauses.append(f"[time] >= '{start_date}'")
    if end_date:
        where_clauses.append(f"[time] <= '{end_date}'")
    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    sql = f"""
        SELECT [time] as date, symbol, [open], high, low, [close]
        FROM AI_stock_Prices
        {where_sql}
        ORDER BY symbol, date
    """
    prices = pd.read_sql(sql, engine)

    # 1) Optimize parameters on the whole universe / in-sample period.
    best_params, best_metrics = optimize_parameters(
        prices,
        threshold_grid=threshold_grid,
        auto_grid=auto_grid,
        risk_grid=risk_grid,
        rr_grid=rr_grid,
        sl_buffer_grid=sl_buffer_grid,
        objective=objective,
    )

    # 2) Run FVG detection and backtest once more with best params, save results.
    all_fvg_rows = []
    all_backtest_rows = []

    for symbol, df_sym in prices.groupby("symbol"):
        fvg_list = detect_fvg_for_symbol(
            df_sym,
            threshold_per=best_params["threshold_per"],
            auto=best_params["auto"],
        )
        fvg_df = mark_mitigation(df_sym, fvg_list)
        if not fvg_df.empty:
            all_fvg_rows.append(fvg_df)

        trades_df = backtest_fvg_strategy(
            df_sym,
            fvg_df,
            risk_per_trade=best_params["risk_per_trade"],
            rr_target=best_params["rr_target"],
            sl_buffer_mult=best_params["sl_buffer_mult"],
        )
        if not trades_df.empty:
            all_backtest_rows.append(trades_df)

    if all_fvg_rows:
        all_fvg_df = pd.concat(all_fvg_rows, ignore_index=True)
        all_fvg_df.to_sql(save_table_signals, engine, if_exists="append", index=False)

    if all_backtest_rows:
        all_backtest_df = pd.concat(all_backtest_rows, ignore_index=True)
        all_backtest_df.to_sql(save_table_backtest, engine, if_exists="append", index=False)

    params_row = best_params.copy()
    params_row.update(best_metrics)
    params_row_df = pd.DataFrame([params_row])
    params_row_df.to_sql(save_table_params, engine, if_exists="append", index=False)


if __name__ == "__main__":
    # Example small grids; expand as needed.
    threshold_grid = [0.0, 0.2, 0.5, 1.0]      # percent gap height threshold
    auto_grid = [False]                        # you can add True to explore auto mode
    risk_grid = [0.005, 0.01, 0.02]            # 0.5%, 1%, 2% risk per trade
    rr_grid = [1.5, 2.0, 3.0]                  # reward:risk target
    sl_buffer_grid = [0.8, 1.0, 1.2]           # SL just inside/at/outside FVG

    run_daily_scan_with_optimization(
        threshold_grid=threshold_grid,
        auto_grid=auto_grid,
        risk_grid=risk_grid,
        rr_grid=rr_grid,
        sl_buffer_grid=sl_buffer_grid,
        objective="sharpe_like",               # or "total_pnl"
        start_date='2025-12-01',                       # e.g. "2020-01-01"
        end_date=None,
    )
