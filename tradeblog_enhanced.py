"""
TradeBlog Database Implementation with Mibian Greeks Engine
ENHANCED VERSION with Multi-Batch Support and EOD Processing
Author: Trading System
Date: 2026-01-22
Purpose: Professional iron condor trade tracking with automated Greeks calculation
"""

import os
import pyodbc
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import yfinance as yf
import mibian
import json
from pathlib import Path

load_dotenv()


# ============================================================================
# GREEKS ENGINE (Mibian + yfinance)
# ============================================================================

class GreeksEngine:
    """
    Central engine for SPY option Greeks calculation using mibian + yfinance.
    Fetches spot price and option chain from yfinance, computes Greeks via mibian.
    """

    def __init__(self, symbol: str = "SPY", risk_free_rate: float = 0.05):
        """
        Args:
            symbol: Underlying ticker (default SPY)
            risk_free_rate: Annual risk-free rate as decimal (e.g., 0.05 = 5%)
        """
        self.symbol = symbol
        self.r = risk_free_rate

    def _get_spot(self) -> float:
        """Get latest SPY spot price from yfinance."""
        ticker = yf.Ticker(self.symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            raise ValueError(f"Could not fetch price for {self.symbol}")
        return float(hist["Close"][-1])

    def _get_chain(self, expiry: str):
        """Get option chain for given expiry from yfinance."""
        ticker = yf.Ticker(self.symbol)
        try:
            chain = ticker.option_chain(expiry)
            return chain
        except Exception as e:
            raise ValueError(f"Could not fetch option chain for {expiry}: {str(e)}")

    def _get_mid_price(self, df, strike: float) -> float:
        """Extract mid price from option chain row."""
        row = df[df["strike"] == strike]
        if row.empty:
            raise ValueError(f"No option found for strike {strike}")
        row = row.iloc[0]
        bid = float(row.get("bid", 0.0) or 0.0)
        ask = float(row.get("ask", 0.0) or 0.0)
        last = float(row.get("lastPrice", 0.0) or 0.0)

        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        elif last > 0:
            return last
        else:
            return ask if ask > 0 else bid

    def _days_to_expiry(self, expiry_str: str) -> int:
        """Calculate calendar days to expiry."""
        expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
        days = (expiry_dt - datetime.now()).days
        return max(1, days)

    def get_option_greeks(
        self, expiry: str, strike: float, option_type: str = "call"
    ) -> Dict:
        """
        Compute Greeks for one option leg using mibian.

        Args:
            expiry: Expiration date as 'YYYY-MM-DD'
            strike: Strike price
            option_type: 'call' or 'put'

        Returns:
            Dict with Greeks
        """
        S = self._get_spot()
        T_days = self._days_to_expiry(expiry)
        r_percent = self.r * 100

        chain = self._get_chain(expiry)

        if option_type.lower() == "call":
            mid = self._get_mid_price(chain.calls, strike)
            try:
                iv_model = mibian.BS([S, strike, r_percent, T_days], callPrice=mid)
                iv = iv_model.impliedVolatility
            except Exception:
                iv = 0.25

            bs = mibian.BS([S, strike, r_percent, T_days], iv)
            return {
                "type": "call",
                "spot": S,
                "strike": strike,
                "expiry": expiry,
                "days_to_expiry": T_days,
                "iv": iv,
                "delta": bs.callDelta,
                "gamma": bs.gamma,
                "theta": bs.callTheta,
                "vega": bs.vega,
            }
        else:
            mid = self._get_mid_price(chain.puts, strike)
            try:
                iv_model = mibian.BS([S, strike, r_percent, T_days], putPrice=mid)
                iv = iv_model.impliedVolatility
            except Exception:
                iv = 0.25

            bs = mibian.BS([S, strike, r_percent, T_days], iv)
            return {
                "type": "put",
                "spot": S,
                "strike": strike,
                "expiry": expiry,
                "days_to_expiry": T_days,
                "iv": iv,
                "delta": bs.putDelta,
                "gamma": bs.gamma,
                "theta": bs.putTheta,
                "vega": bs.vega,
            }

    def get_spy_snapshot(self, expiry: str, put_strike: float, call_strike: float) -> Dict:
        """Get complete Greeks snapshot for SPY iron condor."""
        put_g = self.get_option_greeks(expiry, put_strike, "put")
        call_g = self.get_option_greeks(expiry, call_strike, "call")

        return {
            "spot": put_g["spot"],
            "expiry": expiry,
            "days_to_expiry": put_g["days_to_expiry"],
            "put_strike": put_strike,
            "call_strike": call_strike,
            "put_iv": put_g["iv"],
            "call_iv": call_g["iv"],
            "put_delta": -put_g["delta"],
            "put_gamma": -put_g["gamma"],
            "put_vega": -put_g["vega"],
            "put_theta": -put_g["theta"],
            "call_delta": -call_g["delta"],
            "call_gamma": -call_g["gamma"],
            "call_vega": -call_g["vega"],
            "call_theta": -call_g["theta"],
            "total_delta": -put_g["delta"] - call_g["delta"],
            "total_gamma": -put_g["gamma"] - call_g["gamma"],
            "total_vega": -put_g["vega"] - call_g["vega"],
            "total_theta": -put_g["theta"] - call_g["theta"],
        }


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

class DatabaseConnection:
    """Handle SQL Server connection and execution."""

    def __init__(self):
        self.server = os.getenv("DB_SERVER", "localhost")
        self.database = os.getenv("DB_NAME", "TradeBlog")
        self.user = os.getenv("DB_USER", "sa")
        self.password = os.getenv("DB_PASSWORD", "")
        self.connection = None

    def connect(self) -> bool:
        """Establish SQL Server connection."""
        try:
            conn_str = (
                f"Driver={{ODBC Driver 17 for SQL Server}};"
                f"Server={self.server};"
                f"Database={self.database};"
                f"UID={self.user};"
                f"PWD={self.password};"
            )
            self.connection = pyodbc.connect(conn_str)
            return True
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False

    def test_connection(self) -> bool:
        """Test database connection."""
        if not self.connect():
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            print("✅ Database connection successful!")
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return False

    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Execute SELECT query."""
        if not self.connection:
            self.connect()
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def execute_sp(self, sp_name: str, params: tuple = ()) -> List[tuple]:
        """Execute stored procedure."""
        if not self.connection:
            self.connect()
        cursor = self.connection.cursor()
        cursor.execute(f"EXEC {sp_name} {','.join(['?' for _ in params])}", params)
        self.connection.commit()
        return cursor.fetchall()

    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT and return identity using SCOPE_IDENTITY()."""
        if not self.connection:
            self.connect()
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        cursor.execute("SELECT SCOPE_IDENTITY()")
        identity = cursor.fetchone()[0]
        self.connection.commit()
        return int(identity) if identity else 0

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute UPDATE."""
        if not self.connection:
            self.connect()
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        self.connection.commit()
        return cursor.rowcount

    def close(self):
        """Close connection."""
        if self.connection:
            self.connection.close()


# ============================================================================
# TRADEBLOG MANAGER
# ============================================================================

class TradeBlogManager:
    """Main interface for TradeBlog operations."""

    def __init__(self):
        self.db = DatabaseConnection()
        self.greeks_engine = GreeksEngine(symbol="SPY", risk_free_rate=0.05)
        self.db.test_connection()

    # ========== BATCH OPERATIONS ==========

    def insert_batch(self, batch_data: Dict) -> int:
        """Insert new batch (iron condor entry)."""
        query = """
            INSERT INTO [dbo].[Trade_Batches] (
                BatchName, EntryDate, ExpirationDate,
                PutLongStrike, PutShortStrike,
                CallShortStrike, CallLongStrike,
                EntryPrice, CreditCollected, SpreadWidth,
                NumberOfSpreads, EntryIVRank, EntryVIX,
                IsActive, RollCount, CreatedAt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, GETDATE())
        """
        params = (
            batch_data["batch_name"],
            batch_data["entry_date"],
            batch_data["expiration_date"],
            batch_data["put_long_strike"],
            batch_data["put_short_strike"],
            batch_data["call_short_strike"],
            batch_data["call_long_strike"],
            batch_data["entry_price"],
            batch_data["credit_collected"],
            batch_data["spread_width"],
            batch_data["number_of_spreads"],
            batch_data["entry_iv_rank"],
            batch_data["entry_vix"],
        )
        batch_id = self.db.execute_insert(query, params)
        print(f"✅ Batch inserted: {batch_data['batch_name']} (ID: {batch_id})")
        return batch_id

    def get_batch(self, batch_id: int) -> Dict:
        """Retrieve batch details."""
        query = """
            SELECT BatchID, BatchName, EntryDate, ExpirationDate,
                   PutShortStrike, CallShortStrike, CreditCollected
            FROM [dbo].[Trade_Batches] WHERE BatchID = ?
        """
        rows = self.db.execute_query(query, (batch_id,))
        if not rows:
            return {}
        r = rows[0]
        return {
            "batch_id": r[0],
            "batch_name": r[1],
            "entry_date": r[2],
            "expiration_date": r[3],
            "put_short_strike": r[4],
            "call_short_strike": r[5],
            "credit_collected": r[6],
        }

    def get_all_active_batches(self) -> List[Dict]:
        """Get all active batches."""
        query = """
            SELECT BatchID, BatchName, PutShortStrike, CallShortStrike, ExpirationDate
            FROM [dbo].[Trade_Batches] WHERE IsActive = 1
            ORDER BY EntryDate DESC
        """
        rows = self.db.execute_query(query)
        return [
            {
                "batch_id": r[0],
                "batch_name": r[1],
                "put_strike": r[2],
                "call_strike": r[3],
                "expiry": r[4].strftime("%Y-%m-%d") if r[4] else "2026-03-07",
            }
            for r in rows
        ]

    def close_batch(self, batch_id: int, closed_price: float, reason: str = ""):
        """Close a batch."""
        query = """
            UPDATE [dbo].[Trade_Batches]
            SET IsActive = 0, ClosedDate = GETDATE(), ClosedPrice = ?, ClosedReason = ?
            WHERE BatchID = ?
        """
        self.db.execute_update(query, (closed_price, reason, batch_id))
        print(f"✅ Batch {batch_id} closed at {closed_price}")

    # ========== SNAPSHOT OPERATIONS ==========

    def insert_snapshot(self, batch_id: int, snapshot_data: Dict) -> int:
        """Insert intraday or EOD snapshot."""
        snapshot_time = snapshot_data.get("snapshot_time", datetime.now())
        is_eod = snapshot_data.get("is_eod", 0)
        snapshot_type = snapshot_data.get("snapshot_type", "INTRADAY_2H")

        query = """
            INSERT INTO [dbo].[Trade_Snapshots] (
                BatchID, SnapshotTime, IsEOD, SnapshotType,
                SPYPrice, IVRank, VIX,
                PutDelta, PutGamma, PutVega, PutTheta,
                CallDelta, CallGamma, CallVega, CallTheta,
                CurrentPnL, ProbabilityOfProfit, ExpectedValue,
                OverallStatus, RecommendedAction, RecommendationReason,
                HardStopTriggered, HardStopReason, CreatedAt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
        """
        params = (
            batch_id,
            snapshot_time,
            is_eod,
            snapshot_type,
            snapshot_data["spy_price"],
            snapshot_data["iv_rank"],
            snapshot_data.get("vix", 0.0),
            snapshot_data["put_delta"],
            snapshot_data.get("put_gamma", 0.0),
            snapshot_data.get("put_vega", 0.0),
            snapshot_data.get("put_theta", 0.0),
            snapshot_data["call_delta"],
            snapshot_data.get("call_gamma", 0.0),
            snapshot_data.get("call_vega", 0.0),
            snapshot_data.get("call_theta", 0.0),
            snapshot_data["current_pnl"],
            snapshot_data.get("probability_of_profit", 0.0),
            snapshot_data.get("expected_value", 0.0),
            snapshot_data.get("overall_status", "GREEN"),
            snapshot_data["recommended_action"],
            snapshot_data.get("recommendation_reason", ""),
            snapshot_data.get("hard_stop", False),
            snapshot_data.get("hard_stop_reason", ""),
        )
        snapshot_id = self.db.execute_insert(query, params)
        eod_flag = " (EOD)" if is_eod else ""
        print(f"✅ Snapshot captured (Batch {batch_id}, ID: {snapshot_id}){eod_flag}")
        return snapshot_id

    def get_latest_snapshot(self, batch_id: int) -> Dict:
        """Get most recent snapshot for batch."""
        query = """
            SELECT TOP 1 SnapshotID, SPYPrice, CurrentPnL, OverallStatus
            FROM [dbo].[Trade_Snapshots]
            WHERE BatchID = ?
            ORDER BY SnapshotTime DESC
        """
        rows = self.db.execute_query(query, (batch_id,))
        if not rows:
            return {}
        r = rows[0]
        return {
            "snapshot_id": r[0],
            "spy_price": r[1],
            "current_pnl": r[2],
            "status": r[3],
        }

    def get_batch_snapshots(self, batch_id: int, limit: int = 100) -> List[Dict]:
        """Get all snapshots for batch."""
        query = f"""
            SELECT TOP {limit} SnapshotTime, SPYPrice, CurrentPnL, OverallStatus, IsEOD
            FROM [dbo].[Trade_Snapshots]
            WHERE BatchID = ?
            ORDER BY SnapshotTime DESC
        """
        rows = self.db.execute_query(query, (batch_id,))
        return [
            {
                "snapshot_time": r[0],
                "spy_price": r[1],
                "current_pnl": r[2],
                "status": r[3],
                "is_eod": r[4],
            }
            for r in rows
        ]

    # ========== HARD STOP OPERATIONS ==========

    def log_hard_stop(
        self,
        batch_id: int,
        stop_type: str,
        stop_value: float,
        spy_price: float,
        notes: str = "",
    ):
        """Log hard stop event."""
        query = """
            INSERT INTO [dbo].[Trade_HardStopLog] (
                BatchID, StopType, StopValue, SPYPriceAtTrigger, Notes, TriggeredAt
            ) VALUES (?, ?, ?, ?, ?, GETDATE())
        """
        self.db.execute_insert(query, (batch_id, stop_type, stop_value, spy_price, notes))
        print(f"🛑 Hard stop logged: {stop_type} on Batch {batch_id}")

    def get_hard_stops(self, batch_id: int) -> List[Dict]:
        """Get all hard stops for batch."""
        query = """
            SELECT StopType, TriggeredAt, StopValue, Notes
            FROM [dbo].[Trade_HardStopLog]
            WHERE BatchID = ?
            ORDER BY TriggeredAt DESC
        """
        rows = self.db.execute_query(query, (batch_id,))
        return [
            {
                "stop_type": r[0],
                "triggered_at": r[1],
                "stop_value": r[2],
                "notes": r[3],
            }
            for r in rows
        ]

    # ========== TRADE ACTION OPERATIONS ==========

    def log_trade_action(
        self,
        batch_id: int,
        action_type: str,
        spy_price: float,
        notes: str = "",
    ):
        """Log trade action (ENTERED, HOLD, ROLLED, CLOSED)."""
        query = """
            INSERT INTO [dbo].[Trade_Actions] (
                BatchID, ActionType, ActionDate, SPYPriceAtAction, Notes
            ) VALUES (?, ?, GETDATE(), ?, ?)
        """
        self.db.execute_insert(query, (batch_id, action_type, spy_price, notes))
        print(f"📝 Action logged: {action_type} on Batch {batch_id}")

    def get_trade_actions(self, batch_id: int) -> List[Dict]:
        """Get all actions for batch."""
        query = """
            SELECT ActionType, ActionDate, SPYPriceAtAction, Notes
            FROM [dbo].[Trade_Actions]
            WHERE BatchID = ?
            ORDER BY ActionDate DESC
        """
        rows = self.db.execute_query(query, (batch_id,))
        return [
            {
                "action_type": r[0],
                "action_date": r[1],
                "spy_price": r[2],
                "notes": r[3],
            }
            for r in rows
        ]

    # ========== ROLL OPERATIONS ==========

    def record_roll(self, roll_data: Dict) -> int:
        """Record roll transaction."""
        query = """
            INSERT INTO [dbo].[Trade_RolledPositions] (
                OriginalBatchID, RollDate, ClosedSide,
                ClosedAtPrice, ClosedAtLoss,
                NewCreditCollected, RollingCost, RollReason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            roll_data["original_batch_id"],
            roll_data.get("roll_date", datetime.now()),
            roll_data["closed_side"],
            roll_data["closed_at_price"],
            roll_data["closed_at_loss"],
            roll_data["new_credit_collected"],
            roll_data["rolling_cost"],
            roll_data["roll_reason"],
        )
        roll_id = self.db.execute_insert(query, params)
        print(f"🔄 Roll recorded (ID: {roll_id})")
        return roll_id

    def get_rolls(self, batch_id: int) -> List[Dict]:
        """Get all rolls for batch."""
        query = """
            SELECT RollID, RollDate, ClosedSide, ClosedAtPrice, RollingCost
            FROM [dbo].[Trade_RolledPositions]
            WHERE OriginalBatchID = ?
            ORDER BY RollDate DESC
        """
        rows = self.db.execute_query(query, (batch_id,))
        return [
            {
                "roll_id": r[0],
                "roll_date": r[1],
                "closed_side": r[2],
                "closed_at_price": r[3],
                "rolling_cost": r[4],
            }
            for r in rows
        ]

    # ========== PORTFOLIO & ANALYTICS ==========

    def get_portfolio_summary(self, date_str: str = None) -> Dict:
        """Get portfolio summary for given date."""
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        query = """
            SELECT
                COUNT(DISTINCT b.BatchID) AS active_batches,
                COALESCE(SUM(ts.CurrentPnL), 0) AS total_pnl,
                COALESCE(AVG(ts.ProbabilityOfProfit), 0) AS avg_pop,
                COALESCE(SUM(ts.CallDelta + ts.PutDelta), 0) AS total_delta,
                COALESCE(SUM(ts.CallTheta + ts.PutTheta), 0) AS daily_theta
            FROM [dbo].[Trade_Batches] b
            LEFT JOIN [dbo].[Trade_Snapshots] ts ON b.BatchID = ts.BatchID
            WHERE b.IsActive = 1 AND CAST(ts.SnapshotTime AS DATE) = ?
        """
        rows = self.db.execute_query(query, (date_str,))
        if rows:
            r = rows[0]
            return {
                "active_batches": r[0] or 0,
                "total_pnl": r[1] or 0.0,
                "avg_pop": r[2] or 0.0,
                "portfolio_delta": r[3] or 0.0,
                "daily_theta": r[4] or 0.0,
            }
        return {
            "active_batches": 0,
            "total_pnl": 0.0,
            "avg_pop": 0.0,
            "portfolio_delta": 0.0,
            "daily_theta": 0.0,
        }

    def get_performance_statistics(self, batch_id: int) -> Dict:
        """Get performance stats for batch."""
        query = """
            SELECT
                COUNT(*) AS total_snapshots,
                MAX(CurrentPnL) AS max_pnl,
                MIN(CurrentPnL) AS min_pnl,
                AVG(ProbabilityOfProfit) AS avg_pop
            FROM [dbo].[Trade_Snapshots]
            WHERE BatchID = ?
        """
        rows = self.db.execute_query(query, (batch_id,))
        rolls = self.get_rolls(batch_id)

        if rows and rows[0][0] is not None:
            r = rows[0]
            return {
                "total_snapshots": r[0] or 0,
                "current_pnl": r[1] or 0.0,
                "max_pnl": r[1] or 0.0,
                "min_pnl": r[2] or 0.0,
                "avg_pop": r[3] or 0.0,
                "total_rolls": len(rolls),
            }
        return {
            "total_snapshots": 0,
            "current_pnl": 0.0,
            "max_pnl": 0.0,
            "min_pnl": 0.0,
            "avg_pop": 0.0,
            "total_rolls": 0,
        }

    # ========== EXPORT OPERATIONS ==========

    def export_journal_to_csv(self, batch_id: int, output_dir: str = "exports") -> Optional[str]:
        """Export trade journal to CSV with proper error handling."""
        Path(output_dir).mkdir(exist_ok=True)

        query = """
            SELECT
                SnapshotTime, SPYPrice, CurrentPnL, ProbabilityOfProfit,
                PutDelta, CallDelta, RecommendedAction, OverallStatus
            FROM [dbo].[Trade_Snapshots]
            WHERE BatchID = ?
            ORDER BY SnapshotTime
        """
        rows = self.db.execute_query(query, (batch_id,))

        # FIX: Check if data exists before creating DataFrame
        if not rows:
            print(f"⚠️  No snapshots found for batch {batch_id}")
            return None

        df = pd.DataFrame(
            rows,
            columns=[
                "Time",
                "SPY Price",
                "P&L",
                "POP",
                "Put Delta",
                "Call Delta",
                "Action",
                "Status",
            ],
        )

        filename = (
            f"{output_dir}/batch_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df.to_csv(filename, index=False)
        print(f"✅ Journal exported: {filename}")
        return filename

    def export_journal_to_excel(
        self, batch_id: int, output_dir: str = "exports"
    ) -> Optional[str]:
        """Export trade journal to Excel."""
        Path(output_dir).mkdir(exist_ok=True)

        query = """
            SELECT
                SnapshotTime, SPYPrice, CurrentPnL, ProbabilityOfProfit,
                PutDelta, CallDelta, RecommendedAction, OverallStatus
            FROM [dbo].[Trade_Snapshots]
            WHERE BatchID = ?
            ORDER BY SnapshotTime
        """
        rows = self.db.execute_query(query, (batch_id,))

        # FIX: Check if data exists
        if not rows:
            print(f"⚠️  No snapshots found for batch {batch_id}")
            return None

        df = pd.DataFrame(
            rows,
            columns=[
                "Time",
                "SPY Price",
                "P&L",
                "POP",
                "Put Delta",
                "Call Delta",
                "Action",
                "Status",
            ],
        )

        filename = (
            f"{output_dir}/batch_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        df.to_excel(filename, index=False)
        print(f"✅ Journal exported: {filename}")
        return filename

    def generate_trade_journal(self, batch_id: int) -> pd.DataFrame:
        """Get complete trade journal as DataFrame."""
        query = """
            SELECT
                SnapshotTime, SPYPrice, CurrentPnL, ProbabilityOfProfit,
                PutDelta, CallDelta, PutTheta, CallTheta, RecommendedAction
            FROM [dbo].[Trade_Snapshots]
            WHERE BatchID = ?
            ORDER BY SnapshotTime
        """
        rows = self.db.execute_query(query, (batch_id,))
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(
            rows,
            columns=[
                "SnapshotTime",
                "SPYPrice",
                "CurrentPnL",
                "POP",
                "PutDelta",
                "CallDelta",
                "PutTheta",
                "CallTheta",
                "Action",
            ],
        )


# ============================================================================
# INTRADAY CAPTURE (Single Batch)
# ============================================================================

class IntraDayCapture:
    """Automated intraday snapshot capture for single batch (legacy)."""

    def __init__(self, manager: TradeBlogManager):
        self.manager = manager

    def capture_intraday(self, batch_id: int, position_data: Dict) -> int:
        """Capture intraday snapshot for single batch."""
        snapshot_data = {
            "snapshot_time": datetime.now(),
            "is_eod": 0,
            "snapshot_type": "INTRADAY_2H",
            **position_data,
        }
        return self.manager.insert_snapshot(batch_id, snapshot_data)

    def capture_eod(self, batch_id: int, position_data: Dict) -> int:
        """Capture end-of-day snapshot for single batch."""
        snapshot_data = {
            "snapshot_time": datetime.now(),
            "is_eod": 1,
            "snapshot_type": "EOD",
            **position_data,
        }
        return self.manager.insert_snapshot(batch_id, snapshot_data)


# ============================================================================
# AUTOMATED DAILY CAPTURE (Multiple Batches) - NEW
# ============================================================================

class AutomatedDailyCapture:
    """
    ENHANCED: Automated capture for ALL active batches.
    Handles multiple iron condors simultaneously (Week 1, Week 2, etc.)
    """

    def __init__(self, manager: TradeBlogManager):
        self.manager = manager

    def capture_all_intraday(self) -> Dict:
        """
        Capture intraday snapshots for ALL active batches.
        Each batch gets its own snapshot with correct BatchID reference.
        
        Returns:
            Dict with results for each batch
        """
        batches = self.manager.get_all_active_batches()

        if not batches:
            print("⚠️  No active batches to capture")
            return {}

        results = {}

        print(f"\n📊 Intraday Capture ({datetime.now().strftime('%H:%M:%S')})")
        print(f"   Processing {len(batches)} active batch(es)...\n")

        for batch in batches:
            batch_id = batch["batch_id"]
            batch_name = batch["batch_name"]
            put_strike = batch["put_strike"]
            call_strike = batch["call_strike"]
            expiry = batch["expiry"]

            try:
                # Get Greeks for this specific batch
                spy_snap = self.manager.greeks_engine.get_spy_snapshot(
                    expiry=expiry,
                    put_strike=put_strike,
                    call_strike=call_strike,
                )

                # Calculate metrics
                pop = MetricsCalculator.calculate_pop(
                    spy_snap["put_delta"],
                    spy_snap["call_delta"],
                )

                ev = MetricsCalculator.calculate_ev(pop, 140.0, -360.0)
                status = MetricsCalculator.calculate_status(
                    spy_snap["put_delta"],
                    spy_snap["call_delta"],
                )

                action, reason = DecisionEngine.recommend_action(
                    spy_snap["put_delta"],
                    spy_snap["call_delta"],
                    spy_snap["days_to_expiry"],
                    current_pnl=35.0,
                )

                # Build position data
                position_data = {
                    "spy_price": spy_snap["spot"],
                    "iv_rank": 58.0,
                    "vix": 16.5,
                    "put_delta": spy_snap["put_delta"],
                    "put_gamma": spy_snap["put_gamma"],
                    "put_vega": spy_snap["put_vega"],
                    "put_theta": spy_snap["put_theta"],
                    "call_delta": spy_snap["call_delta"],
                    "call_gamma": spy_snap["call_gamma"],
                    "call_vega": spy_snap["call_vega"],
                    "call_theta": spy_snap["call_theta"],
                    "current_pnl": 35.0,
                    "probability_of_profit": pop,
                    "expected_value": ev,
                    "overall_status": status,
                    "recommended_action": action,
                    "recommendation_reason": reason,
                    "hard_stop": False,
                }

                # Capture intraday (is_eod=0)
                snapshot_id = self.manager.insert_snapshot(batch_id, position_data)

                results[batch_name] = {
                    "status": "✅",
                    "snapshot_id": snapshot_id,
                    "action": action,
                    "reason": reason,
                    "pop": pop,
                    "ev": ev,
                }

                print(f"  ✅ {batch_name}: {action} (POP: {pop:.1f}%, EV: ${ev:.2f})")

            except Exception as e:
                results[batch_name] = {
                    "status": "❌",
                    "error": str(e),
                }
                print(f"  ❌ {batch_name}: {str(e)}")

        return results

    def capture_all_eod(self) -> Dict:
        """
        Capture end-of-day snapshots for ALL active batches.
        Final snapshot of the day, marked with is_eod=1.
        
        Returns:
            Dict with results for each batch
        """
        batches = self.manager.get_all_active_batches()

        if not batches:
            print("⚠️  No active batches to capture")
            return {}

        results = {}

        print(f"\n📊 EOD Capture ({datetime.now().strftime('%H:%M:%S')})")
        print(f"   Processing {len(batches)} active batch(es)...\n")

        for batch in batches:
            batch_id = batch["batch_id"]
            batch_name = batch["batch_name"]
            put_strike = batch["put_strike"]
            call_strike = batch["call_strike"]
            expiry = batch["expiry"]

            try:
                # Get final Greeks
                spy_snap = self.manager.greeks_engine.get_spy_snapshot(
                    expiry=expiry,
                    put_strike=put_strike,
                    call_strike=call_strike,
                )

                # Calculate metrics
                pop = MetricsCalculator.calculate_pop(
                    spy_snap["put_delta"],
                    spy_snap["call_delta"],
                )
                ev = MetricsCalculator.calculate_ev(pop, 140.0, -360.0)
                status = MetricsCalculator.calculate_status(
                    spy_snap["put_delta"],
                    spy_snap["call_delta"],
                )
                action, reason = DecisionEngine.recommend_action(
                    spy_snap["put_delta"],
                    spy_snap["call_delta"],
                    spy_snap["days_to_expiry"],
                    35.0,
                )

                position_data = {
                    "spy_price": spy_snap["spot"],
                    "iv_rank": 58.0,
                    "vix": 16.5,
                    "put_delta": spy_snap["put_delta"],
                    "put_gamma": spy_snap["put_gamma"],
                    "put_vega": spy_snap["put_vega"],
                    "put_theta": spy_snap["put_theta"],
                    "call_delta": spy_snap["call_delta"],
                    "call_gamma": spy_snap["call_gamma"],
                    "call_vega": spy_snap["call_vega"],
                    "call_theta": spy_snap["call_theta"],
                    "current_pnl": 35.0,
                    "probability_of_profit": pop,
                    "expected_value": ev,
                    "overall_status": status,
                    "recommended_action": action,
                    "recommendation_reason": reason,
                    "hard_stop": False,
                    "is_eod": 1,  # KEY: Mark as EOD
                    "snapshot_type": "EOD",
                }

                # Capture EOD (is_eod=1)
                snapshot_id = self.manager.insert_snapshot(batch_id, position_data)

                results[batch_name] = {
                    "status": "✅",
                    "snapshot_id": snapshot_id,
                    "eod_action": action,
                }

                print(f"  ✅ {batch_name}: EOD Snapshot {snapshot_id} - {action}")

            except Exception as e:
                results[batch_name] = {
                    "status": "❌",
                    "error": str(e),
                }
                print(f"  ❌ {batch_name}: {str(e)}")

        return results


# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """Calculate trading metrics (POP, EV, status)."""

    @staticmethod
    def calculate_pop(put_delta: float, call_delta: float) -> float:
        """Calculate Probability of Profit."""
        return (1 - abs(put_delta)) * (1 - abs(call_delta)) * 100

    @staticmethod
    def calculate_ev(pop: float, max_profit: float, max_loss: float) -> float:
        """Calculate Expected Value."""
        pop_decimal = pop / 100
        return (pop_decimal * max_profit) - ((1 - pop_decimal) * abs(max_loss))

    @staticmethod
    def calculate_status(put_delta: float, call_delta: float) -> str:
        """Determine batch status."""
        if abs(put_delta) > 0.35 or abs(call_delta) > 0.35:
            return "RED"
        elif abs(put_delta) > 0.25 or abs(call_delta) > 0.25:
            return "YELLOW"
        return "GREEN"


# ============================================================================
# DECISION ENGINE
# ============================================================================

class DecisionEngine:
    """Generate trading recommendations based on batch metrics."""

    @staticmethod
    def recommend_action(
        put_delta: float,
        call_delta: float,
        days_to_expiry: int,
        current_pnl: float,
        max_loss: float = -360.0,
    ) -> Tuple[str, str]:
        """Generate action recommendation."""
        if current_pnl <= max_loss:
            return "CLOSE", f"Max loss (-${abs(max_loss)}) hit"

        if days_to_expiry > 10:
            if abs(put_delta) > 0.35:
                return "ROLL", f"Put ITM (delta {put_delta:.2f}), days {days_to_expiry}"
            if abs(call_delta) > 0.35:
                return "ROLL", f"Call ITM (delta {call_delta:.2f}), days {days_to_expiry}"

        return "HOLD", "Let theta work"


# ============================================================================
# MAIN USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize
    manager = TradeBlogManager()

    # Example: Create batch
    batch_data = {
        "batch_name": "BATCH_1",
        "entry_date": datetime(2026, 1, 22).date(),
        "expiration_date": datetime(2026, 3, 7).date(),
        "put_long_strike": 665.0,
        "put_short_strike": 670.0,
        "call_short_strike": 710.0,
        "call_long_strike": 715.0,
        "entry_price": 689.50,
        "credit_collected": 140.0,
        "spread_width": 500.0,
        "number_of_spreads": 20,
        "entry_iv_rank": 58.0,
        "entry_vix": 16.5,
    }
    batch_id = manager.insert_batch(batch_data)

    # Example: Capture using AUTOMATED multi-batch system
    try:
        capture = AutomatedDailyCapture(manager)
        
        # Intraday capture (10 AM, 12 PM, 2 PM)
        print("\n=== INTRADAY CAPTURE ===")
        results = capture.capture_all_intraday()
        
        # EOD capture (4:05 PM)
        print("\n=== EOD CAPTURE ===")
        eod_results = capture.capture_all_eod()
        
        # Export
        print("\n=== EXPORTS ===")
        csv_file = manager.export_journal_to_csv(batch_id)
        if csv_file:
            print(f"✅ Journal: {csv_file}")
        else:
            print("⚠️  No data to export")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
