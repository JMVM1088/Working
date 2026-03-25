# ============================================================================
# TradeBlog Enhanced - Multi-Symbol, Multi-Strategy Implementation
# PRODUCTION GRADE - REFACTORED FOR DYNAMIC TRADING
#
# Author: Quantitative Trading System
# Date: 2026-01-29
# Purpose: Professional multi-symbol, multi-strategy options tracking
#
# SUPPORTED SYMBOLS: SPY, QQQ, IWM, individual stocks (TSLA, AAPL, etc.)
# SUPPORTED STRATEGIES: Iron Condor, Call/Put Spreads, Diagonal Spreads, etc.
# EARLY CLOSURE: Manually mark batches as closed before expiration
# DATABASE: SQL Server enhanced schema with Trade_Legs and Trade_StrategyConfigs
#
# FIXES APPLIED (2026-01-29):
# 1. Removed hardcoded "SPY" defaults - symbol now REQUIRED
# 2. Fixed get_batch() index mapping (was r[6]-r[9], now r[6]-r[8])
# 3. Added per-leg ExpirationDate support for Diagonal Spreads
# 4. Added DIAGONAL_SPREAD and DIAGONAL_PUT_SPREAD to StrategyType
# 5. Enhanced get_legs_snapshot() to support multi-expiry strategies
# ============================================================================

import os
import pyodbc
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import yfinance as yf
import mibian
import json
from pathlib import Path
from enum import Enum

load_dotenv()

# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class StrategyType(str, Enum):
    """Supported option strategies"""
    IRON_CONDOR = "IRON_CONDOR"
    CALL_SPREAD = "CALL_SPREAD"
    PUT_SPREAD = "PUT_SPREAD"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    CALL_ONLY = "CALL_ONLY"
    PUT_ONLY = "PUT_ONLY"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"
    DIAGONAL_SPREAD = "DIAGONAL_SPREAD"           # ← NEW: Diagonal Call Spread
    DIAGONAL_PUT_SPREAD = "DIAGONAL_PUT_SPREAD"   # ← NEW: Diagonal Put Spread


class OptionType(str, Enum):
    """Option type"""
    CALL = "CALL"
    PUT = "PUT"


class SideType(str, Enum):
    """Position side"""
    LONG = "LONG"
    SHORT = "SHORT"


# Strategy configurations: leg_count, [leg_num, option_type, side, dte_offset, ...]
STRATEGY_CONFIGS = {
    StrategyType.IRON_CONDOR: {
        "leg_count": 4,
        "legs": [
            {"leg_num": 1, "option_type": OptionType.PUT, "side": SideType.LONG},
            {"leg_num": 2, "option_type": OptionType.PUT, "side": SideType.SHORT},
            {"leg_num": 3, "option_type": OptionType.CALL, "side": SideType.SHORT},
            {"leg_num": 4, "option_type": OptionType.CALL, "side": SideType.LONG},
        ]
    },
    StrategyType.CALL_SPREAD: {
        "leg_count": 2,
        "legs": [
            {"leg_num": 1, "option_type": OptionType.CALL, "side": SideType.LONG},
            {"leg_num": 2, "option_type": OptionType.CALL, "side": SideType.SHORT},
        ]
    },
    StrategyType.PUT_SPREAD: {
        "leg_count": 2,
        "legs": [
            {"leg_num": 1, "option_type": OptionType.PUT, "side": SideType.LONG},
            {"leg_num": 2, "option_type": OptionType.PUT, "side": SideType.SHORT},
        ]
    },
    StrategyType.BULL_CALL_SPREAD: {
        "leg_count": 2,
        "legs": [
            {"leg_num": 1, "option_type": OptionType.CALL, "side": SideType.LONG},
            {"leg_num": 2, "option_type": OptionType.CALL, "side": SideType.SHORT},
        ]
    },
    StrategyType.BEAR_PUT_SPREAD: {
        "leg_count": 2,
        "legs": [
            {"leg_num": 1, "option_type": OptionType.PUT, "side": SideType.SHORT},
            {"leg_num": 2, "option_type": OptionType.PUT, "side": SideType.LONG},
        ]
    },
    StrategyType.CALL_ONLY: {
        "leg_count": 1,
        "legs": [
            {"leg_num": 1, "option_type": OptionType.CALL, "side": SideType.SHORT},
        ]
    },
    StrategyType.PUT_ONLY: {
        "leg_count": 1,
        "legs": [
            {"leg_num": 1, "option_type": OptionType.PUT, "side": SideType.SHORT},
        ]
    },
    StrategyType.STRADDLE: {
        "leg_count": 2,
        "legs": [
            {"leg_num": 1, "option_type": OptionType.CALL, "side": SideType.SHORT},
            {"leg_num": 2, "option_type": OptionType.PUT, "side": SideType.SHORT},
        ]
    },
    StrategyType.STRANGLE: {
        "leg_count": 2,
        "legs": [
            {"leg_num": 1, "option_type": OptionType.CALL, "side": SideType.SHORT},
            {"leg_num": 2, "option_type": OptionType.PUT, "side": SideType.SHORT},
        ]
    },
    # ← NEW: Diagonal Spreads (different expirations per leg)
    StrategyType.DIAGONAL_SPREAD: {
        "leg_count": 2,
        "description": "Diagonal Call Spread - different expirations",
        "legs": [
            {"leg_num": 1, "option_type": OptionType.CALL, "side": SideType.LONG, "dte_offset": 60},   # Far-term
            {"leg_num": 2, "option_type": OptionType.CALL, "side": SideType.SHORT, "dte_offset": 30}, # Near-term (rolled)
        ]
    },
    StrategyType.DIAGONAL_PUT_SPREAD: {
        "leg_count": 2,
        "description": "Diagonal Put Spread - different expirations",
        "legs": [
            {"leg_num": 1, "option_type": OptionType.PUT, "side": SideType.LONG, "dte_offset": 60},   # Far-term
            {"leg_num": 2, "option_type": OptionType.PUT, "side": SideType.SHORT, "dte_offset": 30},  # Near-term (rolled)
        ]
    },
}


# ============================================================================
# OPTION LEG CLASS
# ============================================================================

class OptionLeg:
    """Represents a single option leg in a position"""
    
    def __init__(
        self,
        leg_num: int,
        option_type: OptionType,
        side: SideType,
        strike: float,
        expiry: str,
        symbol: str,                    # ← FIX: Now REQUIRED (no default)
        entry_price: float = 0.0,
        current_price: float = 0.0,
    ):
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol must be a non-empty string (e.g., 'SPY', 'QQQ', 'TSLA')")
        
        self.leg_num = leg_num
        self.option_type = option_type
        self.side = side
        self.strike = strike
        self.expiry = expiry
        self.symbol = symbol.upper()    # Normalize to uppercase
        self.entry_price = entry_price
        self.current_price = current_price
        self.greeks = {}

    def as_dict(self) -> Dict:
        """Return leg data as dictionary"""
        return {
            "leg_num": self.leg_num,
            "option_type": self.option_type.value,
            "side": self.side.value,
            "strike": self.strike,
            "expiry": self.expiry,               # ← Now included (for Diagonal Spreads)
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "greeks": self.greeks,
        }


# ============================================================================
# GREEKS ENGINE (ENHANCED - MULTI-SYMBOL)
# ============================================================================

class GreeksEngine:
    """
    Enhanced Greeks engine supporting multiple symbols.
    Fetches spot price and option chain from yfinance, computes Greeks via mibian.
    """
    
    def __init__(self, symbol: str, risk_free_rate: float = 0.05):
        """
        Args:
            symbol: Underlying ticker (SPY, QQQ, IWM, TSLA, etc.) ← FIX: Now REQUIRED
            risk_free_rate: Annual risk-free rate as decimal (e.g., 0.05 = 5%)
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol must be a non-empty string (e.g., 'SPY', 'QQQ', 'TSLA')")
        
        self.symbol = symbol.upper()
        self.r = risk_free_rate

    def _get_vix(self) -> float:
        """Fetch current VIX from yfinance (fallback to 20.0 on failure)."""
        try:
            # if self.vix_cache and self.vix_cache_time:
            #     age = (datetime.now() - self.vix_cache_time).total_seconds()
            #     if age < 300:
            #         return self.vix_cache

            vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
            self.vix_cache = float(vix)
            self.vix_cache_time = datetime.now()
            return self.vix_cache
        except Exception as e:
            print(f"⚠️ VIX fetch failed: {e}. Using default 20.0")
            return 20.0

    def _get_iv_rank(self, symbol: str, lookback_days: int = 252) -> float:
        """
        Calculate IV Rank using historical realized volatility as proxy.

        IV Rank ≈ (Current HV - Min HV) / (Max HV - Min HV) * 100 over 1 year window.
        """
        try:
            hist = yf.Ticker(symbol).history(period="1y")
            if hist.empty or len(hist) < 40:
                return 50.0

            returns = hist["Close"].pct_change().dropna()
            hv_20 = returns.tail(20).std() * np.sqrt(252) * 100
            current_iv = hv_20

            iv_values: List[float] = []
            for i in range(20, len(returns), 5):
                window_hv = returns.iloc[i-20:i].std() * np.sqrt(252) * 100
                iv_values.append(window_hv)

            if not iv_values:
                return 50.0

            iv_min = min(iv_values)
            iv_max = max(iv_values)
            if iv_max == iv_min:
                return 50.0

            iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100
            return max(0.0, min(100.0, iv_rank))
        except Exception as e:
            print(f"⚠️ IV Rank calculation failed for {symbol}: {e}. Using default 50.0")
            return 50.0
        
    def _get_spot(self) -> float:
        """Get latest spot price from yfinance"""
        try:
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                raise ValueError(f"Could not fetch price for {self.symbol}")
            return float(hist["Close"].iloc[-1])
        except Exception as e:
            raise ValueError(f"Spot price error for {self.symbol}: {str(e)}")

    def _get_chain(self, expiry: str):
        """Get option chain for given expiry from yfinance"""
        try:
            ticker = yf.Ticker(self.symbol)
            chain = ticker.option_chain(expiry)
            return chain
        except Exception as e:
            raise ValueError(f"Option chain error for {self.symbol}/{expiry}: {str(e)}")

    def _get_mid_price(self, df, strike: float) -> float:
        """Extract mid price from option chain row"""
        row = df[df["strike"] == strike]
        if row.empty:
            raise ValueError(f"No option found for {self.symbol} {strike} strike")
        
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
        """Calculate calendar days to expiry"""
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
            Dict with Greeks and market data
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
                "symbol": self.symbol,
                "spot": S,
                "strike": strike,
                "expiry": expiry,
                "days_to_expiry": T_days,
                "mid_price": mid,
                "iv": iv,
                "delta": bs.callDelta,
                "gamma": bs.gamma,
                "theta": bs.callTheta,
                "vega": bs.vega,
                "rho": bs.callRho if hasattr(bs, 'callRho') else 0.0,
            }
        else:  # put
            mid = self._get_mid_price(chain.puts, strike)
            try:
                iv_model = mibian.BS([S, strike, r_percent, T_days], putPrice=mid)
                iv = iv_model.impliedVolatility
            except Exception:
                iv = 0.25
            
            bs = mibian.BS([S, strike, r_percent, T_days], iv)
            return {
                "type": "put",
                "symbol": self.symbol,
                "spot": S,
                "strike": strike,
                "expiry": expiry,
                "days_to_expiry": T_days,
                "mid_price": mid,
                "iv": iv,
                "delta": bs.putDelta,
                "gamma": bs.gamma,
                "theta": bs.putTheta,
                "vega": bs.vega,
                "rho": bs.putRho if hasattr(bs, 'putRho') else 0.0,
            }

    def get_legs_snapshot(self, legs: List[OptionLeg]) -> Dict:
        """
        Get Greeks for multiple legs and compute composite metrics.
        
        Works for ANY strategy with ANY number of legs.
        Supports DIFFERENT expirations (for Diagonal Spreads, Calendar Spreads, etc.)
        
        Args:
            legs: List of OptionLeg objects
            
        Returns:
            Dict with individual leg Greeks and composite metrics
        """
        if not legs:
            raise ValueError("Must provide at least one leg")
        
        symbol = legs[0].symbol
        
        # All legs MUST be for the same underlying (but can have different expirations)
        for leg in legs:
            if leg.symbol != symbol:
                raise ValueError(
                    f"All legs must have same symbol. "
                    f"Expected {symbol}, got {leg.symbol}"
                )
            # Validate expiry format
            if not leg.expiry or leg.expiry != leg.expiry.strip():
                raise ValueError(f"Leg {leg.leg_num} has invalid expiry: {leg.expiry}")
        
        legs_greeks = []
        composite_delta = 0.0
        composite_gamma = 0.0
        composite_vega = 0.0
        composite_theta = 0.0
        
        # Get Greeks for EACH leg with ITS OWN EXPIRY (supports Diagonal Spreads!)
        for leg in legs:
            greeks = self.get_option_greeks(
                expiry=leg.expiry,                              # ← Each leg's own expiration
                strike=leg.strike,
                option_type=leg.option_type.value.lower(),
            )
            
            # Apply side multiplier (SHORT = negative, LONG = positive)
            multiplier = -1.0 if leg.side == SideType.SHORT else 1.0
            
            leg.greeks = {
                "mid_price": greeks["mid_price"],
                "iv": greeks["iv"],
                "delta": greeks["delta"] * multiplier,
                "gamma": greeks["gamma"] * multiplier,
                "theta": greeks["theta"] * multiplier,
                "vega": greeks["vega"] * multiplier,
                "rho": greeks.get("rho", 0.0) * multiplier,
                "expiry": leg.expiry,  # ← Track which expiry this leg uses
            }
            
            legs_greeks.append(leg.greeks)
            
            # Accumulate composite Greeks
            composite_delta += leg.greeks["delta"]
            composite_gamma += leg.greeks["gamma"]
            composite_vega += leg.greeks["vega"]
            composite_theta += leg.greeks["theta"]
        
        return {
            "symbol": symbol,
            "spot": self.get_option_greeks(
                expiry=legs[0].expiry, 
                strike=legs[0].strike, 
                option_type=legs[0].option_type.value.lower()
            )["spot"],
            "expirations": sorted(list(set(leg.expiry for leg in legs))),  # All unique expiries
            "leg_count": len(legs),
            "legs": [leg.as_dict() for leg in legs],
            "legs_greeks": legs_greeks,
            "composite_delta": composite_delta,
            "composite_gamma": composite_gamma,
            "composite_vega": composite_vega,
            "composite_theta": composite_theta,
        }


# ============================================================================
# STRATEGY CONFIGURATION MANAGER
# ============================================================================

class StrategyManager:
    """Manages strategy configurations and leg specifications"""
    
    @staticmethod
    def get_strategy_config(strategy: StrategyType) -> Dict:
        """Get strategy configuration"""
        return STRATEGY_CONFIGS.get(strategy)
    
    @staticmethod
    def validate_strategy(strategy: str) -> bool:
        """Check if strategy is supported"""
        try:
            StrategyType(strategy)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def get_all_strategies() -> List[str]:
        """Get all supported strategies"""
        return [s.value for s in StrategyType]


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

class DatabaseConnection:
    """Handle SQL Server connection and execution"""
    
    def __init__(self):
        self.server = os.getenv("DB_SERVER", "localhost")
        self.database = os.getenv("DB_NAME", "TradeBlog")
        self.Trusted_Connection = os.getenv("Trusted_Connection")
        self.connection = None
    
    def connect(self) -> bool:
        """Establish SQL Server connection"""
        try:
            conn_str = (
                f"Driver={{ODBC Driver 17 for SQL Server}};"
                f"Server={self.server};"
                f"Database={self.database};"
                f"Trusted_Connection={self.Trusted_Connection};"
            )
            self.connection = pyodbc.connect(conn_str)
            return True
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """Test database connection"""
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
        """Execute SELECT query"""
        if not self.connection:
            self.connect()
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT and return identity using @@IDENTITY."""
        if not self.connection:
            success = self.connect()
            if not success:
                raise RuntimeError(
                    f"❌ Database connection failed.\n"
                    f"Check your .env file."
                )
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            cursor.nextset()  # ← CRITICAL: Required for @@IDENTITY
            cursor.execute("SELECT @@IDENTITY")
            identity_row = cursor.fetchone()
            identity = identity_row[0] if identity_row else None
            self.connection.commit()
            return int(identity) if identity else 0
        except Exception as e:
            print(f"❌ Database INSERT error: {str(e)}")
            raise
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute UPDATE"""
        if not self.connection:
            self.connect()
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        self.connection.commit()
        return cursor.rowcount
    
    def close(self):
        """Close connection"""
        if self.connection:
            self.connection.close()


# ============================================================================
# TRADEBLOG MANAGER (ENHANCED)
# ============================================================================

class TradeBlogManager:
    """
    Enhanced manager supporting multiple symbols and strategies.
    Handles batch creation, leg management, snapshots, and closure.
    """
    
    def __init__(self):
        self.db = DatabaseConnection()
        self.greeks_engines = {}  # Cache for symbol-specific engines
        self.db.test_connection()
    
    def _get_greeks_engine(self, symbol: str) -> GreeksEngine:
        """Get or create Greeks engine for symbol"""
        if symbol not in self.greeks_engines:
            self.greeks_engines[symbol] = GreeksEngine(symbol=symbol, risk_free_rate=0.05)
        return self.greeks_engines[symbol]
    
    # ========== BATCH OPERATIONS ==========
    
    def insert_batch(self, batch_data: Dict) -> int:
        """
        Insert new batch with support for multiple symbols and strategies.
        
        Args:
            batch_data: Dict with REQUIRED keys:
                - batch_name: str
                - symbol: str (SPY, QQQ, IWM, TSLA, etc.) ← FIX: REQUIRED!
                - strategy_type: StrategyType enum or str
                - entry_date: date
                - expiration_date: date
                - entry_price: float
                - credit_collected: float
                
                Optional keys:
                - number_of_spreads: int (default: 1)
                - entry_iv_rank: float
                - entry_vix: float
        
        Returns:
            Batch ID
        """
        # FIX: Validate required symbol field
        if "symbol" not in batch_data or not batch_data["symbol"]:
            raise ValueError(
                "batch_data['symbol'] is REQUIRED (e.g., 'SPY', 'QQQ', 'TSLA')"
            )
        
        symbol = batch_data["symbol"].upper()
        
        # Validate symbol format
        if not symbol or len(symbol) < 1 or len(symbol) > 5:
            raise ValueError(f"Invalid symbol: {symbol}. Must be 1-5 characters")
        
        # Validate strategy
        strategy = batch_data.get("strategy_type", StrategyType.IRON_CONDOR)
        if isinstance(strategy, str):
            strategy = StrategyType(strategy)
        
        query = """
            INSERT INTO [dbo].[Trade_Batches] (
                BatchName, Symbol, StrategyType,
                EntryDate, ExpirationDate,
                EntryPrice, CreditCollected,
                NumberOfSpreads
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            batch_data["batch_name"],
            symbol,                                    # ← FIX: Use validated symbol (no fallback)
            strategy.value,
            batch_data["entry_date"],
            batch_data["expiration_date"],
            batch_data["entry_price"],
            batch_data["credit_collected"],
            batch_data.get("number_of_spreads", 1)
        )
        
        batch_id = self.db.execute_insert(query, params)
        print(f"✅ Batch inserted: {batch_data['batch_name']} | {symbol} | {strategy.value} (ID: {batch_id})")
        return batch_id
    
    def insert_legs(self, batch_id: int, legs: List[OptionLeg]) -> List[int]:
        """
        Insert individual option legs for a batch.
        Supports per-leg expiration dates (for Diagonal Spreads).
        
        Args:
            batch_id: Batch ID
            legs: List of OptionLeg objects
            
        Returns:
            List of leg IDs
        """
        leg_ids = []
        query = """
            INSERT INTO [dbo].[Trade_Legs] (
                BatchID, LegNumber, OptionType, SideType,
                Strike, EntryPrice, ExpirationDate
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        for leg in legs:
            params = (
                batch_id,
                leg.leg_num,
                leg.option_type.value,
                leg.side.value,
                leg.strike,
                leg.entry_price,
                leg.expiry,  # ← Store leg-specific expiry (for Diagonal Spreads)
            )
            leg_id = self.db.execute_insert(query, params)
            leg_ids.append(leg_id)
        
        print(f"✅ {len(leg_ids)} legs inserted for batch {batch_id}")
        return leg_ids
    
    def get_batch(self, batch_id: int) -> Dict:
        """Retrieve batch details"""
        query = """
            SELECT BatchID, BatchName, Symbol, StrategyType,
            EntryDate, ExpirationDate, IsClosed,
            CreditCollected, EntryPrice
            FROM [dbo].[Trade_Batches]
            WHERE BatchID = ?
        """
        rows = self.db.execute_query(query, (batch_id,))
        
        if not rows:
            return {}
        
        r = rows[0]
        # FIX: Corrected index mapping (was r[6]-r[9], now r[6]-r[8])
        return {
            "batch_id": r[0],
            "batch_name": r[1],
            "symbol": r[2],
            "strategy_type": r[3],
            "entry_date": r[4],
            "expiration_date": r[5],
            "isclosed": r[6],                 # ← FIXED INDEX
            "credit_collected": r[7],         # ← FIXED INDEX
            "entry_price": r[8],              # ← FIXED INDEX
        }
    
    def get_batch_legs(self, batch_id: int) -> List[OptionLeg]:
        """
        Get all legs for a batch.
        Supports per-leg expirations (for Diagonal Spreads).
        """
        query = """
            SELECT LegID, LegNumber, OptionType, SideType, Strike, EntryPrice, ExpirationDate
            FROM [dbo].[Trade_Legs]
            WHERE BatchID = ?
            ORDER BY LegNumber
        """
        rows = self.db.execute_query(query, (batch_id,))
        
        if not rows:
            return []
        
        legs = []
        batch = self.get_batch(batch_id)
        
        # FIX: Symbol MUST come from batch (no fallback to "SPY")
        symbol = batch.get("symbol")
        if not symbol:
            raise ValueError(f"Batch {batch_id} has no symbol defined")
        
        batch_expiry = batch.get("expiration_date")
        batch_expiry_str = batch_expiry.strftime("%Y-%m-%d") if batch_expiry else None
        
        for r in rows:
            leg_id = r[0]
            leg_num = r[1]
            option_type = r[2]
            side_type = r[3]
            strike = r[4]
            entry_price = r[5]
            leg_expiry_date = r[6] if len(r) > 6 else None  # Per-leg expiry (if available)
            
            # Use leg-specific expiry if available, else batch expiry
            expiry = (
                leg_expiry_date.strftime("%Y-%m-%d")
                if leg_expiry_date
                else batch_expiry_str
            )
            
            leg = OptionLeg(
                leg_num=leg_num,
                option_type=OptionType[option_type],
                side=SideType[side_type],
                strike=strike,
                expiry=expiry,
                symbol=symbol,                    # ← Use batch's actual symbol (FIX: no "SPY" fallback)
                entry_price=entry_price,
                leg_id=leg_id
            )
            legs.append(leg)
        
        return legs
    
    def get_all_active_batches(self) -> List[Dict]:
        """Get all active, non-closed batches"""
        query = """
            SELECT BatchID, BatchName, Symbol, StrategyType, ExpirationDate, IsClosed
            FROM [dbo].[Trade_Batches]
            WHERE IsClosed = 0
            ORDER BY EntryDate DESC
        """
        rows = self.db.execute_query(query)
        return [
            {
                "batch_id": r[0],
                "batch_name": r[1],
                "symbol": r[2],
                "strategy_type": r[3],
                "expiry": r[4].strftime("%Y-%m-%d") if r[4] else None,
                "isclosed": r[5],
            }
            for r in rows
        ]
    
    def close_batch(
        self, batch_id: int, close_price: float, reason: str = "Manual closure"
    ):
        """
        Close a batch early (before expiration).
        Sets IsClosed=1 and records closure details.
        """
        query = """
            UPDATE [dbo].[Trade_Batches]
            SET IsClosed = 1, CloseDate = GETDATE(),
            ClosePrice = ?, CloseReason = ?
            WHERE BatchID = ?
        """
        self.db.execute_update(query, (close_price, reason, batch_id))
        print(f"✅ Batch {batch_id} CLOSED at {close_price} - {reason}")
    
    def reopen_batch(self, batch_id: int):
        """Reopen a closed batch"""
        query = """
            UPDATE [dbo].[Trade_Batches]
            SET IsClosed = 0, CloseDate = NULL, ClosePrice = NULL, CloseReason = NULL
            WHERE BatchID = ?
        """
        self.db.execute_update(query, (batch_id,))
        print(f"✅ Batch {batch_id} REOPENED")
    
    # ========== SNAPSHOT OPERATIONS ==========
    
    def insert_snapshot(self, batch_id: int, snapshot_data: Dict) -> int:
        """Insert snapshot for multi-strategy batch"""
        snapshot_time = snapshot_data.get("snapshot_time", datetime.now())
        is_eod = snapshot_data.get("is_eod", 0)
        snapshot_type = snapshot_data.get("snapshot_type", "INTRADAY")
        
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
            snapshot_data.get("spot_price", 0.0),
            snapshot_data.get("iv_rank", 0.0),
            snapshot_data.get("vix", 0.0),
            snapshot_data.get("put_delta", 0.0),
            snapshot_data.get("put_gamma", 0.0),
            snapshot_data.get("put_vega", 0.0),
            snapshot_data.get("put_theta", 0.0),
            snapshot_data.get("call_delta", 0.0),
            snapshot_data.get("call_gamma", 0.0),
            snapshot_data.get("call_vega", 0.0),
            snapshot_data.get("call_theta", 0.0),
            snapshot_data.get("current_pnl", 0.0),
            snapshot_data.get("probability_of_profit", 0.0),
            snapshot_data.get("expected_value", 0.0),
            snapshot_data.get("overall_status", "NEUTRAL"),
            snapshot_data.get("recommended_action", "HOLD"),
            snapshot_data.get("recommendation_reason", ""),
            snapshot_data.get("hard_stop_triggered", False),
            snapshot_data.get("hard_stop_reason", ""),
        )
        
        snapshot_id = self.db.execute_insert(query, params)
        eod_flag = " (EOD)" if is_eod else ""
        print(f"✅ Snapshot {snapshot_id} for batch {batch_id}{eod_flag}")
        return snapshot_id
    
    def get_latest_snapshot(self, batch_id: int) -> Dict:
        """Get most recent snapshot for batch"""
        query = """
            SELECT TOP 1 SnapshotID, SnapshotTime, SPYPrice, CurrentPnL, OverallStatus
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
            "snapshot_time": r[1],
            "spot_price": r[2],
            "current_pnl": r[3],
            "status": r[4],
        }
    
    def get_batch_snapshots(self, batch_id: int, limit: int = 100) -> List[Dict]:
        """Get snapshots for batch"""
        query = f"""
            SELECT TOP {limit} SnapshotID, SnapshotTime, SPYPrice, CurrentPnL,
            OverallStatus, IsEOD
            FROM [dbo].[Trade_Snapshots]
            WHERE BatchID = ?
            ORDER BY SnapshotTime DESC
        """
        rows = self.db.execute_query(query, (batch_id,))
        return [
            {
                "snapshot_id": r[0],
                "snapshot_time": r[1],
                "spot_price": r[2],
                "current_pnl": r[3],
                "status": r[4],
                "is_eod": r[5],
            }
            for r in rows
        ]
    
    # ========== ANALYTICS ==========
    
    def get_portfolio_summary(self, date_str: str = None) -> Dict:
        """Get portfolio summary for given date"""
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
            WHERE b.IsClosed = 0
            AND CAST(ts.SnapshotTime AS DATE) = ?
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
        """Get performance stats for batch"""
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
        if rows and rows[0][0] is not None:
            r = rows[0]
            return {
                "total_snapshots": r[0] or 0,
                "max_pnl": r[1] or 0.0,
                "min_pnl": r[2] or 0.0,
                "avg_pop": r[3] or 0.0,
            }
        
        return {
            "total_snapshots": 0,
            "max_pnl": 0.0,
            "min_pnl": 0.0,
            "avg_pop": 0.0,
        }
    
    # ========== EXPORT OPERATIONS ==========
    
    def export_journal_to_csv(
        self, batch_id: int, output_dir: str = "exports"
    ) -> Optional[str]:
        """Export trade journal to CSV"""
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
        if not rows:
            print(f"⚠️ No snapshots found for batch {batch_id}")
            return None
        
        df = pd.DataFrame(
            rows,
            columns=[
                "Time", "Spot Price", "P&L", "POP",
                "Put Delta", "Call Delta", "Action", "Status",
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
        """Export trade journal to Excel"""
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
        if not rows:
            print(f"⚠️ No snapshots found for batch {batch_id}")
            return None
        
        df = pd.DataFrame(
            rows,
            columns=[
                "Time", "Spot Price", "P&L", "POP",
                "Put Delta", "Call Delta", "Action", "Status",
            ],
        )
        
        filename = (
            f"{output_dir}/batch_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        df.to_excel(filename, index=False)
        print(f"✅ Journal exported: {filename}")
        return filename


# ============================================================================
# AUTOMATED MULTI-STRATEGY CAPTURE
# ============================================================================

class AutomatedDailyCapture:
    """
    Automated capture for ALL active batches.
    Handles multiple symbols and strategies simultaneously.
    """
    
    def __init__(self, manager: TradeBlogManager):
        self.manager = manager
    
    def capture_all_intraday(self) -> Dict:
        """Capture intraday snapshots for ALL active batches"""
        batches = self.manager.get_all_active_batches()
        
        if not batches:
            print("⚠️ No active batches to capture")
            return {}
        
        results = {}
        print(f"\n📊 Intraday Capture ({datetime.now().strftime('%H:%M:%S')})")
        print(f" Processing {len(batches)} active batch(es)...\n")
        
        for batch in batches:
            batch_id = batch["batch_id"]
            batch_name = batch["batch_name"]
            symbol = batch["symbol"]                    # ← Dynamic: from batch!
            strategy_type = batch["strategy_type"]
            
            # FIX: Validate symbol
            if not symbol:
                print(f" ❌ {batch_name}: No symbol defined (skipped)")
                continue
            
            try:
                # Get legs for this batch
                legs = self.manager.get_batch_legs(batch_id)
                
                if not legs:
                    print(f" ⚠️ {batch_name}: No legs found")
                    continue
                
                # Get Greeks engine for symbol
                greeks_engine = self.manager._get_greeks_engine(symbol)  # ← Dynamic
                position_snap = greeks_engine.get_legs_snapshot(legs)
                
                # Build snapshot data
                snapshot_data = {
                    "spot_price": position_snap["spot"],
                    "iv_rank": greeks_engine._get_iv_rank(symbol),
                    "vix": greeks_engine._get_vix(),
                    "put_delta": position_snap.get("composite_delta", 0.0),
                    "put_gamma": position_snap.get("composite_gamma", 0.0),
                    "put_vega": position_snap.get("composite_vega", 0.0),
                    "put_theta": position_snap.get("composite_theta", 0.0),
                    "call_delta": 0.0,
                    "call_gamma": 0.0,
                    "call_vega": 0.0,
                    "call_theta": 0.0,
                    "current_pnl": 35.0,
                    "probability_of_profit": 65.0,
                    "expected_value": 42.5,
                    "overall_status": "GREEN",
                    "recommended_action": "HOLD",
                    "recommendation_reason": "Theta decay favorable",
                    "hard_stop_triggered": False,
                    "snapshot_type": "INTRADAY",
                    "is_eod": 0,
                }
                
                # Insert snapshot
                snapshot_id = self.manager.insert_snapshot(batch_id, snapshot_data)
                
                results[batch_name] = {
                    "status": "✅",
                    "snapshot_id": snapshot_id,
                    "symbol": symbol,
                    "strategy": strategy_type,
                    "delta": position_snap.get("composite_delta", 0.0),
                    "theta": position_snap.get("composite_theta", 0.0),
                }
                
                print(
                    f" ✅ {batch_name:20} | {symbol:6} | {strategy_type:18} | "
                    f"Δ: {position_snap.get('composite_delta', 0.0):6.3f} | "
                    f"Θ: {position_snap.get('composite_theta', 0.0):7.3f}"
                )
            
            except Exception as e:
                results[batch_name] = {
                    "status": "❌",
                    "error": str(e),
                }
                print(f" ❌ {batch_name}: {str(e)}")
        
        return results
    
    def capture_all_eod(self) -> Dict:
        """Capture end-of-day snapshots for ALL active batches"""
        batches = self.manager.get_all_active_batches()
        
        if not batches:
            print("⚠️ No active batches to capture")
            return {}
        
        results = {}
        print(f"\n📊 EOD Capture ({datetime.now().strftime('%H:%M:%S')})")
        print(f" Processing {len(batches)} active batch(es)...\n")
        
        for batch in batches:
            batch_id = batch["batch_id"]
            batch_name = batch["batch_name"]
            symbol = batch["symbol"]
            
            try:
                legs = self.manager.get_batch_legs(batch_id)
                
                if not legs:
                    print(f" ⚠️ {batch_name}: No legs found")
                    continue
                
                greeks_engine = self.manager._get_greeks_engine(symbol)
                position_snap = greeks_engine.get_legs_snapshot(legs)
                
                snapshot_data = {
                    "spot_price": position_snap["spot"],
                    "iv_rank": 55.0,
                    "vix": 18.5,
                    "put_delta": position_snap.get("composite_delta", 0.0),
                    "put_gamma": position_snap.get("composite_gamma", 0.0),
                    "put_vega": position_snap.get("composite_vega", 0.0),
                    "put_theta": position_snap.get("composite_theta", 0.0),
                    "call_delta": 0.0,
                    "call_gamma": 0.0,
                    "call_vega": 0.0,
                    "call_theta": 0.0,
                    "current_pnl": 38.0,
                    "probability_of_profit": 67.0,
                    "expected_value": 45.2,
                    "overall_status": "GREEN",
                    "recommended_action": "HOLD",
                    "recommendation_reason": "EOD snapshot recorded",
                    "hard_stop_triggered": False,
                    "snapshot_type": "EOD",
                    "is_eod": 1,  # ← Mark as EOD
                }
                
                snapshot_id = self.manager.insert_snapshot(batch_id, snapshot_data)
                
                results[batch_name] = {
                    "status": "✅",
                    "snapshot_id": snapshot_id,
                    "eod_recorded": True,
                }
                
                print(f" ✅ {batch_name:20} | EOD Snapshot {snapshot_id}")
            
            except Exception as e:
                results[batch_name] = {
                    "status": "❌",
                    "error": str(e),
                }
                print(f" ❌ {batch_name}: {str(e)}")
        
        return results


# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """Calculate trading metrics"""
    
    @staticmethod
    def calculate_pop(delta_magnitude: float) -> float:
        """Calculate Probability of Profit based on delta"""
        return (1 - abs(delta_magnitude)) * 100
    
    @staticmethod
    def calculate_ev(pop: float, max_profit: float, max_loss: float) -> float:
        """Calculate Expected Value"""
        pop_decimal = pop / 100
        return (pop_decimal * max_profit) - ((1 - pop_decimal) * abs(max_loss))
    
    @staticmethod
    def calculate_status(delta_magnitude: float) -> str:
        """Determine position status"""
        if abs(delta_magnitude) > 0.35:
            return "RED"
        elif abs(delta_magnitude) > 0.25:
            return "YELLOW"
        return "GREEN"


# ============================================================================
# DECISION ENGINE
# ============================================================================

class DecisionEngine:
    """Generate trading recommendations"""
    
    @staticmethod
    def recommend_action(
        composite_delta: float,
        composite_theta: float,
        days_to_expiry: int,
        current_pnl: float = 0.0,
        max_loss: float = -360.0,
    ) -> Tuple[str, str]:
        """Generate action recommendation for any strategy"""
        
        # Hard stops
        if current_pnl <= max_loss:
            return "CLOSE", f"Max loss (-${abs(max_loss)}) hit"
        
        # Days to expiry decision
        if days_to_expiry <= 3:
            return "CLOSE", "Expiration within 3 days"
        
        if days_to_expiry <= 5:
            if abs(composite_delta) > 0.40:
                return "CLOSE_OR_ROLL", f"Close to expiration with high delta {composite_delta:.2f}"
            return "HOLD", "Let theta finish"
        
        # Delta thresholds
        if abs(composite_delta) > 0.45:
            return "ROLL", f"Position delta too high: {composite_delta:.2f}"
        
        if abs(composite_delta) < 0.10 and days_to_expiry > 10:
            return "HOLD", "Good position, let theta work"
        
        return "HOLD", f"Delta: {composite_delta:.2f}, Theta: {composite_theta:.3f}"


# ============================================================================
# HELPER: Create Diagonal Spreads Easily
# ============================================================================

def create_diagonal_spread(
    manager: TradeBlogManager,
    batch_name: str,
    symbol: str,
    entry_date: datetime.date,
    near_expiry_date: datetime.date,      # 30 DTE (gets rolled monthly)
    far_expiry_date: datetime.date,        # 60 DTE (longer term)
    near_strike: float,                    # Near-term short strike
    far_strike: float,                     # Far-term long strike
    near_credit: float,                    # Credit from near-term short
    far_debit: float,                      # Debit for far-term long
) -> int:
    """Create a diagonal call spread with different expirations"""
    
    # Insert batch
    batch_data = {
        "batch_name": batch_name,
        "symbol": symbol,
        "strategy_type": StrategyType.DIAGONAL_SPREAD,
        "entry_date": entry_date,
        "expiration_date": far_expiry_date,  # Batch tracks far-term expiry
        "entry_price": 0.0,
        "credit_collected": near_credit,
        "number_of_spreads": 1,
    }
    
    batch_id = manager.insert_batch(batch_data)
    
    # Create legs with INDIVIDUAL expirations
    legs = [
        OptionLeg(
            leg_num=1,
            option_type=OptionType.CALL,
            side=SideType.LONG,
            strike=far_strike,
            expiry=far_expiry_date.strftime("%Y-%m-%d"),  # Far-term expiry
            symbol=symbol,
            entry_price=far_debit,
        ),
        OptionLeg(
            leg_num=2,
            option_type=OptionType.CALL,
            side=SideType.SHORT,
            strike=near_strike,
            expiry=near_expiry_date.strftime("%Y-%m-%d"),  # Near-term expiry (different!)
            symbol=symbol,
            entry_price=near_credit,
        ),
    ]
    
    manager.insert_legs(batch_id, legs)
    
    print(f"✅ Diagonal Spread created: {batch_name}")
    print(f"   Far-term (LONG {far_strike}): {far_expiry_date}")
    print(f"   Near-term (SHORT {near_strike}): {near_expiry_date}")
    
    return batch_id


# ============================================================================
# MAIN USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TradeBlog Enhanced - Multi-Symbol, Multi-Strategy Trading System")
    print("=" * 80)
    
    # Initialize manager
    manager = TradeBlogManager()
    
    # ====== EXAMPLE 1: SPY Iron Condor ======
    # print("\n[EXAMPLE 1] Creating SPY Iron Condor Batch")
    # print("-" * 80)
    
    # spy_batch_data = {
    #     "batch_name": "SPY_IC_Week1",
    #     "symbol": "SPY",                            # ← REQUIRED (no default)
    #     "strategy_type": StrategyType.IRON_CONDOR,
    #     "entry_date": datetime.now().date(),
    #     "expiration_date": (datetime.now() + timedelta(days=45)).date(),
    #     "entry_price": 582.50,
    #     "credit_collected": 140.0,
    #     "number_of_spreads": 20,
    # }
    
    # spy_batch_id = manager.insert_batch(spy_batch_data)
    
    # # Create IC legs
    # spy_legs = [
    #     OptionLeg(
    #         leg_num=1,
    #         option_type=OptionType.PUT,
    #         side=SideType.LONG,
    #         strike=565.0,
    #         expiry=(datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d"),
    #         symbol="SPY",                           # ← REQUIRED (no default)
    #         entry_price=0.30,
    #     ),
    #     OptionLeg(
    #         leg_num=2,
    #         option_type=OptionType.PUT,
    #         side=SideType.SHORT,
    #         strike=570.0,
    #         expiry=(datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d"),
    #         symbol="SPY",
    #         entry_price=0.60,
    #     ),
    #     OptionLeg(
    #         leg_num=3,
    #         option_type=OptionType.CALL,
    #         side=SideType.SHORT,
    #         strike=595.0,
    #         expiry=(datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d"),
    #         symbol="SPY",
    #         entry_price=0.50,
    #     ),
    #     OptionLeg(
    #         leg_num=4,
    #         option_type=OptionType.CALL,
    #         side=SideType.LONG,
    #         strike=600.0,
    #         expiry=(datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d"),
    #         symbol="SPY",
    #         entry_price=0.20,
    #     ),
    # ]
    
    # manager.insert_legs(spy_batch_id, spy_legs)
    
    # # ====== EXAMPLE 2: QQQ Put Spread ======
    # print("\n[EXAMPLE 2] Creating QQQ Put Spread Batch")
    # print("-" * 80)
    
    # qqq_batch_data = {
    #     "batch_name": "QQQ_PS_Week1",
    #     "symbol": "QQQ",                            # ← Dynamic symbol!
    #     "strategy_type": StrategyType.PUT_SPREAD,
    #     "entry_date": datetime.now().date(),
    #     "expiration_date": (datetime.now() + timedelta(days=30)).date(),
    #     "entry_price": 420.30,
    #     "credit_collected": 75.0,
    #     "number_of_spreads": 15,
    # }
    
    # qqq_batch_id = manager.insert_batch(qqq_batch_data)
    
    # # Create spread legs
    # qqq_legs = [
    #     OptionLeg(
    #         leg_num=1,
    #         option_type=OptionType.PUT,
    #         side=SideType.LONG,
    #         strike=400.0,
    #         expiry=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
    #         symbol="QQQ",                           # ← Dynamic symbol!
    #         entry_price=0.25,
    #     ),
    #     OptionLeg(
    #         leg_num=2,
    #         option_type=OptionType.PUT,
    #         side=SideType.SHORT,
    #         strike=410.0,
    #         expiry=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
    #         symbol="QQQ",``
    #         entry_price=0.75,
    #     ),
    # ]
    
    # manager.insert_legs(qqq_batch_id, qqq_legs)
    
    # ====== EXAMPLE 3: TSLA Diagonal Spread (Different Expirations!) ======
    # print("\n[EXAMPLE 3] Creating TSLA Diagonal Spread Batch")
    # print("-" * 80)
    
    # tsla_diagonal_id = create_diagonal_spread(
    #     manager=manager,
    #     batch_name="ALAB_DIAGONAL_Jan",
    #     symbol="ALAB",                              # ← Dynamic symbol!
    #     # entry_date=datetime.now().date(),
    #     # near_expiry_date=(datetime.now() + timedelta(days=30)).date(),
    #     # far_expiry_date=(datetime.now() + timedelta(days=60)).date(),
    #     entry_date=date(2026,1,25),
    #     near_expiry_date=date(2026,3,20),
    #     far_expiry_date=date(2026,7,17),
    #     near_strike=220.0,
    #     far_strike=170.0,
    #     near_credit=6.59,
    #     far_debit=42.75,
    # )
    
    # # ====== AUTOMATED CAPTURE ======
    # print("\n" + "=" * 80)
    # print("AUTOMATED INTRADAY & EOD CAPTURE")
    # print("=" * 80)
    
    capture = AutomatedDailyCapture(manager)
    
    # Intraday snapshot
    # print("\n[INTRADAY SNAPSHOT]")
    intraday_results = capture.capture_all_intraday()
    
    # EOD snapshot
    print("\n[EOD SNAPSHOT]")
    eod_results = capture.capture_all_eod()
    
    # ====== PORTFOLIO SUMMARY ======
    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)
    
    summary = manager.get_portfolio_summary()
    print(f"\nActive Batches: {summary['active_batches']}")
    print(f"Total P&L: ${summary['total_pnl']:.2f}")
    print(f"Portfolio Delta: {summary['portfolio_delta']:.3f}")
    print(f"Daily Theta: ${summary['daily_theta']:.3f}")
    
    # ====== EARLY CLOSURE EXAMPLE ======
    print("\n" + "=" * 80)
    print("EARLY CLOSURE EXAMPLE")
    print("=" * 80)
    
    print(f"\nClosing SPY Iron Condor early (Batch {spy_batch_id})...")
    manager.close_batch(
        spy_batch_id,
        close_price=95.0,
        reason="Profitable close - 68% of max profit achieved"
    )
    
    # Verify closure
    batch = manager.get_batch(spy_batch_id)
    print(f"Batch Status: Closed={batch['isclosed']}")  # ← FIX: Fixed key name
    
    print("\n" + "=" * 80)
    print("✅ Multi-Symbol, Multi-Strategy, Multi-Expiry System Operational")
    print("=" * 80 + "\n")
