"""
SNDK-Style Option Screener
Combines SQL Server volatility metrics with Yahoo Finance option pricing
Filters for true high-premium persistence (not just high realized volatility)

Author: Trading Systems Developer
Date: March 21, 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyodbc
from sqlalchemy import create_engine, text  # Add 'text' here
import warnings
import time
warnings.filterwarnings('ignore')

class SNDKOptionScreener:
    def __init__(self, sql_connection_string):
        """
        Initialize with SQL Server connection string
        Example: "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=TradingDB;UID=user;PWD=pass"
        """
        self.conn_str = sql_connection_string
        self.engine = create_engine(f"mssql+pyodbc:///?odbc_connect={sql_connection_string}")

    def fetch_high_score_stocks(self, min_premium_score=80, lookback_days=5):
        """ Fetch stocks from SQL Server with high PremiumScore
        FIXED: Removed nested aggregate/subquery error
        """
        query = f"""exec sp_SNDK_Filter {lookback_days},{min_premium_score} """
        try:
            df = pd.read_sql(query, self.engine)
            print(f"Fetched {len(df)} high-score stocks from SQL Server")
            return df
        except Exception as e:
            print(f"SQL Error: {e}")
            return pd.DataFrame()    

    def fetch_high_score_stocks_BKP(self, min_premium_score=80, lookback_days=5):
        """ Fetch stocks from SQL Server with high PremiumScore
        FIXED: Removed nested aggregate/subquery error
        """
        query = f"""
        WITH DailyMetrics AS (
            SELECT 
                Symbol,
                [Time] as TradeDate,
                [Open], High, Low, [Close], Volume,
                LAG([Close]) OVER (PARTITION BY Symbol ORDER BY [Time]) AS PrevClose,
                (High - Low) / [Close] * 100.0 AS DailyRangePct,
                ([Close] - LAG([Close]) OVER (PARTITION BY Symbol ORDER BY [Time])) 
                    / LAG([Close]) OVER (PARTITION BY Symbol ORDER BY [Time]) * 100.0 AS CloseToCloseChange,
                AVG((High - Low) / [Close] * 100.0) OVER (
                    PARTITION BY Symbol 
                    ORDER BY [Time] 
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS AvgRange5Day,
                ROW_NUMBER() OVER (PARTITION BY Symbol ORDER BY [Time] DESC) AS rn_latest
            FROM ai_stock_prices p (nolock)
            WHERE [Time] >= DATEADD(day, -{lookback_days}, GETDATE())
            and [Close] > 50
            and Not ([Open] =0 
                or  [High] =0 
                or  [Low] =0 
                or [Close] =0 )
        ),
        LatestData AS (
            SELECT 
                Symbol,
                [Close] AS LatestPrice,
                DailyRangePct AS LatestRangePct,
                CloseToCloseChange AS LatestChange,
                AvgRange5Day,
                Volume AS LatestVolume,
                CAST(
                    (DailyRangePct * 10) + 
                    (ABS(CloseToCloseChange) * 5) + 
                    (CASE WHEN CloseToCloseChange < -5 THEN 20 ELSE 0 END)
                    AS INT
                ) AS PremiumScore
            FROM DailyMetrics
            WHERE rn_latest = 1
        )
        SELECT 
            Symbol,
            LatestPrice AS CurrentPrice,
            LatestRangePct,
            LatestChange,
            AvgRange5Day,
            PremiumScore
        FROM LatestData
        WHERE 
            PremiumScore >= {min_premium_score}
            AND LatestVolume > 500000
        ORDER BY PremiumScore DESC
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            print(f"Fetched {len(df)} high-score stocks from SQL Server")
            return df
        except Exception as e:
            print(f"SQL Error: {e}")
            return pd.DataFrame()
       
    def fetch_high_score_stocks_Old(self, min_premium_score=80, lookback_days=5):
        """
        Fetch stocks from SQL Server with high PremiumScore from your OLHC scanner
        """
        query = f"""
        WITH DailyMetrics AS (
            SELECT 
                Symbol,
                [Time] as TradeDate,
                [Open], High, Low, [Close], Volume,
                LAG([Close]) OVER (PARTITION BY Symbol ORDER BY [Time]) AS PrevClose,
                (High - Low) / [Close] * 100.0 AS DailyRangePct,
                ([Close] - LAG([Close]) OVER (PARTITION BY Symbol ORDER BY [Time])) 
                    / LAG([Close]) OVER (PARTITION BY Symbol ORDER BY [Time]) * 100.0 AS CloseToCloseChange,
                AVG((High - Low) / [Close] * 100.0) OVER (
                    PARTITION BY Symbol 
                    ORDER BY [Time] 
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS AvgRange5Day
            FROM ai_stock_prices p (nolock)
            WHERE [time] >= DATEADD(day, -{lookback_days}, GETDATE())
            and Not ([Open] =0 
                or  [High] =0 
                or  [Low] =0 
                or [Close] =0 )
        )
        SELECT DISTINCT
            Symbol,
            MAX(CASE WHEN TradeDate = (SELECT MAX(TradeDate) FROM DailyMetrics) THEN [Close] END) AS CurrentPrice,
            MAX(CASE WHEN TradeDate = (SELECT MAX(TradeDate) FROM DailyMetrics) THEN DailyRangePct END) AS LatestRangePct,
            MAX(CASE WHEN TradeDate = (SELECT MAX(TradeDate) FROM DailyMetrics) THEN CloseToCloseChange END) AS LatestChange,
            MAX(AvgRange5Day) AS Avg5DayRange,
            CAST(
                MAX(DailyRangePct * 10 + ABS(CloseToCloseChange) * 5 + 
                    CASE WHEN CloseToCloseChange < -5 THEN 20 ELSE 0 END)
                AS INT
            ) AS PremiumScore
        FROM DailyMetrics
        GROUP BY Symbol
        HAVING 
            CAST(
                MAX(DailyRangePct * 10 + ABS(CloseToCloseChange) * 5 + 
                    CASE WHEN CloseToCloseChange < -5 THEN 20 ELSE 0 END)
                AS INT
            ) >= {min_premium_score}
            AND MAX(CASE WHEN TradeDate = (SELECT MAX(TradeDate) FROM DailyMetrics) THEN Volume END) > 500000
        ORDER BY PremiumScore DESC
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            print(f"Fetched {len(df)} high-score stocks from SQL Server")
            return df
        except Exception as e:
            print(f"SQL Error: {e}")
            return pd.DataFrame()

    def get_option_chain(self, symbol, max_expiry_days=45):
        """Get option chain from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return None, None, None
                
            target_date = datetime.now() + timedelta(days=max_expiry_days)
            valid_expirations = [exp for exp in expirations 
                               if datetime.strptime(exp, '%Y-%m-%d') <= target_date]
            
            expiry = valid_expirations[0] if valid_expirations else expirations[0]
            opt_chain = ticker.option_chain(expiry)
            current_price = ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice', 0)
            
            return opt_chain, current_price, expiry
        except Exception as e:
            print(f"Error fetching options for {symbol}: {e}")
            return None, None, None

    def calculate_iv_metrics(self, calls, puts, current_price):
        """
        Calculate key IV metrics from option chain
        FIXED: Handles empty DataFrames and missing data
        """
        # Validate inputs
        if calls is None or calls.empty or current_price == 0:
            print(f"  Warning: Empty calls data or zero price")
            return {}
        
        # Ensure required columns exist
        required_cols = ['strike', 'lastPrice', 'impliedVolatility']
        missing_cols = [col for col in required_cols if col not in calls.columns]
        if missing_cols:
            print(f"  Warning: Missing columns {missing_cols}")
            return {}
        
        # Filter out rows with zero or null implied volatility
        valid_calls = calls[calls['impliedVolatility'].notna() & (calls['impliedVolatility'] > 0)].copy()
        
        if valid_calls.empty:
            print(f"  Warning: No valid options with implied volatility")
            return {}
        
        try:
            # Find ATM options (closest strike to current price)
            valid_calls['diff'] = abs(valid_calls['strike'] - current_price)
            atm_idx = valid_calls['diff'].idxmin()
            atm_call = valid_calls.loc[atm_idx]
            
            # OTM call ~8% out
            otm_target = current_price * 1.08
            valid_calls['otm_diff'] = abs(valid_calls['strike'] - otm_target)
            otm_idx = valid_calls['otm_diff'].idxmin()
            otm_call = valid_calls.loc[otm_idx]
            
            metrics = {
                'ATM_IV': float(atm_call['impliedVolatility']) * 100,
                'ATM_Strike': float(atm_call['strike']),
                'ATM_Premium': float(atm_call['lastPrice']) if pd.notna(atm_call['lastPrice']) else 0,
                'OTM_Strike': float(otm_call['strike']),
                'OTM_Premium': float(otm_call['lastPrice']) if pd.notna(otm_call['lastPrice']) else 0,
                'OTM_IV': float(otm_call['impliedVolatility']) * 100,
            }
            
            return metrics
            
        except Exception as e:
            print(f"  Error calculating metrics: {e}")
            return {}



    def calculate_iv_metrics_old(self, calls, puts, current_price):
        """Calculate key IV metrics from option chain"""
        if calls is None or current_price == 0:
            return {}
            
        calls['diff'] = abs(calls['strike'] - current_price)
        atm_call = calls.loc[calls['diff'].idxmin()]
        
        # OTM call ~8% out (comparable to SNDK 770/710)
        otm_target = current_price * 1.08
        calls['otm_diff'] = abs(calls['strike'] - otm_target)
        otm_call = calls.loc[calls['otm_diff'].idxmin()]
        
        return {
            'ATM_IV': atm_call['impliedVolatility'] * 100 if 'impliedVolatility' in atm_call else 0,
            'OTM_Strike': otm_call['strike'],
            'OTM_Premium': otm_call['lastPrice'] if 'lastPrice' in otm_call else 0,
            'OTM_IV': otm_call['impliedVolatility'] * 100 if 'impliedVolatility' in otm_call else 0,
            'ATM_Delta': atm_call.get('delta', 0),
            'OTM_OI': otm_call.get('openInterest', 0)
        }

    def get_historical_volatility(self, symbol, days=20):
        """Calculate 20-day historical volatility"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            if len(hist) < 20:
                return 0
                
            log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
            hv = log_returns.std() * np.sqrt(252) * 100
            return hv
        except:
            return 0

    def calculate_quality_score(self, atm_iv, realized_vol, premium_pct, hv_20, 
                           sql_change, days_to_earnings=999, symbol=""):
        """
        Balanced scoring: Keeps SNDK, filters RH, rewards quality setups
        """
        score = 50  # Start neutral
        iv_rv_ratio = atm_iv / realized_vol if realized_vol > 0 else 0
        
        # 1. IV/RV Ratio (key metric)
        if 3.0 <= iv_rv_ratio <= 6.0:
            score += 20  # Sweet spot
        elif 6.0 < iv_rv_ratio <= 10.0:
            score += 15  # Good (SNDK=8.38)
        elif 10.0 < iv_rv_ratio <= 15.0:
            score += 5
        elif iv_rv_ratio > 15.0:
            score -= 15  # Extreme (RH=19.1)
        elif iv_rv_ratio < 2.0:
            score -= 10
        
        # 2. Earnings Proximity (CRITICAL - Less aggressive than before)
        if days_to_earnings <= 5:
            score -= 25  # Very close
        elif days_to_earnings <= 12:  # RH at ~11 days gets caught here
            score -= 15  # Close - moderate penalty
        elif days_to_earnings <= 21:
            score -= 5   # Approaching
        # >21 days = no penalty (SNDK, KTOS, RKLB, LEU all safe)
        
        # 3. IV Level
        if 50 <= atm_iv <= 90:
            score += 15
        elif 90 < atm_iv <= 120:
            score += 10
        elif 120 < atm_iv <= 150:
            score += 5
        elif atm_iv > 150:
            score -= 10
        
        # 4. Directional Context
        if sql_change < -5 and iv_rv_ratio > 3.0:
            score += 10  # Fear = opportunity (SNDK -8%)
        elif sql_change > 10 and iv_rv_ratio > 10:
            score -= 10
        
        return max(0, min(100, score))


    def calculate_quality_score_V2(self, atm_iv, realized_vol, premium_pct, hv_20, 
                           sql_change, days_to_earnings=999, symbol=""):
        """
        CORRECTED: Penalizes pre-earnings/event risk, rewards sustainable premium
        """
        score = 0
        iv_rv_ratio = atm_iv / realized_vol if realized_vol > 0 else 0
        
        # 1. IV/RV Ratio (30 points) - PENALIZE EXTREME VALUES
        if 3.0 < iv_rv_ratio <= 6.0:
            score += 25  # Sweet spot (SNDK territory)
        elif 6.0 < iv_rv_ratio <= 10.0:
            score += 20  # Good but elevated
        elif 10.0 < iv_rv_ratio <= 15.0:
            score += 5   # Suspicious - check for event risk
        elif iv_rv_ratio > 15.0:
            score -= 20  # PENALTY: Likely pre-earnings spike (RH problem)
        elif iv_rv_ratio < 1.5:
            score -= 10  # IV too low vs realized
        
        # 2. EARNINGS PROXIMITY (CRITICAL FIX)
        if days_to_earnings <= 3:
            score -= 40  # IV crush imminent
        elif days_to_earnings <= 7:
            score -= 25  # High risk (RH is here)
        elif days_to_earnings <= 14:
            score -= 10  # Moderate risk
        
        # 3. Absolute IV (20 points)
        if 60 <= atm_iv <= 100:
            score += 20  # Ideal range
        elif 100 < atm_iv <= 140:
            score += 10  # Elevated
        elif atm_iv > 140:
            score -= 15  # Extreme = event risk
        
        # 4. IV-HV Spread (15 points)
        spread = atm_iv - hv_20
        if 20 <= spread <= 50:
            score += 15  # Healthy premium
        elif spread > 70:
            score -= 10  # Extreme spread = event risk
        
        # 5. Directional Context (10 points)
        if sql_change < -5 and 2.0 < iv_rv_ratio < 10.0:
            score += 10  # Fear-driven decline (SNDK-like)
        elif sql_change > 10 and iv_rv_ratio > 10:
            score -= 15  # Euphoria + high IV = crash risk
        
        return max(0, min(100, score))

    def get_days_to_earnings(self, symbol):
        """Fetch earnings date from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            if calendar is not None and not calendar.empty:
                earnings_date = pd.to_datetime(calendar.index[0])
                days = (earnings_date.date() - datetime.now().date()).days
                return max(0, days)
            return 999  # No earnings found
        except:
            return 999

    def calculate_quality_score_old(self, atm_iv, realized_vol, premium_pct, hv_20, sql_change):
        """
        CORRECTED ALGORITHM: Distinguishes SNDK-like setups from post-catalyst false positives
        """
        score = 0
        iv_rv_ratio = atm_iv / realized_vol if realized_vol > 0 else 0
        
        # 1. IV/RV Ratio (40 points max) - KEY METRIC
        if iv_rv_ratio > 1.5: score += 10
        if iv_rv_ratio > 3.0: score += 15
        if iv_rv_ratio > 6.0: score += 15  # SNDK territory
            
        # 2. Absolute IV Level (20 points)
        if atm_iv > 60: score += 10
        if atm_iv > 90: score += 10
            
        # 3. IV vs Historical Volatility (20 points)
        iv_hv_spread = atm_iv - hv_20
        if iv_hv_spread > 20: score += 10
        if iv_hv_spread > 40: score += 10
            
        # 4. Post-Catalyst Penalty (detects AAOI-type setups)
        if sql_change > 10 and realized_vol > 15:
            score -= 25  # Heavy penalty for post-catalyst spikes
        elif sql_change < -5:
            score += 10  # Reward for downside fear (more sustainable)
            
        # 5. Premium Persistence (10 points)
        premium_efficiency = premium_pct / realized_vol if realized_vol > 0 else 0
        if premium_efficiency > 0.5: score += 5
        if premium_efficiency > 0.8: score += 5
            
        return max(0, min(100, score))

    def analyze_stock(self, symbol, sql_data):
        """Complete analysis with ATM premium > 6% filter"""
        print(f"Analyzing {symbol}...")
        
        # Extract SQL data
        realized_vol = sql_data.get('LatestRangePct', 0)
        sql_change = sql_data.get('LatestChange', 0)
        sql_score = sql_data.get('PremiumScore', 0)
        
        # Get options
        opt_chain, current_price, expiry = self.get_option_chain(symbol)
        if opt_chain is None or current_price == 0:
            return None
        
        opt_metrics = self.calculate_iv_metrics(opt_chain.calls, opt_chain.puts, current_price)
        if not opt_metrics or opt_metrics.get('ATM_IV', 0) == 0:
            return None
        
        # Calculate metrics
        hv_20 = self.get_historical_volatility(symbol)
        atm_iv = opt_metrics['ATM_IV']
        atm_premium = opt_metrics.get('ATM_Premium', 0)  # ATM call premium
        
        # NEW FILTER: ATM premium > 6% of stock price
        atm_premium_pct = (atm_premium / current_price) * 100 if current_price > 0 else 0
        
        print(f"  {symbol}: ATM Premium = ${atm_premium:.2f} ({atm_premium_pct:.2f}% of stock)")
        
        if atm_premium_pct < 6.0:
            print(f"  Skipped: ATM premium {atm_premium_pct:.2f}% < 6% threshold")
            return None  # Filter out low premium stocks
        
        # Continue with rest of analysis...
        otm_premium = opt_metrics.get('OTM_Premium', 0)
        premium_pct = (otm_premium / current_price) * 100 if current_price > 0 else 0
        
        days_to_earnings = self.get_days_to_earnings(symbol)
        
        quality_score = self.calculate_quality_score(
            atm_iv=atm_iv,
            realized_vol=realized_vol,
            premium_pct=premium_pct,
            hv_20=hv_20,
            sql_change=sql_change,
            days_to_earnings=days_to_earnings,
            symbol=symbol
        )
        
        return {
            'Scan_Date': datetime.now(),
            'Symbol': symbol,
            'Stock_Price': round(current_price, 2),
            'SQL_Score': sql_score,
            'SQL_Realized_Vol': round(realized_vol, 2),
            'SQL_Change_Pct': round(sql_change, 2),
            'Expiry': expiry,
            'ATM_IV': round(atm_iv, 1),
            'ATM_Premium': round(atm_premium, 2),  # NEW FIELD
            'ATM_Premium_Pct': round(atm_premium_pct, 2),  # NEW FIELD
            'OTM_IV': round(opt_metrics.get('OTM_IV', atm_iv), 1),
            'HV_20': round(hv_20, 1),
            'IV_RV_Ratio': round(atm_iv / realized_vol if realized_vol > 0 else 0, 2),
            'OTM_Strike': opt_metrics.get('OTM_Strike', 0),
            'OTM_Premium': round(otm_premium, 2),
            'Premium_Pct_Stock': round(premium_pct, 2),
            'Quality_Score': quality_score,
            'Setup_Type': 'HIGH_PREMIUM' if atm_premium_pct > 8 else 'SNDK_LIKE' if quality_score >= 80 else 'ELEVATED_VOL',
            'IsTopPick': 0
        }



    def analyze_stock_V1(self, symbol, sql_data):
        """Complete analysis combining SQL metrics with option pricing"""
        print(f"Analyzing {symbol}...")
        
        # Extract SQL data
        realized_vol = sql_data.get('LatestRangePct', 0)
        sql_change = sql_data.get('LatestChange', 0)
        sql_score = sql_data.get('PremiumScore', 0)
        
        # Get options
        opt_chain, current_price, expiry = self.get_option_chain(symbol)
        if opt_chain is None or current_price == 0:
            return None
        
        opt_metrics = self.calculate_iv_metrics(opt_chain.calls, opt_chain.puts, current_price)
        if not opt_metrics or opt_metrics.get('ATM_IV', 0) == 0:
            return None
        
        # Calculate metrics
        hv_20 = self.get_historical_volatility(symbol)
        atm_iv = opt_metrics['ATM_IV']
        premium_pct = (opt_metrics.get('OTM_Premium', 0) / current_price) * 100 if current_price > 0 else 0
        
        # CRITICAL: Define days_to_earnings BEFORE using it
        days_to_earnings = self.get_days_to_earnings(symbol)
        
        # Now calculate quality score with all parameters defined
        quality_score = self.calculate_quality_score(
            atm_iv=atm_iv,
            realized_vol=realized_vol,
            premium_pct=premium_pct,
            hv_20=hv_20,
            sql_change=sql_change,
            days_to_earnings=days_to_earnings,
            symbol=symbol
        )
        
        return {
            'Scan_Date': datetime.now(),
            'Symbol': symbol,
            'Stock_Price': round(current_price, 2),
            'SQL_Score': sql_score,
            'SQL_Realized_Vol': round(realized_vol, 2),
            'SQL_Change_Pct': round(sql_change, 2),
            'Expiry': expiry,
            'ATM_IV': round(atm_iv, 1),
            'OTM_IV': round(opt_metrics.get('OTM_IV', atm_iv), 1),
            'HV_20': round(hv_20, 1),
            'IV_RV_Ratio': round(atm_iv / realized_vol if realized_vol > 0 else 0, 2),
            'IV_Premium_Over_HV': round(atm_iv - hv_20, 1),
            'OTM_Strike': opt_metrics.get('OTM_Strike', 0),
            'OTM_Premium': round(opt_metrics.get('OTM_Premium', 0), 2),
            'Premium_Pct_Stock': round(premium_pct, 2),
            'Premium_Efficiency': round(premium_pct / realized_vol if realized_vol > 0 else 0, 2),
            'Quality_Score': quality_score,
            'Setup_Type': 'SNDK_LIKE' if quality_score >= 80 else 'ELEVATED_VOL' if quality_score >= 60 else 'POST_CATALYST/AVOID'
        }



    def analyze_stock_old(self, symbol, sql_data):
        """Complete analysis combining SQL metrics with option pricing"""
        print(f"Analyzing {symbol}...")
         # EXTRACT VALUES FROM sql_data FIRST
        realized_vol = sql_data.get('LatestRangePct', 0)
        sql_change = sql_data.get('LatestChange', 0)  # <-- ADD THIS LINE
        sql_score = sql_data.get('PremiumScore', 0)
        opt_chain, current_price, expiry = self.get_option_chain(symbol)
        if opt_chain is None:
            return None
            
        opt_metrics = self.calculate_iv_metrics(opt_chain.calls, opt_chain.puts, current_price)
        if not opt_metrics:
            return None
        if opt_metrics.get('ATM_IV', 0) == 0:
            print(f"  Skipped: Zero ATM IV")
            return None   
        hv_20 = self.get_historical_volatility(symbol)
        atm_iv = opt_metrics['ATM_IV']
        realized_vol = sql_data.get('LatestRangePct', 0)
        premium_pct = (opt_metrics['OTM_Premium'] / current_price) * 100 if current_price > 0 else 0
        
        # Key ratios
        iv_rv_ratio = atm_iv / realized_vol if realized_vol > 0 else 0
        premium_efficiency = premium_pct / realized_vol if realized_vol > 0 else 0
        
        # Calculate quality score
        # quality_score = self.calculate_quality_score(
        #     atm_iv, realized_vol, premium_pct, hv_20, 
        #     sql_data.get('LatestChange', 0)
        # )
        quality_score = self.calculate_quality_score(
        atm_iv=atm_iv,
        realized_vol=realized_vol,
        premium_pct=premium_pct,
        hv_20=hv_20,
        sql_change=sql_change,  # <-- Now works!
        days_to_earnings=days_to_earnings,
        symbol=symbol
    )
        return {
            'Scan_Date': datetime.now(),
            'Symbol': symbol,
            'Stock_Price': round(current_price, 2),
            'SQL_Score': sql_data.get('PremiumScore', 0),
            'SQL_Realized_Vol': round(realized_vol, 2),
            'SQL_Change_Pct': round(sql_data.get('LatestChange', 0), 2),
            'OTM_IV': round(opt_metrics.get('OTM_IV', atm_iv), 1),  # ADD THIS
            'Expiry': expiry,
            'ATM_IV': round(atm_iv, 1),
            'HV_20': round(hv_20, 1),
            'IV_RV_Ratio': round(iv_rv_ratio, 2),
            'IV_Premium_Over_HV': round(atm_iv - hv_20, 1),
            
            'OTM_Strike': opt_metrics['OTM_Strike'],
            'OTM_Premium': round(opt_metrics['OTM_Premium'], 2),
            'Premium_Pct_Stock': round(premium_pct, 2),
            'Premium_Efficiency': round(premium_efficiency, 2),
            
            'Quality_Score': quality_score,
            'Setup_Type': 'SNDK_LIKE' if quality_score >= 80 else 'ELEVATED_VOL' if quality_score >= 60 else 'POST_CATALYST/AVOID'
        }

    def save_run_results(self, results_df, execution_time, market_regime="UNKNOWN"):
        """Save complete run results to SQL Server"""
        if results_df.empty:
            return False
            
        today = datetime.now()
        run_id = today.strftime('%Y%m%d')
        
        try:
            # Prepare results
            results_to_save = results_df.copy()
            
            # Fix column names
            if 'Scan_Date' in results_to_save.columns:
                results_to_save = results_to_save.rename(columns={'Scan_Date': 'RunDateTime'})
            
            # Add missing columns
            results_to_save['RunID'] = run_id
            results_to_save['IsTopPick'] = 0
            results_to_save['Notes'] = None
            
            # Add OTM_IV if missing (fallback to ATM_IV)
            if 'OTM_IV' not in results_to_save.columns:
                results_to_save['OTM_IV'] = results_to_save['ATM_IV']
                
            if 'RunDateTime' not in results_to_save.columns:
                results_to_save['RunDateTime'] = datetime.now()
            
            # Reorder columns to match SQL
            sql_cols = ['RunID', 'RunDateTime', 'Symbol', 'Stock_Price', 'SQL_Score',
                    'SQL_Realized_Vol', 'SQL_Change_Pct', 'Expiry', 'ATM_IV', 'OTM_IV',
                    'HV_20', 'IV_RV_Ratio', 'IV_Premium_Over_HV', 'OTM_Strike',
                    'OTM_Premium', 'Premium_Pct_Stock', 'Premium_Efficiency',
                    'Quality_Score', 'Setup_Type', 'IsTopPick', 'Notes']
            
            available_cols = [c for c in sql_cols if c in results_to_save.columns]
            results_to_save = results_to_save[available_cols]
            
            # Flag top pick
            max_idx = results_to_save['Quality_Score'].idxmax()
            results_to_save.loc[max_idx, 'IsTopPick'] = 1
            
            # Save detailed results
            results_to_save.to_sql('SNDK_Screen_DailyResults', self.engine, 
                                if_exists='append', index=False)
            
            # Save summary
            summary = {
                'RunID': run_id, 'RunDate': today.date(), 'RunTime': today.time().replace(microsecond=0),
                'TotalStocksScanned': len(results_df), 'StocksWithOptions': len(results_df[results_df['ATM_IV'] > 0]),
                'SNDK_LikeSetups': len(results_df[results_df['Quality_Score'] >= 80]),
                'ElevatedVolSetups': len(results_df[(results_df['Quality_Score'] >= 60) & (results_df['Quality_Score'] < 80)]),
                'PostCatalystAvoided': len(results_df[results_df['Quality_Score'] < 60]),
                'AvgQualityScore': round(results_df['Quality_Score'].mean(), 2),
                'TopSymbol': results_df.loc[results_df['Quality_Score'].idxmax(), 'Symbol'],
                'TopQualityScore': int(results_df['Quality_Score'].max()),
                'MarketRegime': market_regime, 'ExecutionTimeSeconds': int(execution_time)
            }
            
            # Delete existing summary for today (avoid PK conflict)
            try:
                with self.engine.connect() as conn:
                    conn.execute(text(f"DELETE FROM SNDK_Screen_RunSummary WHERE RunID = '{run_id}'"))
                    conn.commit()
            except:
                pass  # May not exist
            
            pd.DataFrame([summary]).to_sql('SNDK_Screen_RunSummary', self.engine, if_exists='append', index=False)
            
            print(f"\n✓ Saved to SQL (RunID: {run_id}) | Top: {summary['TopSymbol']} (Score: {summary['TopQualityScore']})")
            return True
            
        except Exception as e:
            print(f"\n✗ Save failed: {e}")
            import traceback
            traceback.print_exc()
            return False


    def run_screen(self, min_sql_score=80, min_quality_score=60, export_to_sql=True, market_regime="UNKNOWN"):
        """
        Main screening function
        """
        print("="*100)
        print("SNDK OPTION SCREENER")
        print("Combining SQL Volatility Metrics with Real-Time Option Pricing")
        print("="*100)
        
        # Step 1: Get SQL candidates
        sql_stocks = self.fetch_high_score_stocks(min_premium_score=min_sql_score)
        if sql_stocks.empty:
            print("No candidates found in SQL database")
            return pd.DataFrame()
            
        # Step 2: Analyze with options data
        results = []
        for _, row in sql_stocks.iterrows():
            analysis = self.analyze_stock(row['Symbol'], row.to_dict())
            if analysis:
                results.append(analysis)
                
        if not results:
            print("No valid option data retrieved")
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        
        # Step 3: Filter by quality score
        df_filtered = df[df['Quality_Score'] >= min_quality_score].sort_values('Quality_Score', ascending=False)
        
        # Display results
        print("\n" + "="*100)
        print(f"RESULTS: {len(df_filtered)} SNDK-like setups found (Quality Score >= {min_quality_score})")
        print("="*100)
        
        display_cols = ['Symbol', 'Stock_Price', 'SQL_Score', 'ATM_IV', 'IV_RV_Ratio', 
                       'Premium_Pct_Stock', 'Quality_Score', 'Setup_Type']
        print(df_filtered[display_cols].to_string(index=False))
        
        # Export
        if export_to_sql and not df_filtered.empty:
            try:
                df_filtered.to_sql('SNDK_Screen_Results', self.engine, if_exists='replace', index=False)
                print(f"Exported to SQL table: SNDK_Screen_Results")
            except Exception as e:
                print(f"Export failed: {e}")
                
        # Detailed analysis of top pick
        if not df_filtered.empty:
            top = df_filtered.iloc[0]
            print(f"\n" + "="*100)
            print(f"TOP SETUP: {top['Symbol']} (Score: {top['Quality_Score']}/100)")
            print("="*100)
            print(f"Price: ${top['Stock_Price']:.2f} | ATM IV: {top['ATM_IV']:.1f}% | IV/RV: {top['IV_RV_Ratio']:.2f}")
            print(f"SQL Volatility Score: {top['SQL_Score']} (Realized: {top['SQL_Realized_Vol']:.1f}%)")
            print(f"Option Premium: ${top['OTM_Premium']:.2f} ({top['Premium_Pct_Stock']:.1f}% of stock)")
            print(f"Setup Type: {top['Setup_Type']}")
            
            if top['Quality_Score'] >= 80:
                print(f"\nSuggested Strategy: Sell Apr 17 Strangle (Strike ±10%)")
                print(f"Expected Credit: ~${top['OTM_Premium']*2:.0f} per strangle")
                print(f"Edge: IV {top['IV_RV_Ratio']:.1f}x realized volatility")

        start_time = time.time()
        execution_time = time.time() - start_time
    
        if export_to_sql and not df_filtered.empty:
            self.save_run_results(df_filtered, execution_time, market_regime)  
            return df_filtered

def main():
    # Update this with your SQL Server credentials
    # db_str = "mssql+pyodbc://@BEELINK/Stock?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
    SQL_CONNECTION = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=BEELINK;"
        "DATABASE=stock;"
        "trusted_connection=yes;"
    )
    
    # Initialize and run
    screener = SNDKOptionScreener(SQL_CONNECTION)
    results = screener.run_screen(
        min_sql_score=80,      # Your SQL scanner threshold
        min_quality_score=60,   # Minimum for SNDK-like setups (80+ is ideal)
        export_to_sql=True,
        market_regime="HIGH_VOL")
# ============================================
# USAGE
# ============================================

if __name__ == "__main__":
    main()
    