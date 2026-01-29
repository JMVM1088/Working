"""
Enhanced VWAP Pullback Trading Strategy
Senior Quant Developer Production Implementation
Built for institutional execution with risk management and backtesting

Requirements:
- pandas, numpy, sqlalchemy
- SQL Server with tables: AI_stock_prices, [AI_Stock_EnhanceAVWAP_signals]
- Python 3.8+
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

@dataclass
class StrategyConfig:
    """Centralized strategy parameters for easy tuning"""
    # Database
    DB_STR = "mssql+pyodbc://@BEELINK/Stock?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
    
    # VWAP Parameters
    VWAP_MIN_PERIOD = 30
    VWAP_MAX_PERIOD = 75
    VWAP_DEFAULT_PERIOD = 50
    VWAP_ZONE_LIMIT = 0.985  # 1.5% below VWAP
    
    # Trend Requirements
    TREND_REQUIRED_DAYS = 3  # 3 of last 5 days above VWAP
    TREND_WINDOW = 5
    
    # Volatility Regime (annualized)
    VOL_MIN = 0.12  # 12%
    VOL_MAX = 0.35  # 35%
    VOL_LOOKBACK = 20
    
    # Momentum Filters
    RSI_PERIOD = 14
    RSI_OVERSOLD = 40
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Volume Filter
    VOL_RATIO_MIN = 1.2  # 20% above average
    VOL_LOOKBACK_PERIOD = 20
    
    # Exit Rules
    STOP_LOSS_PCT = 0.02  # 2% below entry
    PROFIT_TARGET_PCT = 0.03  # 3% above entry
    MAX_HOLD_DAYS = 5
    
    # Backtesting
    PROFIT_FACTOR_MIN = 1.5  # Minimum acceptable profit factor
    DATA_LOOKBACK_DAYS = 250
    MIN_PERIODS_FOR_CALC = 10


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class TechnicalIndicators:
    """Collection of technical indicator calculations"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        Measures momentum: values < 40 indicate oversold
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        MACD < Signal Line indicates potential downside reversal
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    @staticmethod
    def calculate_vwap_period(volatility: float, config: StrategyConfig) -> int:
        """
        Adaptive VWAP period based on volatility regime
        Higher volatility -> longer period (reduce noise)
        Lower volatility -> shorter period (more responsive)
        """
        if volatility < 0.15:
            return config.VWAP_MIN_PERIOD  # Calm: 30-day
        elif volatility < 0.25:
            return config.VWAP_DEFAULT_PERIOD  # Normal: 50-day
        else:
            return config.VWAP_MAX_PERIOD  # Volatile: 75-day
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, lookback: int = 20) -> pd.Series:
        """
        Annualized volatility from daily returns
        """
        return returns.rolling(lookback).std() * np.sqrt(252)


# ============================================================================
# VWAP STRATEGY ENGINE
# ============================================================================

class VWAPPullbackStrategy:
    """
    Institutional-grade VWAP pullback strategy with:
    - Adaptive VWAP periods
    - Volatility regime filtering
    - Momentum confirmation (RSI + MACD)
    - Volume confirmation
    - Risk/reward analysis
    - Comprehensive backtesting metrics
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.engine = create_engine(self.config.DB_STR)
        self.indicators = TechnicalIndicators()
    
    def fetch_market_data(self) -> pd.DataFrame:
        """
        Fetch price data from SQL Server
        Returns: DataFrame with OHLCV data sorted by Symbol and time
        """
        logger.info("Step 1: Fetching market data from SQL Server...")
        
        try:
            query = f"""
            SELECT 
                Symbol, 
                [time], 
                [open],
                [high], 
                [low], 
                [close], 
                [Volume]
            FROM AI_stock_prices 
            WHERE [time] >= DATEADD(day, -{self.config.DATA_LOOKBACK_DAYS}, CAST(GETDATE() AS DATE))
            ORDER BY Symbol, [time]
            """
            
            df = pd.read_sql(query, self.engine)
            df['time'] = pd.to_datetime(df['time'])
            
            if df.empty:
                logger.error("❌ No data found in AI_stock_prices table")
                raise ValueError("Empty dataset returned")
            
            logger.info(f"✓ Loaded {len(df):,} records for {df['Symbol'].nunique()} symbols")
            return df
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {str(e)}")
            raise
    
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all strategy signals per symbol
        """
        logger.info("Step 2: Calculating strategy signals...")
        
        results = []
        
        for symbol, group in df.groupby('Symbol'):
            group = group.sort_values('time').reset_index(drop=True)
            
            try:
                # 1. VWAP CALCULATION (True daily VWAP)
                group = self._calculate_vwap(group)
                
                # 2. VOLATILITY REGIME
                group = self._calculate_volatility_regime(group)
                
                # 3. TREND SCORING
                group = self._calculate_trend(group)
                
                # 4. MOMENTUM INDICATORS
                group = self._calculate_momentum(group)
                
                # 5. VOLUME CONFIRMATION
                group = self._calculate_volume_filter(group)
                
                # 6. COMPOSITE SIGNAL
                group = self._calculate_signal(group)
                
                # 7. EXIT RULES & FORWARD RETURNS
                group = self._calculate_forward_returns(group)
                
                results.append(group)
                
                # Log per-symbol summary
                signals_today = group[group['is_pullback'] == 1]
                if len(signals_today) > 0:
                    logger.info(f"  {symbol}: {len(signals_today)} signals generated")
                
            except Exception as e:
                logger.warning(f"  ⚠️  {symbol} processing failed: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No symbols processed successfully")
        
        final_data = pd.concat(results, ignore_index=True)
        logger.info(f"✓ Signals calculated for {len(final_data):,} total records")
        
        return final_data
    
    def _calculate_vwap(self, group: pd.DataFrame) -> pd.DataFrame:
        """Calculate true daily VWAP with adaptive period"""
        group['hlc3'] = (group['high'] + group['low'] + group['close']) / 3
        group['pv'] = group['hlc3'] * group['Volume']
        
        # Daily cumulative sum (true VWAP)
        group['cum_pv'] = group['pv'].cumsum()
        group['cum_vol'] = group['Volume'].cumsum()
        
        # Avoid division by zero
        group['vwap'] = np.where(
            group['cum_vol'] > 0,
            group['cum_pv'] / group['cum_vol'],
            group['close']
        )
        
        # Store VWAP for reference
        group['vwap_close'] = group['close']
        
        return group
    
    def _calculate_volatility_regime(self, group: pd.DataFrame) -> pd.DataFrame:
        """Calculate annualized volatility and determine regime"""
        group['returns'] = group['close'].pct_change()
        group['volatility'] = self.indicators.calculate_volatility(
            group['returns'], 
            self.config.VOL_LOOKBACK
        )
        
        # Volatility regime filtering
        group['vol_regime_ok'] = (
            (group['volatility'] >= self.config.VOL_MIN) &
            (group['volatility'] <= self.config.VOL_MAX)
        ).astype(int)
        
        # Adaptive VWAP period based on volatility
        group['vwap_period'] = group['volatility'].apply(
            lambda vol: self.indicators.calculate_vwap_period(vol, self.config)
        )
        
        return group
    
    def _calculate_trend(self, group: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend: days above VWAP in rolling window"""
        group['is_above_vwap'] = (group['close'] > group['vwap']).astype(int)
        group['trend_score'] = group['is_above_vwap'].rolling(
            window=self.config.TREND_WINDOW,
            min_periods=1
        ).sum()
        
        return group
    
    def _calculate_momentum(self, group: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators: RSI and MACD"""
        # RSI
        group['rsi'] = self.indicators.calculate_rsi(
            group['close'],
            self.config.RSI_PERIOD
        )
        group['rsi_oversold'] = (group['rsi'] < self.config.RSI_OVERSOLD).astype(int)
        
        # MACD
        group['macd'], group['macd_signal'] = self.indicators.calculate_macd(
            group['close'],
            self.config.MACD_FAST,
            self.config.MACD_SLOW,
            self.config.MACD_SIGNAL
        )
        group['macd_below_signal'] = (group['macd'] < group['macd_signal']).astype(int)
        
        # Momentum confirmation: RSI OR MACD signal
        group['momentum_ok'] = (
            (group['rsi_oversold'] == 1) | (group['macd_below_signal'] == 1)
        ).astype(int)
        
        return group
    
    def _calculate_volume_filter(self, group: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume confirmation: above-average volume"""
        group['vol_ma'] = group['Volume'].rolling(
            window=self.config.VOL_LOOKBACK_PERIOD,
            min_periods=1
        ).mean()
        group['vol_ratio'] = group['Volume'] / group['vol_ma']
        group['volume_ok'] = (
            group['vol_ratio'] >= self.config.VOL_RATIO_MIN
        ).astype(int)
        
        return group
    
    def _calculate_signal(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Composite signal: All conditions must be met
        1. Trend intact (3+ of last 5 days above VWAP)
        2. Low touches VWAP
        3. Close holds value zone (within 1.5% of VWAP)
        4. Volatility in acceptable regime
        5. Momentum confirmation (RSI or MACD)
        6. Volume confirmation (above average)
        """
        group['is_pullback'] = (
            (group['trend_score'] >= self.config.TREND_REQUIRED_DAYS) &
            (group['low'] <= group['vwap']) &
            (group['close'] >= group['vwap'] * self.config.VWAP_ZONE_LIMIT) &
            (group['vol_regime_ok'] == 1) &
            (group['momentum_ok'] == 1) &
            (group['volume_ok'] == 1)
        ).astype(int)
        
        return group
    
    def _calculate_forward_returns(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate forward returns for performance tracking
        Includes multiple holding periods for analysis
        """
        group['fwd_return_1d'] = (
            group['close'].shift(-1) - group['close']
        ) / group['close']
        
        group['fwd_return_5d'] = (
            group['close'].shift(-5) - group['close']
        ) / group['close']
        
        group['fwd_return_10d'] = (
            group['close'].shift(-10) - group['close']
        ) / group['close']
        
        # Exit rules: Does price hit stop loss or profit target first?
        group['hit_stop_loss'] = (group['low'].shift(-1) <= group['close'] * (1 - self.config.STOP_LOSS_PCT)).astype(int)
        group['hit_profit_target'] = (group['high'].shift(-1) >= group['close'] * (1 + self.config.PROFIT_TARGET_PCT)).astype(int)
        
        return group
    
    def upload_signals(self, df: pd.DataFrame) -> int:
        """
        Upload calculated signals to SQL Server
        Only uploads records where signal was generated
        """
        logger.info("Step 3: Uploading signals to SQL Server...")
        
        try:
            # Select only necessary columns
            output_cols = [
                'Symbol', 'time', 'open', 'high', 'low', 'close', 'Volume',
                'vwap', 'volatility', 'vol_regime_ok',
                'trend_score', 'rsi', 'macd', 'volume_ok',
                'is_pullback', 'fwd_return_1d', 'fwd_return_5d', 'fwd_return_10d',
                'hit_stop_loss', 'hit_profit_target'
            ]
            
            upload_df = df[output_cols].dropna(subset=['vwap'])
            upload_df = upload_df[upload_df['is_pullback'] == 1].copy()
            
            if len(upload_df) == 0:
                logger.warning("⚠️  No signals generated for upload")
                return 0
            
            # Upload to SQL Server
            # upload_df.to_sql(
            #     "AI_Stock_EnhanceAVWAP_signals",
            #     self.engine,
            #     if_exists='append',
            #     index=False,
            #     chunksize=1000
            # )
            ##==================
            # Check for existing records
            try:
                existing = pd.read_sql(
                    "SELECT DISTINCT Symbol, time FROM AI_Stock_EnhanceAVWAP_signals",
                    self.engine
                )
                existing['key'] = existing['Symbol'].astype(str) + '|' + existing['time'].astype(str)
            except:
                existing = pd.DataFrame({'key': []})

            # Filter out duplicates
            upload_df['key'] = upload_df['Symbol'].astype(str) + '|' + upload_df['time'].astype(str)
            new_df = upload_df[~upload_df['key'].isin(existing['key'].values)].drop('key', axis=1)

            if len(new_df) > 0:
                new_df.to_sql(
                    "AI_Stock_EnhanceAVWAP_signals",
                    self.engine,
                    if_exists='append',  # ✓ Now only appending NEW records
                    index=False,
                    chunksize=1000
                )
                logger.info(f"✓ Uploaded {len(new_df):,} new records (skipped {len(upload_df) - len(new_df):,} duplicates)")
            else:
                logger.warning(f"⚠️  All {len(upload_df)} records already exist")


            ###==========================
            logger.info(f"✓ Uploaded {len(upload_df):,} signal records")
            return len(upload_df)
            
        except Exception as e:
            logger.error(f"❌ Upload failed: {str(e)}")
            raise
    
    def generate_backtest_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive backtesting metrics
        """
        logger.info("Step 4: Generating backtest report...")
        
        # Filter to signals only
        signals_df = df[df['is_pullback'] == 1].copy()
        
        if len(signals_df) == 0:
            logger.warning("⚠️  No signals to analyze")
            return {}
        
        # Calculate metrics
        winning_trades = signals_df[signals_df['fwd_return_5d'] > 0]
        losing_trades = signals_df[signals_df['fwd_return_5d'] <= 0]
        
        win_rate = len(winning_trades) / len(signals_df) if len(signals_df) > 0 else 0
        avg_win = winning_trades['fwd_return_5d'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['fwd_return_5d'].mean() if len(losing_trades) > 0 else 0
        
        # Profit Factor (gross profit / gross loss)
        gross_profit = winning_trades['fwd_return_5d'].sum() * 100
        gross_loss = abs(losing_trades['fwd_return_5d'].sum()) * 100
        profit_factor = (
            gross_profit / gross_loss 
            if gross_loss > 0 else np.inf
        )
        
        # Expectancy per trade
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Sharpe Ratio (annualized, assuming 252 trading days)
        excess_returns = signals_df['fwd_return_5d'] - 0.00  # Risk-free ≈ 0
        sharpe = (
            (excess_returns.mean() / excess_returns.std()) * np.sqrt(252 / self.config.MAX_HOLD_DAYS)
            if excess_returns.std() > 0 else 0
        )
        
        # Max Drawdown
        cumulative = (1 + signals_df['fwd_return_5d']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win/Loss statistics
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for ret in signals_df['fwd_return_5d']:
            if ret > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Compile metrics
        metrics = {
            'total_signals': len(signals_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'profit_factor': profit_factor,
            'expectancy_pct': expectancy * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd * 100,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'best_trade_pct': signals_df['fwd_return_5d'].max() * 100,
            'worst_trade_pct': signals_df['fwd_return_5d'].min() * 100,
        }
        
        return metrics
    
    def print_report(self, metrics: Dict) -> None:
        """Pretty-print backtesting report"""
        if not metrics:
            return
        
        print("\n" + "="*70)
        print("VWAP PULLBACK STRATEGY - BACKTEST REPORT")
        print("="*70)
        print(f"\nTrades Summary:")
        print(f"  Total Signals:            {metrics['total_signals']:,}")
        print(f"  Winning Trades:           {metrics['winning_trades']:,}")
        print(f"  Losing Trades:            {metrics['losing_trades']:,}")
        print(f"  Win Rate:                 {metrics['win_rate']*100:.1f}%")
        
        print(f"\nP&L Analysis:")
        print(f"  Avg Win:                  +{metrics['avg_win_pct']:.2f}%")
        print(f"  Avg Loss:                 {metrics['avg_loss_pct']:.2f}%")
        print(f"  Profit Factor:            {metrics['profit_factor']:.2f}x")
        print(f"  Expectancy per Trade:     {metrics['expectancy_pct']:.2f}%")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:             {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:             {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Max Consecutive Wins:     {metrics['max_consecutive_wins']}")
        print(f"  Max Consecutive Losses:   {metrics['max_consecutive_losses']}")
        
        print(f"\nExtreme Trades:")
        print(f"  Best Trade:               +{metrics['best_trade_pct']:.2f}%")
        print(f"  Worst Trade:              {metrics['worst_trade_pct']:.2f}%")
        
        print("\n" + "="*70)
        
        # Warnings
        if metrics['profit_factor'] < self.config.PROFIT_FACTOR_MIN:
            print(f"⚠️  WARNING: Profit Factor {metrics['profit_factor']:.2f} below minimum {self.config.PROFIT_FACTOR_MIN}")
            print("   System may be net negative after slippage/commissions")
        
        if metrics['win_rate'] < 0.45:
            print(f"⚠️  WARNING: Win Rate {metrics['win_rate']*100:.1f}% is low")
            print("   Consider adding more filters or adjusting parameters")
        
        if metrics['sharpe_ratio'] < 0.5:
            print(f"⚠️  WARNING: Sharpe Ratio {metrics['sharpe_ratio']:.2f} is weak")
            print("   Returns may not justify risk taken")
        
        print()
    
    def _calculate_vol_ratio(self) -> None:
        """Auto-calculate vol_ratio for new signals"""
        query = """
        WITH vol_stats AS (
            SELECT 
                [signal_id],
                [Volume],
                AVG([Volume]) OVER (
                    PARTITION BY [Symbol] 
                    ORDER BY [time] ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS vol_ma_20
            FROM [dbo].[AI_Stock_EnhanceAVWAP_signals]
            WHERE [vol_ratio] IS NULL
        )
        UPDATE [dbo].[AI_Stock_EnhanceAVWAP_signals]
        SET [vol_ratio] = CASE 
            WHEN vs.[vol_ma_20] > 0 THEN vs.[Volume] / vs.[vol_ma_20]
            ELSE 1.0
        END
        FROM vol_stats vs
        WHERE [dbo].[AI_Stock_EnhanceAVWAP_signals].[signal_id] = vs.[signal_id];
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(query))
                conn.commit()
            logger.info("✓ vol_ratio calculated for new signals")
        except Exception as e:
            logger.warning(f"vol_ratio calculation failed: {str(e)}")

    def run_production_pipeline(self) -> None:
        """Execute full pipeline: fetch -> calculate -> upload -> report"""
        try:
            logger.info("\n" + "="*70)
            logger.info("VWAP PULLBACK STRATEGY - PRODUCTION RUN")
            logger.info("="*70 + "\n")
            
            # Fetch data
            df = self.fetch_market_data()
            
            # Calculate signals
            df = self.calculate_signals(df)
            
            # Upload to database
            num_uploaded = self.upload_signals(df)

            # NEW: Calculate vol_ratio for scoring
            self._calculate_vol_ratio()
            
            # Generate report
            metrics = self.generate_backtest_report(df)
            self.print_report(metrics)
            
            logger.info("\n✓ Production run completed successfully\n")
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {str(e)}\n")
            sys.exit(1)

def DailyRun():
    run_date = datetime.now().date()
    logging.info(f"=== VWAP Daily Run for {run_date} ===")
    config = StrategyConfig()
    # Optional: restrict to last N days of market data for speed
    config.DATA_LOOKBACK_DAYS = 260
    strategy = VWAPPullbackStrategy(config)
    strategy.run_production_pipeline()



#-================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    DailyRun()
    # Below is for historical backtest runs
    # config = StrategyConfig()
    # strategy = VWAPPullbackStrategy(config)
    # strategy.run_production_pipeline()
