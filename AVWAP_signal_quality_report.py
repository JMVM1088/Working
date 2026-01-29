"""
Daily Signal Quality Report & Automated Trader Alerts
Companion to signal_scoring_system.sql

Run this after vwap_strategy.py completes to:
1. Score all new signals
2. Generate trader-friendly report
3. Send alerts for high-quality signals
4. Archive quality metrics for monitoring

Usage:
    python signal_quality_report.py
    
Or integrate into your daily pipeline:
    python vwap_strategy.py && python signal_quality_report.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class SignalQualityAnalyzer:
    """Analyze signal quality and generate trader reports"""
    
    def __init__(self, db_str: str = None):
        if db_str is None:
            db_str = "mssql+pyodbc://@BEELINK/Stock?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
        self.engine = create_engine(db_str)
    
    def fetch_todays_signals(self) -> pd.DataFrame:
        """Fetch all signals generated today with quality metrics"""
        #today = datetime.now().date()
        today = date(2026,1,16)
        query = f"""
        SELECT 
            [signal_id], [Symbol], [time], [close], [vwap], [rsi], [macd],
            [trend_score], [Volume], [volatility],
            [fwd_return_1d], [fwd_return_5d], [fwd_return_10d],
            [hit_stop_loss], [hit_profit_target]
        FROM [dbo].[AI_Stock_EnhanceAVWAP_signals]
        WHERE [is_pullback] = 1
            AND [time] = '{today}'
        ORDER BY [Symbol], [time]
        """
        
        df = pd.read_sql(query, self.engine)
        logger.info(f"✓ Fetched {len(df)} signals from today ({today})")
        return df
    
    def calculate_vol_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume ratio (today's vol / 20-day avg)"""
        # This should already be in DB, but calculate backup
        query = """
        SELECT 
            [signal_id],
            [Volume],
            AVG([Volume]) OVER (PARTITION BY [Symbol] ORDER BY [time] 
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_ma_20
        FROM [dbo].[AI_Stock_AVWAP_signals]
        WHERE [is_pullback] = 1
        """
        
        vol_df = pd.read_sql(query, self.engine)
        vol_df['vol_ratio'] = vol_df['Volume'] / vol_df['vol_ma_20']
        
        return vol_df[['signal_id', 'vol_ratio']]
    
    def score_signals_Old(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score each signal on 0-100 scale
        Components:
        - Momentum (RSI): 0-25 points
        - Volume: 0-25 points
        - Trend: 0-25 points
        - Volatility Regime: 0-25 points
        """
        df = df.copy()
        
        # Merge volume ratio
        vol_ratio_df = self.calculate_vol_ratio(df)
        df = df.merge(vol_ratio_df, on='signal_id', how='left')
        df['vol_ratio'] = df['vol_ratio'].fillna(1.0)
        
        # MOMENTUM SCORE (RSI)
        df['momentum_score'] = 0
        df.loc[df['rsi'] < 25, 'momentum_score'] = 25
        df.loc[(df['rsi'] >= 25) & (df['rsi'] < 30), 'momentum_score'] = 23
        df.loc[(df['rsi'] >= 30) & (df['rsi'] < 35), 'momentum_score'] = 20
        df.loc[(df['rsi'] >= 35) & (df['rsi'] < 40), 'momentum_score'] = 17
        df.loc[(df['rsi'] >= 40) & (df['rsi'] < 45), 'momentum_score'] = 12
        df.loc[(df['rsi'] >= 45) & (df['rsi'] < 50), 'momentum_score'] = 8
        
        # VOLUME SCORE
        df['volume_score'] = 0
        df.loc[df['vol_ratio'] >= 2.0, 'volume_score'] = 25
        df.loc[(df['vol_ratio'] >= 1.7) & (df['vol_ratio'] < 2.0), 'volume_score'] = 22
        df.loc[(df['vol_ratio'] >= 1.5) & (df['vol_ratio'] < 1.7), 'volume_score'] = 20
        df.loc[(df['vol_ratio'] >= 1.3) & (df['vol_ratio'] < 1.5), 'volume_score'] = 17
        df.loc[(df['vol_ratio'] >= 1.2) & (df['vol_ratio'] < 1.3), 'volume_score'] = 14
        df.loc[(df['vol_ratio'] >= 1.1) & (df['vol_ratio'] < 1.2), 'volume_score'] = 10
        df.loc[df['vol_ratio'] < 1.1, 'volume_score'] = 5
        
        # TREND SCORE
        df['trend_persistence_score'] = 0
        df.loc[df['trend_score'] >= 5, 'trend_persistence_score'] = 25
        df.loc[df['trend_score'] == 4, 'trend_persistence_score'] = 22
        df.loc[df['trend_score'] == 3, 'trend_persistence_score'] = 18
        df.loc[df['trend_score'] == 2, 'trend_persistence_score'] = 10
        
        # VOLATILITY SCORE
        df['volatility_score'] = 0
        df.loc[(df['volatility'] >= 0.15) & (df['volatility'] <= 0.30), 'volatility_score'] = 25
        df.loc[(df['volatility'] >= 0.12) & (df['volatility'] < 0.15), 'volatility_score'] = 20
        df.loc[(df['volatility'] > 0.30) & (df['volatility'] <= 0.35), 'volatility_score'] = 20
        df.loc[(df['volatility'] >= 0.08) & (df['volatility'] < 0.12), 'volatility_score'] = 12
        df.loc[(df['volatility'] > 0.35) & (df['volatility'] <= 0.40), 'volatility_score'] = 10
        df.loc[df['volatility'] < 0.08, 'volatility_score'] = 2
        df.loc[df['volatility'] > 0.40, 'volatility_score'] = 2
        
        # TOTAL QUALITY SCORE
        df['total_quality_score'] = (
            df['momentum_score'] + 
            df['volume_score'] + 
            df['trend_persistence_score'] + 
            df['volatility_score']
        )
        
        # QUALITY TIER
        df['quality_tier'] = 'D Poor'
        df.loc[df['total_quality_score'] >= 90, 'quality_tier'] = 'A+ Elite'
        df.loc[(df['total_quality_score'] >= 80) & (df['total_quality_score'] < 90), 'quality_tier'] = 'A Excellent'
        df.loc[(df['total_quality_score'] >= 70) & (df['total_quality_score'] < 80), 'quality_tier'] = 'B+ Good'
        df.loc[(df['total_quality_score'] >= 60) & (df['total_quality_score'] < 70), 'quality_tier'] = 'B Fair'
        df.loc[(df['total_quality_score'] >= 50) & (df['total_quality_score'] < 60), 'quality_tier'] = 'C Marginal'
        
        return df
    
    def score_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Updated scoring based on data-driven findings"""
        df = df.copy()
        
        # RSI SCORE (0-50 points) - DOMINANT
        df['rsi_score'] = 0
        df.loc[df['rsi'] < 25, 'rsi_score'] = 50
        df.loc[(df['rsi'] >= 25) & (df['rsi'] < 30), 'rsi_score'] = 48
        df.loc[(df['rsi'] >= 30) & (df['rsi'] < 35), 'rsi_score'] = 42
        df.loc[(df['rsi'] >= 35) & (df['rsi'] < 40), 'rsi_score'] = 35
        df.loc[(df['rsi'] >= 40) & (df['rsi'] < 45), 'rsi_score'] = 25
        df.loc[(df['rsi'] >= 45) & (df['rsi'] < 50), 'rsi_score'] = 15
        
        # VOLATILITY SCORE (0-30) - PREFER 12-15%
        df['vol_score'] = 0
        df.loc[(df['volatility'] >= 0.12) & (df['volatility'] < 0.15), 'vol_score'] = 30
        df.loc[(df['volatility'] >= 0.15) & (df['volatility'] < 0.20), 'vol_score'] = 20
        df.loc[(df['volatility'] >= 0.20) & (df['volatility'] < 0.30), 'vol_score'] = 22
        df.loc[(df['volatility'] >= 0.30) & (df['volatility'] < 0.35), 'vol_score'] = 20
        df.loc[df['volatility'] < 0.12, 'vol_score'] = 15
        df.loc[df['volatility'] >= 0.35, 'vol_score'] = 10
        
        # TREND SCORE (0-20) - EARLY PULLBACKS BEST
        df['trend_component'] = 0
        df.loc[df['trend_score'] == 3, 'trend_component'] = 20  # BEST (56.3% win)
        df.loc[df['trend_score'] == 4, 'trend_component'] = 18
        df.loc[df['trend_score'] == 2, 'trend_component'] = 10
        df.loc[df['trend_score'] == 5, 'trend_component'] = 5   # WORST (52.9% win)
        df.loc[df['trend_score'] == 1, 'trend_component'] = 2
        
        # TOTAL SCORE (0-100, NO VOLUME)
        df['total_quality_score'] = df['rsi_score'] + df['vol_score'] + df['trend_component']
        
        # QUALITY TIER
        df['quality_tier'] = 'D Poor'
        df.loc[df['total_quality_score'] >= 90, 'quality_tier'] = 'A+ Elite'
        df.loc[(df['total_quality_score'] >= 80) & (df['total_quality_score'] < 90), 'quality_tier'] = 'A Excellent'
        df.loc[(df['total_quality_score'] >= 70) & (df['total_quality_score'] < 80), 'quality_tier'] = 'B+ Good'
        df.loc[(df['total_quality_score'] >= 60) & (df['total_quality_score'] < 70), 'quality_tier'] = 'B Fair'
        df.loc[(df['total_quality_score'] >= 50) & (df['total_quality_score'] < 60), 'quality_tier'] = 'C Marginal'
        
        return df



    def generate_daily_report(self, df: pd.DataFrame) -> None:
        """Print trader-friendly daily report"""
        if len(df) == 0:
            logger.info("No signals generated today")
            return
        
        print("\n" + "="*80)
        print(f"DAILY SIGNAL QUALITY REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Summary statistics
        print(f"\nSIGNAL SUMMARY:")
        print(f"  Total Signals:           {len(df):,}")
        
        tier_counts = df['quality_tier'].value_counts()
        
        a_plus = len(df[df['quality_tier'] == 'A+ Elite'])
        a = len(df[df['quality_tier'] == 'A Excellent'])
        b_plus = len(df[df['quality_tier'] == 'B+ Good'])
        b = len(df[df['quality_tier'] == 'B Fair'])
        c_plus = len(df[df['quality_tier'] == 'C Marginal'])
        d = len(df[df['quality_tier'] == 'D Poor'])
        
        elite_tier = a_plus + a
        good_tier = b_plus + b
        poor_tier = c_plus + d
        
        print(f"  A-Tier (Elite+Excellent): {elite_tier:,} ({elite_tier/len(df)*100:.1f}%)")
        print(f"    └─ A+ Elite:             {a_plus:,}")
        print(f"    └─ A Excellent:          {a:,}")
        print(f"  B-Tier (Good+Fair):       {good_tier:,} ({good_tier/len(df)*100:.1f}%)")
        print(f"    └─ B+ Good:              {b_plus:,}")
        print(f"    └─ B Fair:               {b:,}")
        print(f"  C-Tier (Marginal):        {c_plus:,}")
        print(f"  D-Tier (Poor):            {d:,}")
        
        # Top signals by quality
        print(f"\nTOP 10 HIGHEST QUALITY SIGNALS:")
        print("-" * 100)
        
        top_signals = df.nlargest(10, 'total_quality_score')[
            ['Symbol', 'close', 'vwap', 'rsi', 'trend_score', 'vol_ratio', 
             'volatility', 'total_quality_score', 'quality_tier']
        ].copy()
        
        top_signals = top_signals.rename(columns={
            'close': 'Entry',
            'vwap': 'VWAP',
            'rsi': 'RSI',
            'trend_score': 'Trend',
            'vol_ratio': 'VolRatio',
            'volatility': 'Vol%',
            'total_quality_score': 'Score',
            'quality_tier': 'Tier'
        })
        
        for idx, row in top_signals.iterrows():
            print(f"{row['Symbol']:5} | Entry: {row['Entry']:8.2f} | VWAP: {row['VWAP']:8.2f} | "
                  f"RSI: {row['RSI']:5.1f} | Vol: {row['Vol%']*100:5.1f}% | "
                  f"Score: {row['Score']:3.0f} | {row['Tier']}")
        
        # Performance analysis (if we have forward returns)
        print(f"\nQUALITY TIER PERFORMANCE (if we have prior data):")
        print("-" * 80)
        
        for tier in ['A+ Elite', 'A Excellent', 'B+ Good', 'B Fair', 'C Marginal', 'D Poor']:
            tier_df = df[df['quality_tier'] == tier]
            if len(tier_df) == 0:
                continue
            
            # Use fwd_return_5d if available
            if 'fwd_return_5d' in tier_df.columns:
                winners = len(tier_df[tier_df['fwd_return_5d'] > 0])
                total = len(tier_df)
                if total > 0:
                    win_rate = winners / total * 100
                    avg_return = tier_df['fwd_return_5d'].mean() * 100
                    print(f"{tier:20} | Signals: {total:3} | Win Rate: {win_rate:5.1f}% | "
                          f"Avg Return: {avg_return:+6.2f}%")
        
        # Symbol distribution
        print(f"\nSIGNALS BY SYMBOL (Top 10):")
        print("-" * 60)
        
        symbol_counts = df['Symbol'].value_counts().head(10)
        for symbol, count in symbol_counts.items():
            symbol_df = df[df['Symbol'] == symbol]
            avg_score = symbol_df['total_quality_score'].mean()
            print(f"  {symbol:6} | Count: {count:3} | Avg Score: {avg_score:5.1f}")
        
        print("\n" + "="*80 + "\n")
    
    def save_quality_metrics(self, df: pd.DataFrame) -> None:
        """Archive quality scores for historical analysis"""
        metrics = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_signals': len(df),
            'a_tier_count': len(df[df['quality_tier'].str.startswith('A')]),
            'b_tier_count': len(df[df['quality_tier'].str.startswith('B')]),
            'c_tier_count': len(df[df['quality_tier'].str.startswith('C')]),
            'd_tier_count': len(df[df['quality_tier'] == 'D Poor']),
            'avg_quality_score': float(df['total_quality_score'].mean()),
            'max_quality_score': float(df['total_quality_score'].max()),
            'min_quality_score': float(df['total_quality_score'].min()),
            'std_quality_score': float(df['total_quality_score'].std()),
        }
        
        # Append to JSON file for tracking
        history_file = 'signal_quality_history.json'
        
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        
        history.append(metrics)
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"✓ Quality metrics saved to {history_file}")
    
    def filter_elite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only A-tier signals (score >= 80)"""
        return df[df['total_quality_score'] >= 80].sort_values('total_quality_score', ascending=False)
    
    def send_trader_alerts(self, df: pd.DataFrame) -> None:
        """Send alerts for highest quality signals (optional integration with Slack/Email)"""
        elite_signals = self.filter_elite_signals(df)
        
        if len(elite_signals) == 0:
            logger.info("No A-tier signals today to alert")
            return
        
        logger.info(f"\n🔔 TRADER ALERTS: {len(elite_signals)} A-Tier signals detected\n")
        
        for idx, signal in elite_signals.iterrows():
            alert = (
                f"🟢 {signal['Symbol']} | Score: {signal['total_quality_score']:.0f} ({signal['quality_tier']}) | "
                f"Entry: {signal['close']:.2f} | VWAP: {signal['vwap']:.2f} | "
                f"RSI: {signal['rsi']:.1f} | Vol: {signal['vol_ratio']:.2f}x"
            )
            logger.info(alert)
        
        # Optional: Send Slack message (implement as needed)
        # self.send_slack_notification(elite_signals)
    
    def run_full_analysis(self) -> Dict:
        """Execute complete daily quality analysis"""
        logger.info("\n" + "="*80)
        logger.info("STARTING DAILY SIGNAL QUALITY ANALYSIS")
        logger.info("="*80 + "\n")
        
        # Fetch today's signals
        df = self.fetch_todays_signals()
        if len(df) == 0:
            logger.warning("No signals generated today")
            return {}
        
        # Score all signals
        logger.info("Calculating quality scores...")
        df = self.score_signals(df)
        
        # Generate reports
        self.generate_daily_report(df)
        self.save_quality_metrics(df)
        self.send_trader_alerts(df)
        
        # Return elite signals for further action
        elite = self.filter_elite_signals(df)
        
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_signals': len(df),
            'elite_signals': len(elite),
            'avg_quality_score': float(df['total_quality_score'].mean()),
            'elite_symbols': elite['Symbol'].unique().tolist() if len(elite) > 0 else []
        }
        
        logger.info(f"✓ Analysis complete. Elite signals: {len(elite)} / {len(df)}")
        return summary


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    analyzer = SignalQualityAnalyzer()
    summary = analyzer.run_full_analysis()
