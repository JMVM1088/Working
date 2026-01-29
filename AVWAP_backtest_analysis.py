"""
Backtesting & Analysis Utilities for VWAP Strategy
Run this to analyze strategy performance post-production

Usage:
    from backtest_analysis import StrategyAnalyzer
    analyzer = StrategyAnalyzer()
    analyzer.full_backtest_analysis()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyAnalyzer:
    """Comprehensive backtesting and performance analysis"""
    
    def __init__(self, db_str: str = None):
        if db_str is None:
            db_str = "mssql+pyodbc://@BEELINK/Stock?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
        
        self.engine = create_engine(db_str)
        self.signals_df = None
        self.metrics = {}
    
    def fetch_signals(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch signals from database"""
        logger.info("Fetching signals from database...")
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            [signal_id], [Symbol], [time], [close], [Volume],
            [vwap], [volatility], [trend_score], [rsi], [macd],
            [volume_ok], [is_pullback], 
            [fwd_return_1d], [fwd_return_5d], [fwd_return_10d],
            [hit_stop_loss], [hit_profit_target]
        FROM [dbo].[AI_Stock_AVWAP_signals]
        WHERE [is_pullback] = 1
            AND [time] >= '{start_date}'
            AND [time] <= '{end_date}'
        ORDER BY [Symbol], [time]
        """
        
        self.signals_df = pd.read_sql(query, self.engine)
        self.signals_df['time'] = pd.to_datetime(self.signals_df['time'])
        
        logger.info(f"Loaded {len(self.signals_df)} signals from {start_date} to {end_date}")
        return self.signals_df
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtesting metrics"""
        logger.info("Calculating backtest metrics...")
        
        if self.signals_df is None or len(self.signals_df) == 0:
            logger.error("No signals loaded")
            return {}
        
        df = self.signals_df.copy()
        
        # Basic counts
        total = len(df)
        winning = len(df[df['fwd_return_5d'] > 0])
        losing = len(df[df['fwd_return_5d'] <= 0])
        
        # Win/Loss statistics
        avg_win = df[df['fwd_return_5d'] > 0]['fwd_return_5d'].mean()
        avg_loss = df[df['fwd_return_5d'] <= 0]['fwd_return_5d'].mean()
        
        win_rate = winning / total if total > 0 else 0
        
        # Profit Factor
        gross_profit = (df[df['fwd_return_5d'] > 0]['fwd_return_5d'].sum())
        gross_loss = abs(df[df['fwd_return_5d'] <= 0]['fwd_return_5d'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Sharpe Ratio (252 trading days, 5-day holding period)
        returns = df['fwd_return_5d'].dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 / 5) if returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for ret in df['fwd_return_5d']:
            if ret > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        self.metrics = {
            'total_signals': total,
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'best_trade': df['fwd_return_5d'].max(),
            'worst_trade': df['fwd_return_5d'].min(),
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'expectancy_pct': expectancy * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_dd_pct': max_dd * 100,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
        }
        
        return self.metrics
    
    def print_metrics(self) -> None:
        """Pretty print metrics"""
        if not self.metrics:
            return
        
        m = self.metrics
        
        print("\n" + "="*70)
        print("VWAP PULLBACK STRATEGY - BACKTEST REPORT")
        print("="*70)
        
        print(f"\nTRADES SUMMARY:")
        print(f"  Total Signals:            {m['total_signals']:,}")
        print(f"  Winning Trades:           {m['winning_trades']:,}")
        print(f"  Losing Trades:            {m['losing_trades']:,}")
        print(f"  Win Rate:                 {m['win_rate']*100:.1f}%")
        
        print(f"\nP&L ANALYSIS:")
        print(f"  Avg Win:                  +{m['avg_win_pct']:.2f}%")
        print(f"  Avg Loss:                 {m['avg_loss_pct']:.2f}%")
        print(f"  Best Trade:               +{m['best_trade']*100:.2f}%")
        print(f"  Worst Trade:              {m['worst_trade']*100:.2f}%")
        print(f"  Profit Factor:            {m['profit_factor']:.2f}x")
        print(f"  Expectancy per Trade:     {m['expectancy_pct']:.2f}%")
        
        print(f"\nRISK METRICS:")
        print(f"  Sharpe Ratio:             {m['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:             {m['max_dd_pct']:.1f}%")
        print(f"  Max Consecutive Wins:     {m['max_consecutive_wins']}")
        print(f"  Max Consecutive Losses:   {m['max_consecutive_losses']}")
        
        print("\n" + "="*70)
        
        # Warnings
        if m['profit_factor'] < 1.5:
            print(f"\n⚠️  WARNING: Profit Factor {m['profit_factor']:.2f} < 1.5 minimum")
            print("   System may be net negative after fees/slippage")
        
        if m['win_rate'] < 0.45:
            print(f"\n⚠️  WARNING: Win Rate {m['win_rate']*100:.1f}% < 45%")
            print("   Consider additional filtering or parameter tuning")
        
        if m['sharpe_ratio'] < 0.5:
            print(f"\n⚠️  WARNING: Sharpe Ratio {m['sharpe_ratio']:.2f} < 0.5")
            print("   Returns may not justify risk taken")
        
        print()
    
    def analyze_by_volatility_regime(self) -> pd.DataFrame:
        """Analyze performance by volatility regime"""
        logger.info("Analyzing by volatility regime...")
        
        df = self.signals_df.copy()
        
        df['vol_regime'] = pd.cut(df['volatility'], 
            bins=[0, 0.12, 0.20, 0.30, 0.40, 1.0],
            labels=['Very Low', 'Low', 'Normal', 'High', 'Extreme']
        )
        
        analysis = df.groupby('vol_regime').agg({
            'signal_id': 'count',
            'fwd_return_5d': ['sum', 'mean', 'std'],
        }).round(4)
        
        analysis.columns = ['Total_Signals', 'Gross_Return', 'Avg_Return', 'Std_Dev']
        analysis['Win_Rate'] = (
            df[df['fwd_return_5d'] > 0].groupby('vol_regime').size() / 
            df.groupby('vol_regime').size()
        ).round(2)
        
        print("\nPERFORMANCE BY VOLATILITY REGIME:")
        print(analysis)
        
        return analysis
    
    def analyze_by_symbol(self) -> pd.DataFrame:
        """Analyze performance by symbol"""
        logger.info("Analyzing by symbol...")
        
        df = self.signals_df.copy()
        
        analysis = df.groupby('Symbol').agg({
            'signal_id': 'count',
            'fwd_return_5d': ['mean', 'std', 'min', 'max'],
            'rsi': 'mean',
            'volatility': 'mean',
        }).round(4)
        
        analysis.columns = [
            'Total_Signals', 'Avg_Return', 'Std_Dev', 
            'Min_Return', 'Max_Return', 'Avg_RSI', 'Avg_Vol'
        ]
        
        analysis['Win_Rate'] = (
            df[df['fwd_return_5d'] > 0].groupby('Symbol').size() / 
            df.groupby('Symbol').size()
        ).round(2)
        
        analysis = analysis.sort_values('Avg_Return', ascending=False)
        
        print("\nPERFORMANCE BY SYMBOL:")
        print(analysis)
        
        return analysis
    
    def analyze_by_month(self) -> pd.DataFrame:
        """Monthly performance analysis"""
        logger.info("Analyzing by month...")
        
        df = self.signals_df.copy()
        df['year_month'] = df['time'].dt.to_period('M')
        
        analysis = df.groupby('year_month').agg({
            'signal_id': 'count',
            'fwd_return_5d': ['mean', 'sum'],
            'volatility': 'mean',
        }).round(4)
        
        analysis.columns = ['Total_Signals', 'Avg_Return', 'Gross_Return', 'Avg_Vol']
        
        analysis['Win_Rate'] = (
            df[df['fwd_return_5d'] > 0].groupby('year_month').size() / 
            df.groupby('year_month').size()
        ).round(2)
        
        print("\nMONTHLY PERFORMANCE:")
        print(analysis)
        
        return analysis
    
    def generate_equity_curve(self) -> None:
        """Plot equity curve from signals"""
        logger.info("Generating equity curve...")
        
        df = self.signals_df.copy()
        df = df.sort_values('time')
        df['cumulative_return'] = (1 + df['fwd_return_5d']).cumprod()
        
        plt.figure(figsize=(14, 6))
        plt.plot(df['time'], df['cumulative_return'] * 100, linewidth=2)
        plt.title('Strategy Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('equity_curve.png', dpi=150)
        logger.info("✓ Saved equity_curve.png")
        plt.close()
    
    def generate_distribution_plot(self) -> None:
        """Plot distribution of returns"""
        logger.info("Generating return distribution...")
        
        df = self.signals_df.copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Return distribution
        axes[0].hist(df['fwd_return_5d'] * 100, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(df['fwd_return_5d'].mean() * 100, color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {df["fwd_return_5d"].mean()*100:.2f}%')
        axes[0].set_title('Distribution of 5-Day Returns')
        axes[0].set_xlabel('Return (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Win vs Loss distribution
        winning = df[df['fwd_return_5d'] > 0]['fwd_return_5d'] * 100
        losing = df[df['fwd_return_5d'] <= 0]['fwd_return_5d'] * 100
        
        axes[1].hist([winning, losing], bins=30, label=['Winning', 'Losing'], 
                    edgecolor='black', alpha=0.7)
        axes[1].set_title('Winning vs Losing Trade Distribution')
        axes[1].set_xlabel('Return (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('return_distribution.png', dpi=150)
        logger.info("✓ Saved return_distribution.png")
        plt.close()
    
    def full_backtest_analysis(self, start_date: str = None, end_date: str = None) -> None:
        """Run complete backtest analysis"""
        logger.info("\n" + "="*70)
        logger.info("STARTING FULL BACKTEST ANALYSIS")
        logger.info("="*70 + "\n")
        
        # Fetch and analyze
        self.fetch_signals(start_date, end_date)
        self.calculate_metrics()
        self.print_metrics()
        
        # Detailed analysis
        self.analyze_by_volatility_regime()
        self.analyze_by_symbol()
        self.analyze_by_month()
        
        # Generate plots
        self.generate_equity_curve()
        self.generate_distribution_plot()
        
        logger.info("\n✓ Backtest analysis complete\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Run complete analysis for last 2 years
    analyzer = StrategyAnalyzer()
    analyzer.full_backtest_analysis(
        start_date='2024-01-01',
        end_date='2026-01-15'
    )
    
    # Or analyze specific period
    # analyzer.full_backtest_analysis(
    #     start_date='2025-01-01',
    #     end_date='2025-12-31'
    # )
