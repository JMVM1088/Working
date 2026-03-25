"""
🔄 FVG-PPO Advanced Backtest
Enhanced backtesting with detailed analytics
"""

import pandas as pd
import numpy as np
import sqlalchemy as sa
from stable_baselines3 import PPO
from datetime import datetime
from typing import List, Dict, Any, Optional, Deque
from collections import deque
import matplotlib.pyplot as plt
from dataclasses import dataclass

from FVG_PPO_Advanced import (
    FVGConfig, FVG, detect_fvg_for_symbol, 
    TECHNICAL_FEATURES, compute_technical_features,
    FVG_PPO_Env, MarketRegime, detect_market_regime
)


@dataclass
class Trade:
    entry_date: datetime
    exit_date: datetime
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    exit_reason: str
    sharpe_during: float
    max_dd_during: float


class FVG_PPO_Backtest:
    """Professional backtest with risk metrics"""

    def __init__(
        self,
        model_path: str,
        tech_mean_path: str,
        tech_std_path: str,
        fvg_config: Optional[FVGConfig] = None,
        initial_capital: float = 100000.0
    ):
        print(f"📊 Loading PPO model from {model_path}...")
        self.model = PPO.load(model_path)
        self.tech_mean = np.load(tech_mean_path)
        self.tech_std = np.load(tech_std_path)
        self.fvg_config = fvg_config or FVGConfig()
        self.initial_capital = initial_capital

        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.dates: List[datetime] = []
        self.positions: List[float] = []
        self.returns: List[float] = []

    def run_backtest(
        self,
        df: pd.DataFrame,
        plot_results: bool = True,
        save_trades: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive backtest"""

        print(f"\n🔄 Running backtest on {len(df)} bars...")
        df = compute_technical_features(df)
        df = df.dropna().reset_index(drop=True)

        # Prepare arrays
        tech_features = df[TECHNICAL_FEATURES].fillna(0).values.astype(np.float32)
        tech_features = (tech_features - self.tech_mean) / self.tech_std
        prices = df['close'].values
        dates = df['date'].values

        # Initialize
        equity = self.initial_capital
        position = 0.0
        peak_equity = self.initial_capital
        current_trade: Optional[Trade] = None
        returns_window: Deque[float] = deque(maxlen=20)

        start_idx = 200
        self.equity_curve = [equity] * start_idx
        self.dates = list(dates[:start_idx])
        self.positions = [0.0] * start_idx

        for i in range(start_idx, len(df) - 1):
            # Build observation
            tech = tech_features[i]

            # FVG features
            lookback_start = max(0, i - 90)
            df_window = df.iloc[lookback_start:i+1].copy()
            fvgs = detect_fvg_for_symbol(df_window, threshold_per=0.3)

            current_date = dates[i]
            current_price = prices[i]

            # Filter active FVGs
            active_fvgs = []
            for fvg in fvgs:
                days_since = (current_date - fvg.t_time).days
                if days_since <= 15:
                    active_fvgs.append(fvg)
            active_fvgs = active_fvgs[:5]

            # FVG features
            fvg_feats = []
            for j in range(5):
                if j < len(active_fvgs):
                    fvg = active_fvgs[j]
                    gap = fvg.gap_height() / current_price
                    dist = (current_price - fvg.mid_price()) / current_price
                    age = days_since / 15
                    direction = 1.0 if fvg.isbull else -1.0
                    fvg_feats.extend([gap, dist, age, direction, 0.5])
                else:
                    fvg_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # Regime features
            regime = detect_market_regime(df.iloc[:i+1])
            regime_feats = [
                regime.trend_strength,
                regime.adx / 100.0,
                1.0 if regime.volatility_regime == "high" else 0.0,
                1.0 if regime.volatility_regime == "low" else 0.0,
                position
            ]

            obs = np.concatenate([tech, np.array(fvg_feats), np.array(regime_feats)])

            # Get action from PPO
            action, _ = self.model.predict(obs, deterministic=True)
            target_pos = np.clip(action[0], -1.0, 1.0)

            # Execute
            prev_price = prices[i-1] if i > 0 else prices[i]
            price_ret = (current_price - prev_price) / prev_price

            # PnL
            trade_pnl = position * price_ret * equity
            equity += trade_pnl

            # Track returns for metrics
            if equity > 0:
                daily_ret = trade_pnl / equity
                returns_window.append(daily_ret)

            # Trade management
            if abs(target_pos - position) > 0.1:  # Significant change
                if current_trade is not None:
                    # Close existing trade
                    current_trade.exit_date = current_date
                    current_trade.exit_price = current_price
                    current_trade.pnl = (current_price - current_trade.entry_price) / current_trade.entry_price
                    current_trade.pnl *= current_trade.size * self.initial_capital
                    if current_trade.direction == "short":
                        current_trade.pnl *= -1
                    current_trade.return_pct = current_trade.pnl / self.initial_capital * 100

                    if len(returns_window) > 0:
                        current_trade.sharpe_during = np.mean(returns_window) / (np.std(returns_window) + 1e-8) * np.sqrt(252)

                    self.trades.append(current_trade)
                    current_trade = None

                if abs(target_pos) > 0.1:  # Open new trade
                    direction = "long" if target_pos > 0 else "short"
                    current_trade = Trade(
                        entry_date=current_date,
                        exit_date=current_date,
                        direction=direction,
                        entry_price=current_price,
                        exit_price=current_price,
                        size=abs(target_pos),
                        pnl=0.0,
                        return_pct=0.0,
                        exit_reason="open",
                        sharpe_during=0.0,
                        max_dd_during=0.0
                    )

            position = target_pos
            peak_equity = max(peak_equity, equity)

            self.equity_curve.append(equity)
            self.dates.append(current_date)
            self.positions.append(position)

            if equity <= 0:
                break

        # Close final trade
        if current_trade is not None:
            current_trade.exit_date = dates[-1]
            current_trade.exit_price = prices[-1]
            current_trade.pnl = (prices[-1] - current_trade.entry_price) / current_trade.entry_price
            current_trade.pnl *= current_trade.size * self.initial_capital
            if current_trade.direction == "short":
                current_trade.pnl *= -1
            current_trade.return_pct = current_trade.pnl / self.initial_capital * 100
            current_trade.exit_reason = "end"
            self.trades.append(current_trade)

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics()

        if plot_results:
            self._plot_results()

        if save_trades:
            self._save_trade_log()

        return metrics

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate professional trading metrics"""
        equity_arr = np.array(self.equity_curve)
        returns = np.diff(equity_arr) / equity_arr[:-1]

        # Basic metrics
        total_return = (equity_arr[-1] - self.initial_capital) / self.initial_capital * 100

        # Risk metrics
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252) * (returns.mean() - 0.02/252) / returns.std()
            downside = returns[returns < 0]
            sortino = np.sqrt(252) * returns.mean() / downside.std() if len(downside) > 0 else 0
        else:
            sharpe = sortino = 0

        # Drawdown
        cummax = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - cummax) / cummax
        max_dd = drawdown.min() * 100
        calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else 0

        # Trade metrics
        if self.trades:
            trade_returns = [t.return_pct for t in self.trades]
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]

            win_rate = len(wins) / len(self.trades) * 100
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0
            avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
            profit_factor = abs(sum([t.pnl for t in wins]) / sum([t.pnl for t in losses])) if losses else float('inf')

            # Expectancy
            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * abs(avg_loss))
        else:
            win_rate = avg_win = avg_loss = profit_factor = expectancy = 0

        # Exposure
        exposure = np.mean(np.abs(self.positions)) * 100

        metrics = {
            'total_return_pct': total_return,
            'annualized_return_pct': total_return / (len(equity_arr) / 252) * 100 if len(equity_arr) > 0 else 0,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown_pct': max_dd,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'num_trades': len(self.trades),
            'avg_trade_return_pct': np.mean(trade_returns) if self.trades else 0,
            'exposure_pct': exposure,
            'final_equity': equity_arr[-1]
        }

        # Print report
        print("\n" + "="*70)
        print("📊 FVG-PPO ADVANCED BACKTEST RESULTS")
        print("="*70)
        print(f"Total Return:        {total_return:>10.2f}%")
        print(f"Annualized Return:   {metrics['annualized_return_pct']:>10.2f}%")
        print(f"Sharpe Ratio:        {sharpe:>10.2f}")
        print(f"Sortino Ratio:       {sortino:>10.2f}")
        print(f"Calmar Ratio:        {calmar:>10.2f}")
        print(f"Max Drawdown:        {max_dd:>10.2f}%")
        print(f"-"*70)
        print(f"Number of Trades:    {len(self.trades):>10}")
        print(f"Win Rate:            {win_rate:>10.2f}%")
        print(f"Profit Factor:       {profit_factor:>10.2f}")
        print(f"Expectancy:          ${expectancy:>9.2f}")
        print(f"Avg Trade Return:    {metrics['avg_trade_return_pct']:>10.2f}%")
        print(f"Exposure:            {exposure:>10.2f}%")
        print(f"="*70)

        return metrics

    def _plot_results(self):
        """Generate professional charts"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        equity_arr = np.array(self.equity_curve)
        dates = self.dates

        # Equity curve
        ax1 = axes[0]
        ax1.plot(dates, equity_arr, label='Equity', color='#2E86AB', linewidth=1.5)
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('FVG-PPO Advanced: Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        cummax = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - cummax) / cummax * 100
        ax2.fill_between(dates, drawdown, 0, color='#E94F37', alpha=0.3)
        ax2.plot(dates, drawdown, color='#E94F37', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # Position sizing
        ax3 = axes[2]
        colors = ['green' if p > 0 else 'red' if p < 0 else 'gray' for p in self.positions]
        ax3.bar(dates, self.positions, color=colors, alpha=0.6, width=1)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Position Sizing', fontsize=12)
        ax3.set_ylabel('Position (-1 to 1)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('fvg_ppo_advanced_backtest.png', dpi=150, bbox_inches='tight')
        print("\n📈 Chart saved: fvg_ppo_advanced_backtest.png")
        plt.show()

    def _save_trade_log(self):
        """Save detailed trade log"""
        if not self.trades:
            return

        df = pd.DataFrame([
            {
                'Entry Date': t.entry_date,
                'Exit Date': t.exit_date,
                'Direction': t.direction,
                'Size': t.size,
                'Entry Price': t.entry_price,
                'Exit Price': t.exit_price,
                'PnL ($)': t.pnl,
                'Return (%)': t.return_pct,
                'Exit Reason': t.exit_reason
            }
            for t in self.trades
        ])

        df.to_csv('fvg_ppo_trades.csv', index=False)
        print("📄 Trade log saved: fvg_ppo_trades.csv")


if __name__ == "__main__":
    SQL_CONN = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

    # Load test data
    print("📊 Loading test data (2019-2023)...")
    engine = sa.create_engine(SQL_CONN)
    query = """
    SELECT symbol, TradeDate as date, [open], high, low, [close], volume
    FROM AI_ETF_Prices
    WHERE symbol = 'SPY' AND TradeDate >= '2019-01-01' AND TradeDate <= '2023-12-31'
    ORDER BY TradeDate
    """
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])

    # Run backtest
    backtester = FVG_PPO_Backtest(
        model_path="fvg_ppo_v2_spy.zip",
        tech_mean_path="fvg_ppo_v2_spy_tech_mean.npy",
        tech_std_path="fvg_ppo_v2_spy_tech_std.npy",
        fvg_config=FVGConfig(threshold_per=0.3, max_active_fvgs=5)
    )

    metrics = backtester.run_backtest(df)