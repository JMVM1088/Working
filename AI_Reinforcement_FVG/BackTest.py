"""
🔄 FVG-DQN BACKTESTING SCRIPT
✅ Load trained FVG-DQN model
✅ Run backtest on out-of-sample data
✅ Generate performance metrics and trades
"""

import pandas as pd
import numpy as np
import sqlalchemy as sa
from stable_baselines3 import DQN
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Import from training script (ensure FVG_DQN_Integrated.py is in same directory)
from FVG_DQN_Integrated import (
    FVGConfig, FVG, detect_fvg_for_symbol, get_fvg_features,
    TECHNICAL_FEATURES, compute_technical_features, 
    FVG_DQN_Env, load_data, prepare_data
)


@dataclass
class Trade:
    """Trade record for backtest"""
    entry_date: datetime
    exit_date: Optional[datetime]
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    pnl: float
    return_pct: float
    exit_reason: str  # 'close', 'sl', 'tp', 'end'


class FVG_DQN_Backtest:
    """
    Backtest framework for FVG-DQN model.
    """

    def __init__(
        self,
        model_path: str,
        tech_mean_path: str,
        tech_std_path: str,
        fvg_config: Optional[FVGConfig] = None,
        transaction_cost: float = 0.0005
    ):
        print(f"📊 Loading model from {model_path}...")
        self.model = DQN.load(model_path)
        self.tech_mean = np.load(tech_mean_path)
        self.tech_std = np.load(tech_std_path)
        self.fvg_config = fvg_config or FVGConfig()
        self.transaction_cost = transaction_cost

        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.dates: List[datetime] = []

    def run_backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
        plot_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtest on provided data.
        """
        print(f"\n🔄 Running backtest on {len(df)} bars...")

        # Prepare data
        df = prepare_data(df)

        # Initialize
        equity = initial_capital
        position = 0.0
        current_trade: Optional[Trade] = None

        # Prepare arrays
        tech_features = df[TECHNICAL_FEATURES].fillna(0).values.astype(np.float32)
        tech_features = (tech_features - self.tech_mean) / self.tech_std
        prices = df['close'].values
        dates = df['date'].values

        start_idx = max(200, self.fvg_config.lookback_days)

        self.equity_curve = [equity] * start_idx
        self.dates = list(dates[:start_idx])

        for i in range(start_idx, len(df) - 1):
            # Build observation
            tech = tech_features[i]
            fvg_feats, active_fvgs = get_fvg_features(df, i, self.fvg_config)
            state = np.array([position, equity / initial_capital], dtype=np.float32)
            obs = np.concatenate([tech, fvg_feats, state])

            # Get action
            action, _ = self.model.predict(obs, deterministic=True)
            target_pos = [0.0, 1.0, -1.0][int(action)]

            # Execute
            current_price = prices[i]
            next_price = prices[i + 1]

            # Transaction costs
            if target_pos != position:
                pos_change = abs(target_pos - position)
                cost = pos_change * equity * self.transaction_cost
                equity -= cost

                # Record trade entry/exit
                if position == 0 and target_pos != 0:
                    # Opening trade
                    current_trade = Trade(
                        entry_date=dates[i],
                        exit_date=None,
                        direction='long' if target_pos > 0 else 'short',
                        entry_price=current_price,
                        exit_price=None,
                        pnl=0.0,
                        return_pct=0.0,
                        exit_reason='open'
                    )
                elif position != 0 and target_pos == 0:
                    # Closing trade
                    if current_trade:
                        if position > 0:  # Long
                            pnl = (current_price - current_trade.entry_price) / current_trade.entry_price * equity
                        else:  # Short
                            pnl = (current_trade.entry_price - current_price) / current_trade.entry_price * equity

                        current_trade.exit_date = dates[i]
                        current_trade.exit_price = current_price
                        current_trade.pnl = pnl
                        current_trade.return_pct = pnl / initial_capital * 100
                        current_trade.exit_reason = 'close'
                        self.trades.append(current_trade)
                        current_trade = None

            # Calculate PnL
            if position != 0:
                price_ret = (next_price - current_price) / current_price
                if position > 0:
                    equity += position * price_ret * equity
                else:
                    equity += position * price_ret * equity

            position = target_pos

            self.equity_curve.append(equity)
            self.dates.append(dates[i])

            if equity <= 0:
                print(f"💥 Ruin at bar {i}!")
                break

        # Close any open trade at end
        if current_trade and position != 0:
            final_price = prices[-1]
            if position > 0:
                pnl = (final_price - current_trade.entry_price) / current_trade.entry_price * equity
            else:
                pnl = (current_trade.entry_price - final_price) / current_trade.entry_price * equity

            current_trade.exit_date = dates[-1]
            current_trade.exit_price = final_price
            current_trade.pnl = pnl
            current_trade.return_pct = pnl / initial_capital * 100
            current_trade.exit_reason = 'end'
            self.trades.append(current_trade)

        # Calculate metrics
        metrics = self._calculate_metrics(initial_capital)

        # Plot
        if plot_results:
            self._plot_results()

        return metrics

    def _calculate_metrics(self, initial_capital: float) -> Dict[str, Any]:
        """Calculate performance metrics."""
        equity_arr = np.array(self.equity_curve)
        returns = np.diff(equity_arr) / equity_arr[:-1]

        total_return = (equity_arr[-1] - initial_capital) / initial_capital * 100

        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - cummax) / cummax
        max_dd = drawdown.min() * 100

        # Trade metrics
        if self.trades:
            trade_returns = [t.return_pct for t in self.trades]
            win_rate = len([t for t in self.trades if t.pnl > 0]) / len(self.trades) * 100
            avg_trade = np.mean(trade_returns)
            profit_factor = (
                sum([t.pnl for t in self.trades if t.pnl > 0]) / 
                abs(sum([t.pnl for t in self.trades if t.pnl < 0])) 
                if sum([t.pnl for t in self.trades if t.pnl < 0]) != 0 else float('inf')
            )
        else:
            win_rate = avg_trade = profit_factor = 0.0

        metrics = {
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'final_equity': equity_arr[-1],
            'num_trades': len(self.trades),
            'win_rate_pct': win_rate,
            'avg_trade_return_pct': avg_trade,
            'profit_factor': profit_factor
        }

        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS")
        print("="*60)
        print(f"Total Return:     {total_return:>10.2f}%")
        print(f"Sharpe Ratio:     {sharpe:>10.2f}")
        print(f"Max Drawdown:     {max_dd:>10.2f}%")
        print(f"Final Equity:     ${equity_arr[-1]:>10,.2f}")
        print(f"Number of Trades: {len(self.trades):>10}")
        print(f"Win Rate:         {win_rate:>10.2f}%")
        print(f"Avg Trade Return: {avg_trade:>10.2f}%")
        print(f"Profit Factor:    {profit_factor:>10.2f}")
        print("="*60)

        return metrics

    def _plot_results(self):
        """Plot equity curve and drawdown."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        ax1.plot(self.dates, self.equity_curve, label='Equity', color='blue')
        ax1.set_title('FVG-DQN Backtest Results')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        equity_arr = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - cummax) / cummax * 100
        ax2.fill_between(self.dates, drawdown, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('fvg_dqn_backtest.png', dpi=150)
        print("\n📈 Chart saved: fvg_dqn_backtest.png")
        plt.show()

    def get_trade_log(self) -> pd.DataFrame:
        """Get DataFrame of all trades."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'Entry Date': t.entry_date,
                'Exit Date': t.exit_date,
                'Direction': t.direction,
                'Entry Price': t.entry_price,
                'Exit Price': t.exit_price,
                'PnL ($)': t.pnl,
                'Return (%)': t.return_pct,
                'Exit Reason': t.exit_reason
            }
            for t in self.trades
        ])


def run_backtest_example():
    """Example backtest execution."""

    # Database connection
    SQL_CONN = (
        "mssql+pyodbc://localhost/Stock?"
        "driver=ODBC+Driver+17+for+SQL+Server&"
        "trusted_connection=yes"
    )

    # Load out-of-sample data (2019-2023)
    print("📊 Loading test data...")
    engine = sa.create_engine(SQL_CONN)
    query = """
    SELECT 
        symbol, 
        TradeDate as date, 
        [open], 
        high, 
        low, 
        [close], 
        volume
    FROM AI_ETF_Prices
    WHERE symbol = 'SPY' 
      AND TradeDate >= '2019-01-01' 
      AND TradeDate <= '2023-12-31'
    ORDER BY TradeDate
    """
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])

    # Initialize backtester
    backtester = FVG_DQN_Backtest(
        model_path="fvg_dqn_spy.zip",
        tech_mean_path="fvg_dqn_spy_tech_mean.npy",
        tech_std_path="fvg_dqn_spy_tech_std.npy",
        fvg_config=FVGConfig(threshold_per=0.5, max_active_fvgs=3)
    )

    # Run backtest
    metrics = backtester.run_backtest(df, initial_capital=100000.0)

    # Save trade log
    trade_log = backtester.get_trade_log()
    if not trade_log.empty:
        trade_log.to_csv('fvg_dqn_trades.csv', index=False)
        print("\n📄 Trade log saved: fvg_dqn_trades.csv")

    return metrics, backtester


if __name__ == "__main__":
    try:
        metrics, backtester = run_backtest_example()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()