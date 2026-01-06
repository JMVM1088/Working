import numpy as np

class Backtester:
    def __init__(self, initial_capital=100000, fee=0.0005):
        self.initial_capital = initial_capital
        self.fee = fee # 0.05% per trade

    def run(self, df):
        df = df.copy()
        df['Market_Ret'] = df['Close'].pct_change()
        
        # Execution happens on next bar's return
        df['Strategy_Ret'] = df['Position_Size'].shift(1) * df['Market_Ret']
        
        # Apply Fees on position changes
        trades = df['Position_Size'].diff().abs()
        df['Strategy_Ret'] -= trades * self.fee
        
        # Performance Metrics
        df['Equity'] = self.initial_capital * (1 + df['Strategy_Ret']).cumprod()
        
        days = len(df)
        ann_ret = (df['Equity'].iloc[-1] / self.initial_capital) ** (252/days) - 1
        ann_std = df['Strategy_Ret'].std() * np.sqrt(252)
        sharpe = ann_ret / ann_std if ann_std != 0 else 0
        
        dd = (df['Equity'] - df['Equity'].cummax()) / df['Equity'].cummax()
        
        return df, {"Sharpe": sharpe, "MaxDD": dd.min(), "Return": ann_ret}