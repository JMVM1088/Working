import numpy as np

class Backtester:
    def __init__(self, initial_capital=100000, fee=0.0005):
        self.initial_capital = initial_capital
        self.fee = fee

    def run(self, df):
        df = df.copy()
        df['Market_Ret'] = df['Close'].pct_change()
        df['Strategy_Ret'] = df['Position_Size'].shift(1) * df['Market_Ret']
        
        # Subtract fees on rebalancing
        trades = df['Position_Size'].diff().abs()
        df['Strategy_Ret'] -= trades * self.fee
        
        df['Equity'] = self.initial_capital * (1 + df['Strategy_Ret']).cumprod()
        
        sharpe = (df['Strategy_Ret'].mean() / (df['Strategy_Ret'].std() + 1e-9)) * np.sqrt(252)
        dd = (df['Equity'] - df['Equity'].cummax()) / df['Equity'].cummax()
        
        return df, {"Sharpe": sharpe, "MaxDD": dd.min()}