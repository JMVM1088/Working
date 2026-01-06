import numpy as np

class EnsembleEngine:
    def __init__(self, threshold=0.4, risk_per_trade=0.01, stability_limit=0.5):
        self.threshold = threshold
        self.risk_per_trade = risk_per_trade
        self.stability_limit = stability_limit

    def generate_signals(self, df, last_run_score=None):
        # 1. Scoring Logic
        s1 = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
        s2 = np.where(df['MACD'] > df['MACD_Sig'], 1, -1)
        s3 = np.where(df['ROC'] > 0, 1, -1)
        s4 = np.where(df['Stoch'] < 20, 1, np.where(df['Stoch'] > 80, -1, 0))
        s5 = np.where(df['MA_Slope'] > 0, 1, -1)

        df['Ensemble_Score'] = (s1 + s2 + s3 + s4 + s5) / 5
        df['Regime'] = np.where(df['Close'] > df['SMA_200'], 1, -1)
        
        # 2. Filter by Regime
        df['Filtered_Score'] = np.where(np.sign(df['Ensemble_Score']) == df['Regime'], 
                                        df['Ensemble_Score'], 0)
        
        # 3. Stability Check
        df['Stability_Flag'] = "Stable"
        if last_run_score is not None:
            score_delta = abs(df['Filtered_Score'].iloc[-1] - last_run_score)
            if score_delta > self.stability_limit:
                df.iloc[-1, df.columns.get_loc('Filtered_Score')] = 0
                df.iloc[-1, df.columns.get_loc('Stability_Flag')] = "Unstable"

        # 4. Final Position Sizing
        df['Final_Score'] = np.where(abs(df['Filtered_Score']) >= self.threshold, 
                                     df['Filtered_Score'], 0)
        df['Vol_Adj_Weight'] = (df['Close'] * self.risk_per_trade) / (df['ATR'] + 1e-9)
        df['Position_Size'] = df['Final_Score'] * df['Vol_Adj_Weight'].clip(0, 2)
        
        return df