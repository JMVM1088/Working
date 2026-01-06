import pandas as pd
import numpy as np

class Indicators:
    @staticmethod
    def add_all(df):
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

        # MACD
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Sig'] = df['MACD'].ewm(span=9).mean()

        # ROC
        df['ROC'] = df['Close'].pct_change(10)

        # Stochastics
        low_14, high_14 = df['Low'].rolling(14).min(), df['High'].rolling(14).max()
        df['Stoch'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-9))

        # Slope
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['MA_Slope'] = df['SMA_20'].diff(3) / 3

        # ATR & Regime
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        return df.dropna()