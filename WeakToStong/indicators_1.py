import pandas as pd
import numpy as np

class Indicators:
    @staticmethod
    def add_all(df):
        """
        Calculates the five weak learners, the ATR for risk sizing, 
        and the SMA 200 for the market regime filter.
        """
        # --- 1. RSI (Relative Strength Index) ---
        # Standard 14-period RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # --- 2. MACD (Moving Average Convergence Divergence) ---
        # 12 and 26 period EMAs with a 9-period signal line
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Sig'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # --- 3. ROC (Rate of Change) ---
        # 10-period percentage change
        df['ROC'] = df['Close'].pct_change(periods=10)

        # --- 4. Stochastics ---
        # 14-period %K line
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))

        # --- 5. Moving Average Slope ---
        # Slope of a 20-period SMA over the last 3 bars
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_Slope'] = df['SMA_20'].diff(3) / 3

        # --- 6. ATR (Average True Range) ---
        # Used for Volatility-Adjusted Position Sizing
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        df['TR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()

        # --- 7. Market Regime Filter ---
        # 200-Day Simple Moving Average
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Drop the initial rows that contain NaN values due to the 200-period lookback
        return df.dropna()