import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---   
DB_URL = "mssql+pyodbc://@BEELINK/Stock?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
engine = create_engine(DB_URL)

def fetch_market_data():
    """Fetch 300 days of history to satisfy all look-back periods."""
    query = """
    SELECT Symbol, [time], [open], [high], [low], [close], [Volume]
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY Symbol ORDER BY [time] DESC) as rn
        FROM [dbo].[AI_stock_prices]
    ) t
    WHERE rn <= 300 
    ORDER BY Symbol, [time] ASC
    """
    return pd.read_sql(query, engine)

def compute_indicators(df):
    # Ensure data is sorted for time-series integrity
    df = df.sort_values(['Symbol', 'time'])
    group = df.groupby('Symbol')
    
    # 1. SMAs (Short, Mid, Long)
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{p}'] = group['close'].transform(lambda x: x.rolling(p).mean())

    # 2. Bollinger Bands (10 and 20 periods)
    for p in [10, 20]:
        sma = group['close'].transform(lambda x: x.rolling(p).mean())
        std = group['close'].transform(lambda x: x.rolling(p).std())
        df[f'BB_Upper_{p}'] = sma + (std * 2)
        df[f'BB_Lower_{p}'] = sma - (std * 2)

    # 3. RSI Logic (Standard Vectorized)
    def get_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    for p in [5, 7, 9, 14, 21, 25]:
        df[f'RSI_{p}'] = group['close'].transform(lambda x: get_rsi(x, p))

    # 4. ATR Refactored (Fixes FutureWarning)
    # Calculate True Range (TR) across the whole dataframe safely
    # We use groupby shift to ensure we don't leak data from the previous Symbol
    prev_close = group['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    df['TR_temp'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    for p in [5, 14, 22]:
        df[f'ATR_{p}'] = df.groupby('Symbol')['TR_temp'].transform(lambda x: x.rolling(p).mean())
    
    df.drop(columns=['TR_temp'], inplace=True)

    # 5. MACD (12, 26, 9)
    ema12 = group['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = group['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df.groupby('Symbol')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

    # 6. HV & Percentile (Fixes KeyError: -1)
    df['log_ret'] = group['close'].transform(lambda x: np.log(x / x.shift(1)))
    df['HV_21'] = group['log_ret'].transform(lambda x: x.rolling(21).std() * np.sqrt(252))
    
    # Vectorized Percentile: (Current - Min) / (Max - Min)
    # This avoids the slow .apply() and the KeyError entirely
    rolling_min = group['HV_21'].transform(lambda x: x.rolling(252).min())
    rolling_max = group['HV_21'].transform(lambda x: x.rolling(252).max())
    
    df['HV_Percentile'] = (df['HV_21'] - rolling_min) / (rolling_max - rolling_min)
    # Fill cases where max == min with 0.5 to avoid division by zero
    df['HV_Percentile'] = df['HV_Percentile'].fillna(0.5)

    # 7. TTM Squeeze & Z-Score
    df['EMA_20'] = group['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['KC_Upper'] = df['EMA_20'] + (df['ATR_14'] * 1.5)
    df['KC_Lower'] = df['EMA_20'] - (df['ATR_14'] * 1.5)
    df['Squeeze_On'] = ((df['BB_Upper_20'] < df['KC_Upper']) & (df['BB_Lower_20'] > df['KC_Lower'])).astype(int)
    df['ZScore_20'] = (df['close'] - df['SMA_20']) / group['close'].transform(lambda x: x.rolling(20).std())

    return df

    # 4. ATR (Volatility-adjusted Range)
    def get_atr(sdf, period):
        tr = pd.concat([sdf['high']-sdf['low'], 
                        (sdf['high']-sdf['close'].shift()).abs(), 
                        (sdf['low']-sdf['close'].shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    for p in [5, 14, 22]:
        df[f'ATR_{p}'] = group.apply(lambda x: get_atr(x, p)).reset_index(level=0, drop=True)

    # 5. MACD (12, 26, 9)
    ema12 = group['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = group['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = group['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

    # 6. Historical Volatility (HV) & Percentile
    # Annualized 21-day Realized Volatility
    df['log_ret'] = group['close'].transform(lambda x: np.log(x / x.shift(1)))
    df['HV_21'] = group['log_ret'].transform(lambda x: x.rolling(21).std() * np.sqrt(252))
    
    # HV Percentile (Where current HV ranks over the last 252 days)
    df['HV_Percentile'] = group['HV_21'].transform(
        lambda x: x.rolling(252).apply(lambda y: (y[-1] - y.min()) / (y.max() - y.min()) if y.max() > y.min() else 0.5)
    )

    # 7. TTM Squeeze (Bollinger inside Keltner)
    # Keltner Channel (20 EMA +/- 1.5 ATR)
    df['EMA_20'] = group['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['KC_Upper'] = df['EMA_20'] + (df['ATR_14'] * 1.5)
    df['KC_Lower'] = df['EMA_20'] - (df['ATR_14'] * 1.5)
    df['Squeeze_On'] = ((df['BB_Upper_20'] < df['KC_Upper']) & (df['BB_Lower_20'] > df['KC_Lower'])).astype(int)

    # 8. Mean Reversion (Z-Score of price vs SMA 20)
    df['ZScore_20'] = (df['close'] - df['SMA_20']) / group['close'].transform(lambda x: x.rolling(20).std())

    return df

def generate_signals(df):
    """Business Logic for Trading Desk Flags"""
    def create_flag_string(row):
        f = []
        # Trend
        if row['SMA_50'] > row['SMA_200']: f.append("GOLDEN_CROSS")
        if row['SMA_50'] < row['SMA_200']: f.append("DEATH_CROSS")
        
        # Momentum
        if row['MACD'] > row['MACD_Signal']: f.append("MACD_BULL")
        if row['RSI_14'] > 70: f.append("RSI_OVERBOUGHT")
        if row['RSI_14'] < 30: f.append("RSI_OVERSOLD")
        
        # Options Specific
        if row['Squeeze_On'] == 1: f.append("VOL_SQUEEZE")
        if row['HV_Percentile'] > 0.8: f.append("HV_EXPENSIVE")
        if row['HV_Percentile'] < 0.2: f.append("HV_CHEAP")
        if abs(row['ZScore_20']) > 2.5: f.append("MEAN_REVERSION_STRETCH")
        
        return ";".join(f)

    df['Signal_Flags'] = df.apply(create_flag_string, axis=1)
    return df

def main():
    print(f"[{datetime.datetime.now()}] Job Started.")
    
    # Pipeline
    raw_df = fetch_market_data()
    if raw_df.empty: return
    
    processed_df = compute_indicators(raw_df)
    final_df = generate_signals(processed_df)
    
    # Extract only the latest date to append to the permanent table
    latest_date = final_df['time'].max()

    #---------------------------------
    # daily_batch = final_df[final_df['time'] == latest_date].copy()
    
    # # Cleaning up intermediate columns before SQL load
    # cols_to_save = ['Symbol', 'time', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
    #                 'BB_Upper_10', 'BB_Lower_10', 'BB_Upper_20', 'BB_Lower_20',
    #                 'RSI_14', 'ATR_14', 'MACD', 'HV_21', 'HV_Percentile', 'Squeeze_On', 'ZScore_20', 'Signal_Flags']
    
    # daily_batch[cols_to_save].to_sql("AI_stock_indicators", engine, if_exists='append', index=False)
    #------------------------
    # --- Data Sanitization Block ---
    # 1. Replace Infinity with NaN
    daily_batch = final_df[final_df['time'] == latest_date].copy()
    daily_batch = daily_batch.replace([np.inf, -np.inf], np.nan)
    cols_to_save = ['Symbol', 'time', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
                     'BB_Upper_10', 'BB_Lower_10', 'BB_Upper_20', 'BB_Lower_20',
                     'RSI_14', 'ATR_14', 'MACD', 'HV_21', 'HV_Percentile', 'Squeeze_On', 'ZScore_20', 'Signal_Flags']
    # 2. Fill NaN with None (SQLAlchemy treats None as SQL NULL)
    # We apply this specifically to the numeric columns
    numeric_cols = daily_batch.select_dtypes(include=[np.number]).columns
    daily_batch[numeric_cols] = daily_batch[numeric_cols].where(pd.notnull(daily_batch[numeric_cols]), None)

    # 3. Explicitly Clip values for FLOAT precision if necessary
    # (Optional: prevents values like 1e308 from crashing the insert)
    for col in numeric_cols:
        if daily_batch[col].dtype == object: continue # Skip if already converted to None
        daily_batch[col] = np.clip(daily_batch[col], -1.79e308, 1.79e308)

    # Now proceed to upload
    daily_batch[cols_to_save].to_sql("AI_stock_indicators", engine, if_exists='append', index=False, chunksize=50)



    print(f"[{datetime.datetime.now()}] Job Finished. Processed {len(daily_batch)} symbols.")

if __name__ == "__main__":
    main()