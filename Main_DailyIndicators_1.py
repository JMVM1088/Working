import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine

# --- CONFIGURATION ---
DB_URL = "mssql+pyodbc://@BEELINK/Stock?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
engine = create_engine(DB_URL)

def fetch_market_data():
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
    df = df.sort_values(['Symbol', 'time'])
    group = df.groupby('Symbol')
    
    # --- Standard Indicators ---
    for p in [5, 20, 50, 200]:
        df[f'SMA_{p}'] = group['close'].transform(lambda x: x.rolling(p).mean())

    # Bollinger & Keltner (for Squeeze)
    df['SMA_20'] = group['close'].transform(lambda x: x.rolling(20).mean())
    std_20 = group['close'].transform(lambda x: x.rolling(20).std())
    df['BB_Upper'] = df['SMA_20'] + (std_20 * 2)
    df['BB_Lower'] = df['SMA_20'] - (std_20 * 2)

    # ATR Calculation (Vectorized)
    prev_close = group['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-prev_close).abs(), (df['low']-prev_close).abs()], axis=1).max(axis=1)
    df['ATR_14'] = group.apply(lambda x: tr.loc[x.index].rolling(14).mean()).reset_index(level=0, drop=True)

    # RSI
    def get_rsi(s, p=14):
        delta = s.diff()
        gain = (delta.where(delta > 0, 0)).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
        return 100 - (100 / (1 + (gain / loss)))
    df['RSI_14'] = group['close'].transform(get_rsi)

    # HV & Percentile
    df['HV_21'] = group['close'].transform(lambda x: np.log(x/x.shift(1)).rolling(21).std() * np.sqrt(252))
    h_min = group['HV_21'].transform(lambda x: x.rolling(252).min())
    h_max = group['HV_21'].transform(lambda x: x.rolling(252).max())
    df['HV_Percentile'] = (df['HV_21'] - h_min) / (h_max - h_min)

    # --- Spike & Anomaly Detection (10-day lookback) ---
    # 1. Volume Spike (RVOL)
    avg_vol_10 = group['Volume'].transform(lambda x: x.rolling(10).mean())
    df['RVOL'] = df['Volume'] / avg_vol_10

    # 2. Volatility (ATR) Spike
    avg_atr_10 = group['ATR_14'].transform(lambda x: x.rolling(10).mean())
    df['ATR_Spike'] = df['ATR_14'] / avg_atr_10

    # 3. RSI Velocity (3-day momentum change)
    df['RSI_3d_Change'] = group['RSI_14'].transform(lambda x: x.diff(3))

    return df

def generate_signals(df):
    def get_flags(row):
        f = []
        # Traditional Trends
        if row['SMA_50'] > row['SMA_200']: f.append("BULL_TREND")
        
        # Options Squeeze
        # (Assuming KC calculation is here as per previous script)
        
        # Unusual Spikes (The "New" Signals)
        if row['RVOL'] > 2.0: f.append("VOL_SPIKE")  # 2x normal volume
        if row['ATR_Spike'] > 1.3: f.append("VOLATILITY_EXPANSION") # ATR increased 30% vs last 10 days
        if abs(row['RSI_3d_Change']) > 20: f.append("MOMENTUM_IMPULSE")
        
        if row['HV_Percentile'] < 0.1 and row['RVOL'] > 1.5:
            f.append("CHEAP_VOL_ACCUMULATION") # Low IV but volume is picking up

        return ";".join(f)

    df['Signal_Flags'] = df.apply(get_flags, axis=1)
    return df

def main():
    print("Extracting data...")
    df = fetch_market_data()
    
    print("Calculating Indicators and Spikes...")
    df = compute_indicators(df)
    df = generate_signals(df)
    
    # Get latest snapshot
    latest = df[df['time'] == df['time'].max()].copy()
    
    # Save to SQL
    # Ensure your table [dbo].[AI_stock_indicators_v3] has columns for RVOL, ATR_Spike, etc.
    latest.to_sql("AI_stock_indicators_v1", engine, if_exists='append', index=False)
    print(f"Update complete for {len(latest)} symbols.")

if __name__ == "__main__":
    main()