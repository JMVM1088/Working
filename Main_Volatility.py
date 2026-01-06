import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
import Util

# =============================
# CONFIG
# =============================
START_DATE = "2020-01-01"
WINDOW_SIZE = 5
Z_WINDOW = 30
VOL_REGIME_LOOKBACK = 252

ENGINE_STR = (
    "mssql+pyodbc://@BEELINK/Stock"
    "?driver=ODBC Driver 17 for SQL Server&trusted_connection=yes"
)

# =============================
# FUNCTIONS
# =============================
def yang_zhang(df, window, periods=252):
    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"] / df["Open"])
    log_co = np.log(df["Close"] / df["Open"])
    log_oc = np.log(df["Open"] / df["Close"].shift(1))
    log_cc = np.log(df["Close"] / df["Close"].shift(1))

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc.pow(2).rolling(window).sum() / (window - 1)
    open_vol = log_oc.pow(2).rolling(window).sum() / (window - 1)
    window_rs = rs.rolling(window).sum() / (window - 1)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    return np.sqrt(open_vol + k * close_vol + (1 - k) * window_rs) * np.sqrt(periods)


def classify_alert(z):
    if abs(z) < 1: return "Normal"
    if abs(z) < 2: return "Elevated"
    if abs(z) < 3: return "High"
    return "Extreme"


def compute_vol_regime(df):
    df["VolRank"] = (
        df.groupby("Symbol")["Volatility"]
          .rolling(VOL_REGIME_LOOKBACK)
          .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
          .reset_index(level=0, drop=True)
    )
    df["VolRegime"] = pd.cut(
        df["VolRank"], [0, .33, .66, 1], labels=["Low", "Medium", "High"]
    )
    return df


def classify_pnl(row):
    if row.VolatilityChange > 0 and row.Return < 0: return "Risk-Off"
    if row.VolatilityChange > 0 and row.Return > 0: return "Short-Covering"
    if row.VolatilityChange < 0 and row.Return > 0: return "Risk-On"
    return "Neutral"


# =============================
# MAIN
# =============================
engine = create_engine(ENGINE_STR)

symbols = pd.read_sql(
    "Exec Stock..sp_GetSymbol_SPY", engine
)["Symbol"].tolist()

results = []

for symbol in symbols:
    #df = yf.download(symbol[0], start=START_DATE, progress=False)
    df = Util.get_data(engine, "AI_stock_Prices", "[Time]", symbol[0], "2020-01-01", "2025-12-31")
    vol = yang_zhang(df, WINDOW_SIZE)
    
    tmp = vol.reset_index()
    tmp.columns = ["ReportDate", "Volatility"]
    # tmp = pd.DataFrame({
    # "Date": vol.index,
    # "Volatility": vol.values
    # })
    tmp["Symbol"] = symbol

    tmp["VolatilityChange"] = tmp["Volatility"].diff()
    tmp["ZScore"] = (
        (tmp["VolatilityChange"] -
         tmp["VolatilityChange"].rolling(Z_WINDOW).mean()) /
        tmp["VolatilityChange"].rolling(Z_WINDOW).std()
    )

    tmp["Return"] = np.log(df["Close"] / df["Close"].shift(1)).values
    tmp["AlertLevel"] = tmp["ZScore"].apply(classify_alert)

    results.append(tmp)
    ##++++++++++++++
    valid_results = []

    for df in results:
        if df is None or df.empty:
            continue
        if df.dropna().empty:
            continue
        valid_results.append(df)

    if not valid_results:
        raise RuntimeError("No valid volatility data generated.")

    #vol_df = pd.concat(valid_results, ignore_index=True).dropna()
    ##vol_df = pd.concat(results).dropna()
    vol_df = pd.concat(valid_results, axis=0)
    vol_df["Date"] = pd.to_datetime(vol_df["Date"])
    vol_df = vol_df.dropna()
    vol_df = compute_vol_regime(vol_df)
    vol_df["PnLSignal"] = vol_df.apply(classify_pnl, axis=1)

    vol_df.to_sql(
        "AI_Stock_Volatility",
        engine,
        if_exists="replace",
        index=False
    )

print("✅ Historical run completed successfully.")
