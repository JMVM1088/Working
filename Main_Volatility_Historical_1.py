import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# =============================
# CONFIG
# =============================
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

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
    if abs(z) < 1:
        return "Normal"
    if abs(z) < 2:
        return "Elevated"
    if abs(z) < 3:
        return "High"
    return "Extreme"


def compute_vol_regime(df):
    df["VolRank"] = (
        df.groupby("Symbol")["Volatility"]
          .rolling(VOL_REGIME_LOOKBACK)
          .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
          .reset_index(level=0, drop=True)
    )

    df["VolRegime"] = pd.cut(
        df["VolRank"],
        [0, 0.33, 0.66, 1],
        labels=["Low", "Medium", "High"]
    )
    return df


def classify_pnl(row):
    if row.VolatilityChange > 0 and row.Return < 0:
        return "Risk-Off"
    if row.VolatilityChange > 0 and row.Return > 0:
        return "Short-Covering"
    if row.VolatilityChange < 0 and row.Return > 0:
        return "Risk-On"
    return "Neutral"


def get_data(engine, table, dateField, ticker, start, end):
    query = f"""
        SELECT
            {dateField} AS ReportDate,
            [Open], [High], [Low], [Close], [Volume]
        FROM {table}
        WHERE Symbol = '{ticker}'
          AND {dateField} BETWEEN '{start}' AND '{end}'
        ORDER BY {dateField}
    """
    df = pd.read_sql(query, engine)
    df["ReportDate"] = pd.to_datetime(df["ReportDate"])
    return df


# =============================
# MAIN
# =============================
engine = create_engine(ENGINE_STR)

symbols = pd.read_sql(
    "EXEC Stock..sp_GetSymbol_SPY",
    engine
)["Symbol"].tolist()

results = []

for symbol in symbols:
    df = get_data(
        engine,
        "AI_stock_Prices",
        "[Time]",
        symbol,               # ✅ FIXED (no symbol[0])
        START_DATE,
        END_DATE
    )

    if len(df) < WINDOW_SIZE + Z_WINDOW:
        continue

    vol = yang_zhang(df, WINDOW_SIZE)

    tmp = pd.DataFrame({
        "ReportDate": df["ReportDate"].values,
        "Symbol": symbol,
        "Volatility": vol.values
    })

    tmp["VolatilityChange"] = tmp["Volatility"].diff()

    tmp["ZScore"] = (
        (tmp["VolatilityChange"] -
         tmp["VolatilityChange"].rolling(Z_WINDOW).mean()) /
        tmp["VolatilityChange"].rolling(Z_WINDOW).std()
    )

    tmp["Return"] = np.log(df["Close"] / df["Close"].shift(1)).values
    tmp["AlertLevel"] = tmp["ZScore"].apply(classify_alert)

    # 🔑 DO NOT nuke everything
    tmp = tmp.dropna(subset=["Volatility", "VolatilityChange", "ZScore"])

    if not tmp.empty:
        results.append(tmp)

# =============================
# FINAL CONCAT
# =============================
if not results:
    raise RuntimeError("❌ No historical volatility data generated")

vol_df = pd.concat(results, ignore_index=True)

vol_df = compute_vol_regime(vol_df)
vol_df["PnLSignal"] = vol_df.apply(classify_pnl, axis=1)

# =============================
# PREVENT DUPLICATES
# =============================
existing = pd.read_sql(
    "SELECT ReportDate, Symbol FROM AI_Stock_Volatility",
    engine
)

existing["ReportDate"] = pd.to_datetime(existing["ReportDate"])

vol_df = (
    vol_df.merge(
        existing,
        on=["ReportDate", "Symbol"],
        how="left",
        indicator=True
    )
    .query("_merge == 'left_only'")
    .drop(columns="_merge")
)
# =============================
# WRITE TO SQL
# =============================
if not vol_df.empty:
    vol_df.to_sql(
        "AI_Stock_Volatility",
        engine,
        if_exists="append",
        index=False
    )

print("✅ Historical run completed successfully.")
