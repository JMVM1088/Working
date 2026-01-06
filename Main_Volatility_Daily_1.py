import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
# =============================
# CONFIG
# =============================
WINDOW_SIZE = 5
Z_WINDOW = 30

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

existing = pd.read_sql(
    "SELECT ReportDate, Symbol FROM AI_Stock_Volatility",
    engine
)
existing["ReportDate"] = pd.to_datetime(existing["ReportDate"])

rows = []
END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
for symbol in symbols:
    df = get_data(
        engine,
        "AI_stock_Prices",
        "[Time]",
        symbol,
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

    # keep only the most recent trading day
    latest = tmp.loc[tmp["ReportDate"] == tmp["ReportDate"].max()]

    latest = latest.dropna(
        subset=["Volatility", "VolatilityChange", "ZScore"]
    )

    if not latest.empty:
        rows.append(latest)

# =============================
# FINAL DAILY DF
# =============================
if not rows:
    print("⚠️ No daily volatility rows generated")
    exit(0)

daily_df = pd.concat(rows, ignore_index=True)

# enforce datetime (defensive)
daily_df["ReportDate"] = pd.to_datetime(daily_df["ReportDate"])

# prevent duplicates
daily_df = (
    daily_df.merge(
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
if not daily_df.empty:
    daily_df.to_sql(
        "AI_Stock_Volatility",
        engine,
        if_exists="append",
        index=False
    )

with engine.begin() as conn:
    conn.execute(text("EXEC sp_GenerateDailyVolatilityAlerts"))
    conn.execute(text("EXEC sp_BuildMarketShockIndex"))

print("✅ Daily run completed successfully.")
