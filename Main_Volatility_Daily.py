import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
import Util

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
    query = f"SELECT {dateField} as ReportDate, [Open], [High], [Low], [Close], [Volume] FROM {table} WHERE Symbol = '{ticker}' AND {dateField} <= '{end}' ORDER BY ReportDate ASC"
    df = pd.read_sql(query, engine)
    df['ReportDate'] = pd.to_datetime(df['ReportDate'])
    return df
# =============================
# MAIN
# =============================
engine = create_engine(ENGINE_STR)

symbols = pd.read_sql(
    "SELECT Symbol FROM AI_Stock_Info i (nolock) where i.[Index] like '%S&P 500%' and i.symbol = 'NVDA' order by i.Symbol", engine
)["Symbol"].tolist()

existing = pd.read_sql(
    "SELECT ReportDate, Symbol FROM [AI_Stock_Volatility]", engine
)

rows = []

for symbol in symbols:
    #df = yf.download(symbol, period="1y", progress=False)
    df = get_data(engine, "AI_stock_Prices", "[Time]", symbol, "2025-01-01", "2025-12-31")
    vol = yang_zhang(df, WINDOW_SIZE)

    # tmp = vol.reset_index()
    # tmp.columns = ["ReportDate", "Volatility"]
    # tmp["Symbol"] = symbol
    # tmp["VolatilityChange"] = tmp["Volatility"].diff()

    # tmp["ZScore"] = (
    #     (tmp["VolatilityChange"] -
    #      tmp["VolatilityChange"].rolling(Z_WINDOW).mean()) /
    #     tmp["VolatilityChange"].rolling(Z_WINDOW).std()
    # )

    # rows.append(tmp.tail(1))
    tmp = pd.DataFrame({
        "ReportDate": df["ReportDate"].values,
        "Volatility": vol.values
    })

    tmp["Symbol"] = symbol
    tmp["VolatilityChange"] = tmp["Volatility"].diff()

    tmp["ZScore"] = (
        (tmp["VolatilityChange"] -
         tmp["VolatilityChange"].rolling(Z_WINDOW).mean()) /
        tmp["VolatilityChange"].rolling(Z_WINDOW).std()
    )

    rows.append(tmp.loc[tmp["ReportDate"] == tmp["ReportDate"].max()])


daily_df = pd.concat(rows).dropna()
##daily_df["ReportDate"] = pd.to_datetime(daily_df["ReportDate"]).dt.date
daily_df = daily_df.merge(
    existing, on=["ReportDate", "Symbol"],
    how="left", indicator=True
).query("_merge == 'left_only'").drop(columns="_merge")

if not daily_df.empty:
    daily_df.to_sql(
        "AI_Stock_Volatility",
        engine,
        if_exists="append",
        index=False
    )

with engine.begin() as conn:
    conn.execute(text("EXEC sp_GenerateDailyVolatilityAlerts"))
    #conn.execute(text("EXEC sp_BuildMarketShockIndex"))

print("✅ Daily run completed successfully.")
