import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------
# Parameters
# -----------------------------
ticker = "SPY"
start_capital = 10000.0
ma_window = 200
buy_threshold = 0.04   # 4% above MA
sell_threshold = -0.03 # 3% below MA
position_fraction = 0.20  # 20% of equity
start_date = (datetime.today() - timedelta(days=365*21)).strftime("%Y-%m-%d")
end_date = datetime.today().strftime("%Y-%m-%d")

# -----------------------------
# Download QQQ data (20y+)
# -----------------------------
data = yf.download(ticker, start=start_date, end=end_date)

# If yfinance returns MultiIndex columns, flatten them
if isinstance(data.columns, pd.MultiIndex):
    # For single ticker, take the second level (the field name)
    data.columns = data.columns.get_level_values(0)

# Now keep only Close
data = data[["Close"]].copy()
data.dropna(inplace=True)

data["MA200"] = data["Close"].rolling(ma_window).mean()
data["ma_diff_pct"] = (data["Close"] - data["MA200"]) / data["MA200"]
data["buy_signal"] = data["ma_diff_pct"] > buy_threshold
data["sell_signal"] = data["ma_diff_pct"] < sell_threshold

# -----------------------------
# Backtest loop (vectorizable but kept explicit for clarity)
# -----------------------------
equity = start_capital
cash = start_capital
shares = 0.0

equity_curve = []
trades = []

for date, row in data.iterrows():
    close = row["Close"]
    buy_sig = bool(row["buy_signal"]) if not pd.isna(row["buy_signal"]) else False
    sell_sig = bool(row["sell_signal"]) if not pd.isna(row["sell_signal"]) else False

    # Execute signals at close
    # Sell logic first so same-day flip is allowed cleanly
    if sell_sig and shares > 0:
        proceeds = shares * close
        cash += proceeds
        trade_pl = proceeds - entry_value  # entry_value tracked on last buy
        trades.append({
            "Date": date,
            "Side": "SELL_ALL",
            "Price": close,
            "Shares": shares,
            "Cash_After": cash,
            "Trade_PnL": trade_pl
        })
        shares = 0.0
        entry_value = 0.0

    # Buy logic
    if buy_sig:
        target_investment = equity * position_fraction
        # Only invest additional if current position value < target
        position_value = shares * close
        additional = max(0.0, target_investment - position_value)
        additional = min(additional, cash)  # cannot exceed available cash
        if additional > 0:
            buy_shares = additional // close  # floor to whole shares
            if buy_shares > 0:
                cost = buy_shares * close
                cash -= cost
                shares += buy_shares
                entry_value = shares * close  # update blended entry value
                trades.append({
                    "Date": date,
                    "Side": "BUY",
                    "Price": close,
                    "Shares": buy_shares,
                    "Cash_After": cash,
                    "Trade_PnL": 0.0
                })

    # Mark-to-market equity
    equity = cash + shares * close
    equity_curve.append({"Date": date, "Equity": equity})

equity_df = pd.DataFrame(equity_curve).set_index("Date")

# -----------------------------
# Max drawdown
# -----------------------------
equity_df["Peak"] = equity_df["Equity"].cummax()
equity_df["Drawdown"] = equity_df["Equity"] / equity_df["Peak"] - 1.0
max_drawdown = equity_df["Drawdown"].min()

# -----------------------------
# Results
# -----------------------------
trades_df = pd.DataFrame(trades)

print("====== Strategy Summary ======")
print(f"Start capital: {start_capital:,.2f}")
print(f"End equity:   {equity:,.2f}")
print(f"Total return: {(equity/start_capital - 1)*100:,.2f}%")
print(f"Max drawdown: {max_drawdown*100:,.2f}%")
print(f"Number of trades: {len(trades_df)}")
print()

print("====== All Trades ======")
print(trades_df.to_string(index=False))
