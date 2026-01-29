import yfinance as yf
import mibian
from datetime import datetime

symbol = "SPY"
expiry = "2026-02-20"          # option expiry
strike = 705                   # example strike
r = 0.05                       # risk‑free rate (5% placeholder)

# 1. Get underlying spot
spy = yf.Ticker(symbol)
spot = spy.history(period="1d")["Close"][-1]

# 2. Get option chain
chain = spy.option_chain(expiry)
calls = chain.calls
puts = chain.puts

# pick the row matching your strike
call_row = calls[calls["strike"] == strike].iloc[0]
put_row  = puts[puts["strike"] == strike].iloc[0]

call_mid = (call_row["bid"] + call_row["ask"]) / 2
put_mid  = (put_row["bid"] + put_row["ask"]) / 2

# 3. Time to expiry in days
expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
days_to_exp = (expiry_dt - datetime.now()).days

# 4. Implied vol via Mibian then Greeks
# Call
iv_call_model = mibian.BS([spot, strike, r*100, days_to_exp], callPrice=call_mid)
iv_call = iv_call_model.impliedVolatility

call_greeks = mibian.BS([spot, strike, r*100, days_to_exp], iv_call)

print("CALL Greeks:")
print("  IV   :", iv_call)
print("  Delta:", call_greeks.callDelta)
print("  Gamma:", call_greeks.gamma)
print("  Theta:", call_greeks.callTheta)
print("  Vega :", call_greeks.vega)

# Put
iv_put_model = mibian.BS([spot, strike, r*100, days_to_exp], putPrice=put_mid)
iv_put = iv_put_model.impliedVolatility

put_greeks = mibian.BS([spot, strike, r*100, days_to_exp], iv_put)

print("PUT Greeks:")
print("  IV   :", iv_put)
print("  Delta:", put_greeks.putDelta)
print("  Gamma:", put_greeks.gamma)
print("  Theta:", put_greeks.putTheta)
print("  Vega :", put_greeks.vega)
