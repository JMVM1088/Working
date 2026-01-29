import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, date

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # PDF and CDF of standard normal distribution
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    if option_type == 'call':
        delta = cdf_d1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2) / 365
    else:
        delta = cdf_d1 - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = (S * pdf_d1 * np.sqrt(T)) / 100  # Per 1% change in IV
    
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega}

# 1. Fetch data from yfinance
ticker_symbol = "SPY"
tk = yf.Ticker(ticker_symbol)
underlying_price = tk.fast_info['lastPrice']

# 2. Get option chain for a specific expiration
#expiration = tk.options[0] # Using the nearest expiration date
expiration = date(2026,2,20).strftime('%Y-%m-%d')  # Example expiration date
opt_chain = tk.option_chain(expiration)
calls = opt_chain.calls

# 3. Pick a specific contract (e.g., the first call)
#contract = calls.iloc[0]
strike = 685 # contract['strike']
contract = calls[calls['strike'] == strike]
iv = contract['impliedVolatility']

# 4. Define time to expiry (in years) and risk-free rate
expiry_dt = datetime.strptime(expiration, '%Y-%m-%d')
#expiry_dt = datetime.date(2026,2,20)
days_to_expiry = (expiry_dt - datetime.now()).days
T = days_to_expiry / 365
r = 0.045  # Estimated 2026 risk-free rate (e.g., 4.5%)

# 5. Execute calculation
greeks = calculate_greeks(underlying_price, strike, T, r, iv)

print(f"Ticker: {ticker_symbol} | Strike: {strike} | Expiry: {expiration}")
for greek, value in greeks.items():
    print(f"{greek}: {value:.4f}")
