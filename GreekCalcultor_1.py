import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime,date

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def calculate_iv(market_price, S, K, T, r):
    # If the price is negligible, IV is essentially zero
    if market_price <= 0 or T <= 0:
        return 0.0
    
    # We use a root-finding algorithm (Brent's method) to find where 
    # (Black-Scholes Price - Market Price) == 0
    objective_function = lambda sigma: black_scholes_call(S, K, T, r, sigma) - market_price
    
    try:
        # Search for IV between 0.0001% and 500%
        return brentq(objective_function, 1e-6, 5.0)
    except ValueError:
        return 0.0 # Return 0 if no solution is found (e.g., price is arbitrage-impossible)

def repair_option_df(ticker_symbol, exp_date):
    tk = yf.Ticker(ticker_symbol)
    
    # 1. Get current underlying price
    underlying_price = tk.fast_info['lastPrice']
    
    # 2. Get the options chain
    opt = tk.option_chain(exp_date)
    df = opt.calls # You can repeat this for opt.puts
    
    # 3. Setup parameters
    r = 0.045 # Approximate risk-free rate (4.5%)
    today = datetime.now()
    expiry = datetime.strptime(exp_date, '%Y-%m-%d')
    #expiry = date(2026,2,20).strftime('%Y-%m-%d')  # Example expiration date
    T = (expiry - today).days / 365.0 # Time to expiry in years
    
    # 4. Iterate and fix IV
    def fix_row(row):
        # Only fix if the current IV looks like a placeholder (very small)
        if row['impliedVolatility'] < 0.001:
            # Use mid-price if available, otherwise lastPrice
            mid = (row['bid'] + row['ask']) / 2
            price = mid if mid > 0 else row['lastPrice']
            
            return calculate_iv(price, underlying_price, row['strike'], T, r)
        return row['impliedVolatility']

    df['impliedVolatility'] = df.apply(fix_row, axis=1)
    return df

# Usage
df_fixed = repair_option_df("SPY", "2026-02-20")
print(df_fixed[['contractSymbol', 'lastPrice', 'impliedVolatility']].tail())