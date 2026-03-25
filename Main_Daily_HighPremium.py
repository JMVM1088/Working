"""
Simplified ATM Premium Screener
Filters: Volume > 1M, 1-Month ATM Call Premium % > 6%
Ranks by: ATM Premium % descending
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

class SimpleATMPremiumScreener:
    def __init__(self, sql_connection_string):
        self.engine = create_engine(f"mssql+pyodbc:///?odbc_connect={sql_connection_string}")
        
    def fetch_base_stocks(self, min_volume=1000000):
        """Fetch stocks with volume > 1M from SQL"""
        query = f"""
        SELECT Symbol, [Close] as Price, Volume, 
               (High - Low) / [Close] * 100 as DailyRange
        FROM ai_stock_prices p (nolock)
        WHERE [Time] = (SELECT MAX([Time]) FROM ai_stock_prices (nolock))
            AND Volume >= {min_volume}
        """
        return pd.read_sql(query, self.engine)
    
    def get_atm_option(self, symbol):
        """Get 1-month ATM call option"""
        try:
            ticker = yf.Ticker(symbol)
            price = ticker.info.get('currentPrice', 0)
            if price == 0:
                return None
            
            # Find ~30 day expiry
            exps = ticker.options
            target = datetime.now() + timedelta(days=30)
            expiry = min(exps, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target))
            
            # Get ATM call
            chain = ticker.option_chain(expiry)
            calls = chain.calls
            atm = calls.loc[abs(calls['strike'] - price).idxmin()]
            
            return {
                'Symbol': symbol,
                'Price': price,
                'Expiry': expiry,
                'ATM_Strike': atm['strike'],
                'ATM_Premium': atm['lastPrice'],
                'ATM_IV': atm['impliedVolatility'] * 100 if atm['impliedVolatility'] else 0
            }
        except:
            return None
    
    def run_screen(self, min_premium_pct=6.0):
        """Main screen: Find highest ATM premium % stocks"""
        print("="*80)
        print("ATM PREMIUM SCREENER | Volume > 1M | Premium > 6%")
        print("="*80)
        
        # Get base stocks
        stocks = self.fetch_base_stocks()
        print(f"Fetched {len(stocks)} stocks with volume > 1M\n")
        
        results = []
        for _, row in stocks.iterrows():
            opt = self.get_atm_option(row['Symbol'])
            if not opt:
                continue
            
            # Calculate premium %
            premium_pct = (opt['ATM_Premium'] / opt['Price']) * 100
            
            if premium_pct >= min_premium_pct:
                results.append({
                    'Symbol': row['Symbol'],
                    'Price': round(opt['Price'], 2),
                    'Volume': row['Volume'],
                    'ATM_Premium': round(opt['ATM_Premium'], 2),
                    'Premium_Pct': round(premium_pct, 2),
                    'ATM_IV': round(opt['ATM_IV'], 1),
                    'Expiry': opt['Expiry']
                })
                print(f" {row['Symbol']}: ${opt['Price']:.0f} | Premium {premium_pct:.1f}% | IV {opt['ATM_IV']:.0f}%")
            else:
                print(f" {row['Symbol']}: {premium_pct:.1f}% (below 6%)")
        
        # Rank results
        if not results:
            print("\nNo stocks passed the 6% premium filter")
            return pd.DataFrame()
        
        df = pd.DataFrame(results).sort_values('Premium_Pct', ascending=False)
        df['Rank'] = range(1, len(df)+1)
        
        # Display
        print("\n" + "="*80)
        print(f"TOP {len(df)} STOCKS BY ATM PREMIUM %")
        print("="*80)
        print(df[['Rank', 'Symbol', 'Price', 'Premium_Pct', 'ATM_IV', 'Expiry']].to_string(index=False))
        
        # Top pick
        top = df.iloc[0]
        print(f"#1: {top['Symbol']} | Premium: {top['Premium_Pct']:.1f}% of stock price")
        
        # Save to SQL
        df['RunDate'] = datetime.now()
        df.to_sql('ATM_Premium_Results', self.engine, if_exists='append', index=False)
        print(f"Saved to SQL table: ATM_Premium_Results")
        
        return df
    
def main():
    SQL_CONN = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=BEELINK;"
    "DATABASE=stock;"
    "trusted_connection=yes;"
    )
    screener = SimpleATMPremiumScreener(SQL_CONN)
    results = screener.run_screen(min_premium_pct=6.0)
# USAGE
if __name__ == "__main__":
    main()