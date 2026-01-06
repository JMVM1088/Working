import pandas as pd
from sqlalchemy import create_engine, text
server = "BEELINK"
database = "Stock"
driver = "ODBC Driver 17 for SQL Server"

class DataLoader:
    def __init__(self, server, database, driver="ODBC Driver 17 for SQL Server"):
        self.conn_str = f"mssql+pyodbc://{server}/{database}?driver={driver}&trusted_connection=yes"
        self.engine = create_engine(self.conn_str)

    def get_ohlc(self, ticker, start_date=None, end_date=None):
        query = f"SELECT [Time] as Date, [Open], [High], [Low], [Close], [Volume] FROM AI_Stock_Prices WHERE Symbol = '{ticker}'"
        
        if start_date:
            query += f" AND [Time] >= '{start_date}'"
        if end_date:
            query += f" AND [Time] <= '{end_date}'"
            
        query += " ORDER BY [Date] ASC"
        
        df = pd.read_sql(text(query), self.engine)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        return df
    
    def get_last_score(self, symbol, table_name):
        query = f"SELECT TOP 1 Final_Score FROM {table_name} WHERE Symbol = '{symbol}' ORDER BY RunID DESC"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                return result[0] if result else None
        except Exception:
            return None

    def bulk_insert(self, df, table_name):
        df.to_sql(table_name, self.engine, if_exists='append', index=False, chunksize=50)

class xDataLoader:
    def __init__(self, server, database, driver="ODBC Driver 17 for SQL Server"):
        self.conn_str = f"mssql+pyodbc://{server}/{database}?driver={driver}&trusted_connection=yes"
        self.engine = create_engine(self.conn_str)

    def get_ohlc(self, ticker):
        query = f"SELECT [Time] as Date, [Open], [High], [Low], [Close], [Volume] FROM AI_Stock_Prices WHERE Symbol = '{ticker}' ORDER BY Date ASC"
        df = pd.read_sql(text(query), self.engine)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        return df

    def get_last_score(self, symbol, table_name):
        query = f"SELECT TOP 1 Final_Score FROM {table_name} WHERE Symbol = '{symbol}' ORDER BY RunID DESC"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                return result[0] if result else None
        except Exception:
            return None

    def bulk_insert(self, df, table_name):
        df.to_sql(table_name, self.engine, if_exists='append', index=False, chunksize=50)