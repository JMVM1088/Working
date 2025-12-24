import GetPrice_SQL as m
import Util as u


conStr = (
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        r'SERVER=BEELINK;'  # Replace with your server name
        r'DATABASE=Stock;' # Replace with your database name
        r'Trusted_Connection=yes;'
    )

#m.getDailyPrice('US', 'AI_Stock_Historical',8)
#m.getDailyPrice('ETF', 'AI_ETF_Historical',8)
u.sql_execute_query(conStr, "Stock..sp_AI_UpdatePrice_Main 'US'", None)
u.sql_execute_query(conStr, "Stock..sp_AI_UpdatePrice_Main 'ETF'", None)
print("Daily price data retrieved successfully.")