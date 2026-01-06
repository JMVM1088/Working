import Util as u
from datetime import datetime, time 


# Define the connection string
conStr = (
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        r'SERVER=BEELINK;'  # Replace with your server name
        r'DATABASE=Stock;' # Replace with your database name
        r'Trusted_Connection=yes;'
    )

# Get the current local date and time
now = datetime.now()
current_time = now.time()

# Define 6 pm time object (18:00 in 24-hour format)
six_pm = time(11, 0, 0)

# Check if it's a weekday (Monday=0 to Friday=4)

is_weekday = 0
is_weekday = 0 <= 4
if is_weekday:
    print(f"Today is a weekday ({now.strftime('%A')}).")
    if current_time > six_pm:
        print("It is currently AFTER 6 pm.")
        u.sql_execute_query(conStr, "Stock..sp_AI_UpdatePrice_Main 'CleanUp'", None)
        # Call the function to get daily price data
        u.getDailyPrice(conStr,'US', 'AI_Stock_EOD',8,"E")
        u.getDailyPrice(conStr,'ETF', 'AI_ETF_EOD',8,"E")
        # Call the function to update the price data in the database
        u.sql_execute_query(conStr, "Stock..sp_AI_UpdatePrice_Main 'US'", None)
        u.sql_execute_query(conStr, "Stock..sp_AI_UpdatePrice_Main 'ETF'", None)
    else:

        print("It is currently BEFORE 6 pm (or exactly 6 pm).")
        u.getDailyPrice(conStr,'ETF', 'AI_ETF_Intra',1,"I")
        u.getDailyPrice(conStr,'US', 'AI_Stock_Intra',1,"I")
else:
    print(f"Today is a weekend ({now.strftime('%A')}).")
# Print a success message
print("Daily price data retrieved successfully.")