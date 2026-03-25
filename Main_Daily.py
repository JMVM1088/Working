import Util as u
from datetime import datetime, time 
import Main_Daily_Indicators as mdi
import Main_Daily_Indicators_ETF as mde
import Main_Daily_VWAP as mdv
import Util_Email as ue
import holidays
import FVG_Daily as fvg
import Util as u
import Main_Daily_SNDK as sndk
import Main_Daily_HighPremium as mhp
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
nyse_holidays = holidays.NYSE(years=now.year) 

# Define 6 pm time object (18:00 in 24-hour format)
six_pm = time(18, 0, 0)
# Define log file path
filename = datetime.now().strftime("%Y%m%d_%I%M") + "_DailyRun.txt"
logFile = rf"C:\Trading\Logs\{filename}"
logger = ue.setup_logger("Daily Run", logFile)
# Check if it's a weekday (Monday=0 to Friday=4)
is_weekday = now.weekday() <= 4
try:
    if is_weekday and now.date() not in nyse_holidays:
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
            logger.info("EOD Get Price completed.")
            # Call EOD Indicator and VWAP calculations
            mdi.main()
            logger.info("EOD Stock Indicator calculation completed.")
            mdv.update_recent_weekly_vwap()
            logger.info("EOD VWAP calculation completed.")
            mde.main()
            logger.info("ETF Indicator calculation completed.")
            fvg.generate_daily_recommendations()
            logger.info("FVG calculation completed.")
            # Send success email
            logger.info("EOD ETF indicators run completed successfully.")
            # Update the uptrend workflow and gap up workflow
            u.sql_execute_query(conStr, "Stock..[sp_UpTrend_DailyWorkflow] ", None)
            logger.info("UpTrend Daily Workflow completed successfully.")
            # Update the gap up workflow
            u.sql_execute_query(conStr, "Stock..[sp_DailyRun_GapUp] ", None)
            logger.info("Daily Gap Up completed successfully.")
            #SNDK workflow
            sndk.main()
            logger.info("SNDK workflow completed successfully.")
            # High Premium workflow
            mhp.main()
            logger.info("High Premium workflow completed successfully.")
            ue.send_email("INFO:EOD Daily Run completed", "EOD Daily Run Log File succeed.",None)
        else:
            logger.info("Intra day run")
            #print("It is currently BEFORE 6 pm (or exactly 6 pm).")
            u.getDailyPrice(conStr,'ETF', 'AI_ETF_Intra',1,"I")
            u.getDailyPrice(conStr,'US', 'AI_Stock_Intra',1,"I")
            ue.send_email("INFO:Intra Daily Run completed", "Intra Daily Run Log File succeed.",None)
    else:
        if now.date() in nyse_holidays:
            #logger.info(f"Today is a US holiday: {nyse_holidays.get(now.date())}. No trading activity.")
            ue.send_email("INFO:Intra Daily Run completed", f"Today is a US holiday: {nyse_holidays.get(now.date())}. No trading activity.",None)
        else:
            print(f"Today is a weekend ({now.strftime('%A')}).")
       
    # Print a success message
    print("Daily price data retrieved successfully.")
except Exception as e:
    logger.error(f"An error occurred: {e}")
    ue.send_email("ERR:Daily Run failed", "Intra Daily Run Log File failed.",rf"{logFile}")