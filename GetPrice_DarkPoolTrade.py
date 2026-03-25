import pandas as pd
from sqlalchemy import create_engine
from datetime import date, datetime
import re
from pathlib import Path
import shutil

# --- CONFIG ---
CSV_PATH = "Stage-2_2025-12-23.csv"  # path to your CSV
TABLE_NAME = "Stage2"            # target table name
# xCONN_STR = (
#         r'DRIVER={ODBC Driver 17 for SQL Server};'
#         r'SERVER=BEELINK;'  # Replace with your server name
#         r'DATABASE=Stock;' # Replace with your database name
#         r'Trusted_Connection=yes;'
#     )
CONN_STR = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
# Mapping from source CSV header -> destination table columns
TRADE_COLUMN_MAP = {
    "BusinessDate": "BusinessDate",
    "Time" : "ReportDate",
    "Symbol" : "Symbol",
    "Type": "Type",
    "Price": "Price",
    "Size" :  "Size",
    "Value" : "Value",
    "Filled" : "Filled",
    "Sector" : "Sector",
    "Mcap Proportion" : "Mcap_Proportion"

}
TRADE_DEST_COLUMNS = [
    "BusinessDate",
    "ReportDate",
    "Symbol",
    "Type",
    "Price",
    "Size",
    "Value",
    "Filled",
    "Sector",
    "Mcap_Proportion"
]
DashBoard_COLUMN_MAP = {
    "BusinessDate" : "BusinessDate",
    "Symbol" : "Symbol",
    "Sentiment": "Sentiment",
    "TotalAmount": "TotalAmount",
    "DailyAmount Change": "DailyAmountChange",
    "AtAsk": "AtAsk",
    "AtBid": "AtBid",
    "NetValue": "NetValue",
    "DPTrades": "DPTrades",
    "BlockTrades" : "BlockTrades"
}

DashBoard_DEST_COLUMNS = [
    "BusinessDate" ,
    "Symbol" ,
    "Sentiment",
    "TotalAmount",
    "DailyAmountChange",
    "AtAsk",
    "AtBid",
    "NetValue",
    "DPTrades",
    "BlockTrades"
]
# ---------------

def load_csv_to_sql(sFullPathFileName, sFileName, sTableName,colMap, destCol) -> None:
    # Read CSV with proper handling of commas in quoted fields (Description, etc.) [file:1]
    df = pd.read_csv(
        sFullPathFileName,
        dtype=str,
        keep_default_na=False
    )

    # Extract date from filename and set BusinessDate
    m = re.search(r"\d{4}\d{2}\d{2}", sFileName)  # yyyy-mm-dd [web:3]
    if not m:
        raise ValueError(f"No yyyy-mm-dd date found in file name: {CSV_PATH}")
    # formatted_date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    df["BusinessDate"] = datetime.strptime(m.group(0), "%Y%m%d").strftime("%Y-%m-%d")
    # df["BusinessDate"] = m.group(0)

    # Rename columns to match destination schema
    df = df.rename(columns=colMap)

    # Reorder / restrict to destination columns
    df = df[destCol]

    # Create SQLAlchemy engine
    engine = create_engine(CONN_STR)

    # Write to SQL
    df.to_sql(
        sTableName,
        con=engine,
        if_exists="append",   # change to 'append' once table exists
        index=False
    )

def process_files(
    input_dir: str,
    archive_dir: str,
    pattern: str ,
    tableName: str,
    ColMap: str,
    DestCol: str

) -> None:
    """
    Loop through input_dir, find files matching pattern (e.g. 'Stage_2*'),
    run processing logic, then move each processed file to archive_dir.
    """

    in_path = Path(input_dir)
    arc_path = Path(archive_dir)

    # Ensure archive directory exists
    arc_path.mkdir(parents=True, exist_ok=True)

    # Iterate over matching files, non-recursive
    for file_path in in_path.glob(pattern):  # glob pattern match [web:24][web:37]
        if not file_path.is_file():
            continue

        # --- Your processing logic goes here ---
        load_csv_to_sql(f"{in_path/file_path.name}", file_path.name,tableName,ColMap,DestCol)
        # do_something_with(file_path)

        # --- Move file to archive folder ---
        # Example placeholder: print file name, or call your own function
        print(f"Processing: {in_path/file_path.name}")
        # do_something_with(file_path)

        # --- Move file to archive folder ---
        dest = arc_path / file_path.name
        shutil.move(str(file_path), str(dest))  # move like Unix mv [web:27][web:39]
        print(f"Archived to: {dest}")
def main():
    process_files(
            input_dir=r"C:\Users\jv2mk\Downloads", 
            archive_dir=r"C:\Users\jv2mk\OneDrive\Stock\Screener\archive",
            pattern="DarkPoolTrade*",
            tableName="AI_Darkpool_Trade",
            ColMap=TRADE_COLUMN_MAP,
            DestCol=TRADE_DEST_COLUMNS
            )
    process_files(
            input_dir=r"C:\Users\jv2mk\Downloads", 
            archive_dir=r"C:\Users\jv2mk\OneDrive\Stock\Screener\archive",
            pattern="DarkPoolBlock*",
            tableName="AI_Darkpool_Block",
            ColMap=TRADE_COLUMN_MAP,
            DestCol=TRADE_DEST_COLUMNS
            )
    process_files(
            input_dir=r"C:\Users\jv2mk\Downloads", 
            archive_dir=r"C:\Users\jv2mk\OneDrive\Stock\Screener\archive",
            pattern="DarkPoolDash*",
            tableName="AI_Darkpool_DashBoard",
            ColMap=DashBoard_COLUMN_MAP,
            DestCol=DashBoard_DEST_COLUMNS
            )
if __name__ == "__main__":
    main()

    #m = re.search(r"\d{4}-\d{2}-\d{2}", CSV_PATH)  # match yyyy-mm-dd [web:3]
    #load_csv_to_sql()
