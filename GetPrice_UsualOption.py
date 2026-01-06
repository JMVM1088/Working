import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date
import re
from pathlib import Path
import shutil

# --- CONFIG ---
#CSV_PATH = "Stage-2_2025-12-23.csv"  # path to your CSV
#TABLE_NAME = "UsualOptions"            # target table name
# xCONN_STR = (
#         r'DRIVER={ODBC Driver 17 for SQL Server};'
#         r'SERVER=BEELINK;'  # Replace with your server name
#         r'DATABASE=Stock;' # Replace with your database name
#         r'Trusted_Connection=yes;'
#     )
CONN_STR = "mssql+pyodbc://localhost/Stock?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
# Mapping from source CSV header -> destination table columns
COLUMN_MAP = {
    "BusinessDate ": "BusinessDate",   # note trailing space
    "Symbol": "Symbol",
    "Price~": "Price",
    "Exp Date": "Exp Date",
    "DTE": "DTE",
    "Type": "Type",
    "Strike": "Strike",
    "Moneyness": "Moneyness",
    "Bid": "Bid",
    "Latest": "Last",
    "Ask": "Ask",
    "Volume": "Volume",
    "Open Int": "Open Int",
    "Vol/OI": "Vol/OI",
    "Imp Vol": "Imp Vol",
    "IV": "IV",
    "Delta": "Delta",
    "Time": "Time",
}

DEST_COLUMNS = [
    "BusinessDate",
    "Symbol",
    "Price",
    "Exp Date",
    "DTE",
    "Type",
    "Strike",
    "Moneyness",
    "Bid",
    "Last",
    "Ask",
    "Volume",
    "Open Int",
    "Vol/OI",
    "Imp Vol",
    "IV",
    "Delta",
    "Time",
]

NUMERIC_COLS = [
    "Price",
    "Strike",
    "Bid",
    "Last",
    "Ask",
    "Volume",
    "Open Int",
    "Vol/OI",
    "Imp Vol",
    "IV",
    "Delta",
]

DATE_COLS = ["BusinessDate", "Exp Date", "Time"]

def load_csv_to_sql(sFullPathFileName: str, sFileName: str, sTableName: str) -> None:
    df = pd.read_csv(
        sFullPathFileName,
        dtype=str,
        keep_default_na=False,
    )  # all text first [web:93]

    df = df[~df["Symbol"].str.contains("Downloaded from Barchart.com", na=False)]

    # Example: date from file name MM-DD-YYYY
    m = re.search(r"\d{2}-\d{2}-\d{4}", sFileName)
    if not m:
        raise ValueError(f"No MM-DD-YYYY date found in file name: {sFileName}")
    df["BusinessDate "] = m.group(0)

    # Rename columns
    df = df.rename(columns=COLUMN_MAP)
    df = df[[c for c in DEST_COLUMNS if c in df.columns]]

    # Columns that should be numeric (including Strike)
    numeric_cols = [
        "Price",
        "Strike",
        "Bid",
        "Last",
        "Ask",
        "Volume",
        "Open Int",
        "Vol/OI",
        "Imp Vol",
        "IV",
        "Delta",
    ]

    # 1) Remove commas, 2) remove %, 3) convert to numeric
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)   # 1,005.00 -> 1005.00 [web:79][web:90]
                .str.replace("%", "", regex=False)   # 15.07% -> 15.07
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")  # NaN for bad values [web:88]

    # DTE as integer
    if "DTE" in df.columns:
        df["DTE"] = pd.to_numeric(df["DTE"], errors="coerce").astype("Int64")

    # Dates normalized
    for col in ["BusinessDate", "Exp Date", "Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

    engine = create_engine(CONN_STR)
    df.to_sql(sTableName, con=engine, if_exists="append", index=False)  # [web:60][web:72]




def xload_csv_to_sql(sFullPathFileName, sFileName, sTableName):
    # Read CSV; thousands=',' converts 1,005.00 -> 1005.0 already [web:90][web:93]
    df = pd.read_csv(
        sFullPathFileName,
        dtype=str,
        keep_default_na=False,
        thousands=",",
    )

    # Extract date from filename (MM-DD-YYYY as in your regex)
    m = re.search(r"\d{2}-\d{2}-\d{4}", sFileName)
    if not m:
        raise ValueError(f"No MM-DD-YYYY date found in file name: {sFileName}")
    df["BusinessDate"] = m.group(0)

    # Rename columns to match destination schema
    df = df.rename(columns=COLUMN_MAP)

    # Keep only destination columns that exist
    existing_dest = [c for c in DEST_COLUMNS if c in df.columns]
    df = df[existing_dest]

    # --- Type conversions to avoid varchar->numeric errors ---

    # Clean numeric columns: strip %, spaces; convert to numeric [web:79][web:88]
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # DTE as integer (nullable)
    if "DTE" in df.columns:
        df["DTE"] = pd.to_numeric(df["DTE"], errors="coerce").astype("Int64")

    # Dates to YYYY-MM-DD strings
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

    # Create SQLAlchemy engine
    engine = create_engine(CONN_STR)

    # Write to SQL (append into existing table) [web:60][web:72]
    df.to_sql(
        sTableName,
        con=engine,
        if_exists="append",
        index=False,
    )

def process_stage2_files(
    input_dir: str,
    archive_dir: str,
    pattern: str = "unusual-stock-options-activity*"
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
        load_csv_to_sql(f"{in_path/file_path.name}", file_path.name, "UsualOptions")
        # do_something_with(file_path)

        # --- Move file to archive folder ---
        # Example placeholder: print file name, or call your own function
        print(f"Processing: {in_path/file_path.name}")
        # do_something_with(file_path)

        # --- Move file to archive folder ---
        dest = arc_path / file_path.name
        shutil.move(str(file_path), str(dest))  # move like Unix mv [web:27][web:39]
        print(f"Archived to: {dest}")


if __name__ == "__main__":
    engine = create_engine(CONN_STR)
    with engine.begin() as conn:
        conn.execute(text("EXEC sp_UsualOptions_CleanUp"))
  
    process_stage2_files(
        input_dir=r"C:\Users\jv2mk\Downloads", 
        archive_dir=r"C:\Users\jv2mk\OneDrive\Stock\Screener\archive"
    )
    #m = re.search(r"\d{4}-\d{2}-\d{2}", CSV_PATH)  # match yyyy-mm-dd [web:3]
    #load_csv_to_sql()
    #input_dir=r"C:\Users\jv2mk\Downloads"