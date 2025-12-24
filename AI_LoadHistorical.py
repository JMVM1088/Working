import os
import pandas as pd
import pyodbc
from pathlib import Path
from datetime import datetime
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

# SQL Server Connection Details - MODIFY THESE
connection_string = (
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        r'SERVER=BEELINK;'  # Replace with your server name
        r'DATABASE=Stock;' # Replace with your database name
        r'Trusted_Connection=yes;'
    )
# Folder Configuration
CSV_FOLDER = r"C:\Users\jv2mk\OneDrive\Stock\HistoricalData_AI\ETF_1"  # Path to CSV folder
BACKUP_SUBFOLDER = "Bkp"  # Subfolder name for moved files

# Table Configuration
TABLE_NAME = "AI_ETF_prices"  # Name of SQL table to create/append to

# ============================================================================
# LOGGING SETUP
# ============================================================================

log_file = os.path.join(CSV_FOLDER, f"csv_loader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_sql_connection():
    """Establish connection to SQL Server"""
    try:
        if AUTH_TYPE.lower() == "windows":
            conn_str = f"Driver={{ODBC Driver 17 for SQL Server}};Server={SERVER};Database={DATABASE};Trusted_Connection=yes"
        else:
            conn_str = f"Driver={{ODBC Driver 17 for SQL Server}};Server={SERVER};Database={DATABASE};UID={USERNAME};PWD={PASSWORD}"
        
        conn = pyodbc.connect(conn_str)
        logger.info(f"Connected to SQL Server: {SERVER}.{DATABASE}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to SQL Server: {str(e)}")
        raise

def table_exists(cursor, table_name):
    """Check if table exists in database"""
    cursor.execute(f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", table_name)
    return cursor.fetchone() is not None

def create_table_if_not_exists(cursor, table_name):
    """Create stock_prices table if it doesn't exist"""
    if table_exists(cursor, table_name):
        logger.info(f"Table {table_name} already exists")
        return
    
    create_table_sql = f"""
    CREATE TABLE {table_name} (
        id INT PRIMARY KEY IDENTITY(1,1),
        Symbol NVARCHAR(10) NOT NULL,
        time DATE NOT NULL,
        [open] FLOAT,
        [high] FLOAT,
        [low] FLOAT,
        [close] FLOAT,
        [Volume] BIGINT,
        LoadedAt DATETIME DEFAULT GETDATE(),
        UNIQUE(Symbol, time)
    )
    """
    try:
        cursor.execute(create_table_sql)
        logger.info(f"Created table {table_name}")
    except Exception as e:
        logger.error(f"Error creating table: {str(e)}")
        raise

def insert_data_to_sql_bkp(cursor, table_name, df, symbol):
    """Insert dataframe into SQL Server table"""
    # Add Symbol column
    df.insert(0, 'Symbol', symbol)
    
    # Get columns from dataframe
    columns = df.columns.tolist()
    col_str = ', '.join([f"[{col}]" for col in columns])
    placeholders = ', '.join(['?' for _ in columns])
    
    insert_sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders})"
    
    try:
        # Use fast insertion with executemany
        rows_inserted = 0
        for index, row in df.iterrows():
            try:
                cursor.execute(insert_sql, row.values.tolist())
                rows_inserted += 1
            except pyodbc.IntegrityError as e:
                # Skip duplicate rows (same Symbol + time combination)
                logger.warning(f"Skipped duplicate row for {symbol} at time {row['time']}")
                continue
            except Exception as e:
                logger.error(f"Error inserting row {index}: {str(e)}")
                raise
        
        cursor.commit()
        logger.info(f"Inserted {rows_inserted} rows for {symbol} into {table_name}")
        return rows_inserted
    except Exception as e:
        logger.error(f"Error inserting data for {symbol}: {str(e)}")
        raise
def insert_data_to_sql(cursor, table_name, df, symbol):
    """Insert dataframe into SQL Server table"""
    # Add Symbol column
    df.insert(0, 'Symbol', symbol)
    
    # Restrict to columns that exist in the table
    df = df[['Symbol', 'time', 'open', 'high', 'low', 'close', 'Volume']]
    
    columns = df.columns.tolist()
    col_str = ', '.join([f"[{col}]" for col in columns])
    placeholders = ', '.join(['?' for _ in columns])
    
    insert_sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders})"
    
    try:
        rows_inserted = 0
        for index, row in df.iterrows():
            try:
                cursor.execute(insert_sql, row.values.tolist())
                rows_inserted += 1
            except pyodbc.IntegrityError:
                logger.warning(f"Skipped duplicate row for {symbol} at time {row['time']}")
                continue
            except Exception as e:
                logger.error(f"Error inserting row {index}: {str(e)}")
                raise
        
        cursor.commit()
        logger.info(f"Inserted {rows_inserted} rows for {symbol} into {table_name}")
        return rows_inserted
    except Exception as e:
        logger.error(f"Error inserting data for {symbol}: {str(e)}")
        raise
# ============================================================================
# FILE OPERATIONS
# ============================================================================

def create_backup_folder():
    """Create backup subfolder if it doesn't exist"""
    backup_path = os.path.join(CSV_FOLDER, BACKUP_SUBFOLDER)
    os.makedirs(backup_path, exist_ok=True)
    return backup_path

def move_file_to_backup(file_path, backup_path):
    """Move processed CSV file to backup folder"""
    try:
        filename = os.path.basename(file_path)
        destination = os.path.join(backup_path, filename)
        
        # Handle duplicate filenames in backup
        if os.path.exists(destination):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join(backup_path, f"{base}_{counter}{ext}")):
                counter += 1
            destination = os.path.join(backup_path, f"{base}_{counter}{ext}")
        
        os.rename(file_path, destination)
        logger.info(f"Moved {filename} to {BACKUP_SUBFOLDER}/")
        return True
    except Exception as e:
        logger.error(f"Error moving file {file_path}: {str(e)}")
        return False

# ============================================================================
# DATA TRANSFORMATION
# ============================================================================

def convert_unix_to_date(df):
    """Convert Unix timestamp (seconds) to YYYY-MM-DD format"""
    try:
        # Convert Unix timestamp (seconds) to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Format as YYYY-MM-DD
        df['time'] = df['time'].dt.strftime('%Y-%m-%d')
        
        logger.info("Converted Unix timestamp to YYYY-MM-DD format")
        return df
    except Exception as e:
        logger.error(f"Error converting timestamp: {str(e)}")
        raise

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_csv_files():
    """Main function to process all CSV files"""
    logger.info("=" * 80)
    logger.info(f"Starting CSV to SQL Server Import Process")
    logger.info(f"CSV Folder: {CSV_FOLDER}")
    # logger.info(f"Database: {SERVER}.{DATABASE}")
    # logger.info(f"Table: {TABLE_NAME}")
    # logger.info("=" * 80)
    
    # Validate folder
    if not os.path.isdir(CSV_FOLDER):
        logger.error(f"CSV folder does not exist: {CSV_FOLDER}")
        return
    
    # Find all CSV files
    csv_files = list(Path(CSV_FOLDER).glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {CSV_FOLDER}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Create backup folder
    backup_path = create_backup_folder()
    
    # Connect to database
    #conn = get_sql_connection()
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    
    # Create table if needed
    create_table_if_not_exists(cursor, TABLE_NAME)
    
    # Process each CSV file
    successful = 0
    failed = 0
    
    for csv_file in csv_files:
        filename = csv_file.name
        symbol = csv_file.stem  # Filename without extension
        
        logger.info("-" * 80)
        logger.info(f"Processing: {filename} (Symbol: {symbol})")
        
        try:
            # Read CSV, skipping the first row (header)
            df = pd.read_csv(csv_file, header=None, skiprows=1)
            logger.info(f"Loaded CSV with {len(df)} rows (header skipped)")
            
            # Assign column names based on your data structure
            df.columns = ['time', 'open', 'high', 'low', 'close',  'Volume','Volume MA']
            #df.columns = ['time', 'open', 'high', 'low', 'close', 'Basis', 'Upper', 'Lower', 'Volume', 'Volume MA']
            
            logger.info(f"Columns assigned: {', '.join(df.columns.tolist())}")
            
            # Convert Unix timestamp to YYYY-MM-DD format
            df = convert_unix_to_date(df)
            
            # Data validation
            if df.empty:
                logger.warning(f"CSV file is empty: {filename}")
                failed += 1
                continue
            
            # Insert into database
            rows_inserted = insert_data_to_sql(cursor, TABLE_NAME, df, symbol)
            
            # Move file to backup
            if move_file_to_backup(str(csv_file), backup_path):
                successful += 1
                logger.info(f"✓ Successfully processed {filename}")
            else:
                failed += 1
                logger.error(f"✗ Failed to move {filename}")
        
        except Exception as e:
            logger.error(f"✗ Error processing {filename}: {str(e)}")
            failed += 1
            continue
    
    # Close connection
    cursor.close()
    conn.close()
    
    # Summary
    logger.info("=" * 80)
    logger.info(f"SUMMARY:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total: {len(csv_files)}")
    logger.info(f"  Log file: {log_file}")
    logger.info("=" * 80)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        process_csv_files()
    except Exception as e:
        logger.critical(f"Critical error in main process: {str(e)}")
        raise
