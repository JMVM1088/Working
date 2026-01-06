import pandas as pd

CSV_PATH = r"C:\Users\jv2mk\Downloads\StockList_US_2026-01-02.csv"       # your CSV
TABLE_NAME = "AI_Stock_Info"       # base table name

# crude default type mapping; adjust if you know the schema
DEFAULT_SQL_TYPE = "NVARCHAR(255)"


def main():
    # read only header row
    cols = list(pd.read_csv(CSV_PATH, nrows=0).columns)

    # ensure safe identifiers
    cols_clean = [c.strip().replace(" ", "_") for c in cols]

    # build column list with types
    col_defs = ",\n    ".join(f"[{c}] {DEFAULT_SQL_TYPE} NULL" for c in cols_clean)

    # primary key assumption: first column
    pk_col = cols_clean[0]

    # 1. main table
    create_main = f"""
CREATE TABLE dbo.{TABLE_NAME} (
    [{pk_col}] {DEFAULT_SQL_TYPE} NOT NULL,
    {",\n    ".join(f"[{c}] {DEFAULT_SQL_TYPE} NULL" for c in cols_clean[1:])},
    [LastModifiedUtc] DATETIME2(3) NOT NULL CONSTRAINT DF_{TABLE_NAME}_LastMod DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_{TABLE_NAME} PRIMARY KEY ([{pk_col}])
);
"""

    # 2. staging table
    create_staging = f"""
CREATE TABLE dbo.{TABLE_NAME}_Staging (
    {col_defs}
);
"""

    # 3. change log table
    create_log = f"""
CREATE TABLE dbo.{TABLE_NAME}ChangeLog (
    ChangeLogId   BIGINT IDENTITY(1,1) NOT NULL,
    ChangeDateUtc DATETIME2(3) NOT NULL CONSTRAINT DF_{TABLE_NAME}ChangeLog_Date DEFAULT (SYSUTCDATETIME()),
    ChangeSource  VARCHAR(50) NOT NULL,
    [{pk_col}]    {DEFAULT_SQL_TYPE} NOT NULL,
    ColumnName    SYSNAME NOT NULL,
    OldValue      NVARCHAR(MAX) NULL,
    NewValue      NVARCHAR(MAX) NULL,
    CONSTRAINT PK_{TABLE_NAME}ChangeLog PRIMARY KEY (ChangeLogId)
);

CREATE INDEX IX_{TABLE_NAME}ChangeLog_Date
    ON dbo.{TABLE_NAME}ChangeLog (ChangeDateUtc, [{pk_col}], ColumnName);
"""

    # 4. MERGE + logging proc
    # build comparison list for MERGE
    cmp_exprs = []
    for c in cols_clean[1:]:
        cmp_exprs.append(f"ISNULL(T.[{c}], '') <> ISNULL(S.[{c}], '')")
    cmp_clause = " OR\n            ".join(cmp_exprs) if cmp_exprs else "1 = 0"

    # build OUTPUT old/new columns
    out_cols_old_new = []
    for c in cols_clean[1:]:
        out_cols_old_new.append(f"DELETED.[{c}] AS Old_{c}, INSERTED.[{c}] AS New_{c}")
    out_cols_old_new_sql = ",\n        ".join(out_cols_old_new)

    # temp table definition for merge results
    temp_cols = []
    for c in cols_clean[1:]:
        temp_cols.append(f"    Old_{c} {DEFAULT_SQL_TYPE} NULL,\n    New_{c} {DEFAULT_SQL_TYPE} NULL")
    temp_cols_sql = ",\n".join(temp_cols)

    # CROSS APPLY rows for each column
    cross_values = []
    for c in cols_clean[1:]:
        cross_values.append(f"            ('{c}', m.Old_{c}, m.New_{c})")
    cross_values_sql = ",\n".join(cross_values)

    sp_upsert = f"""
CREATE OR ALTER PROCEDURE dbo.Upsert_{TABLE_NAME}_FromStaging
(
    @ChangeSource VARCHAR(50) = 'PythonImport'
)
AS
BEGIN
    SET NOCOUNT ON;

    IF OBJECT_ID('tempdb..#MergeResults') IS NOT NULL
        DROP TABLE #MergeResults;

    CREATE TABLE #MergeResults (
        MergeAction NVARCHAR(10),
        [{pk_col}]  {DEFAULT_SQL_TYPE} NOT NULL,
{temp_cols_sql}
    );

    MERGE dbo.{TABLE_NAME} AS T
    USING dbo.{TABLE_NAME}_Staging AS S
        ON T.[{pk_col}] = S.[{pk_col}]
    WHEN MATCHED AND (
            {cmp_clause}
        )
        THEN UPDATE SET
{",\n".join(f"             T.[{c}] = S.[{c}]" for c in cols_clean[1:])},
             T.[LastModifiedUtc] = SYSUTCDATETIME()
    WHEN NOT MATCHED BY TARGET
        THEN INSERT ([{pk_col}], {", ".join(f"[{c}]" for c in cols_clean[1:])}, [LastModifiedUtc])
             VALUES (S.[{pk_col}], {", ".join(f"S.[{c}]" for c in cols_clean[1:])}, SYSUTCDATETIME())
    OUTPUT
        $action       AS MergeAction,
        INSERTED.[{pk_col}] AS [{pk_col}],
        {out_cols_old_new_sql}
    INTO #MergeResults;

    INSERT dbo.{TABLE_NAME}ChangeLog (ChangeDateUtc, ChangeSource, [{pk_col}], ColumnName, OldValue, NewValue)
    SELECT
        SYSUTCDATETIME(),
        @ChangeSource,
        m.[{pk_col}],
        v.ColumnName,
        v.OldValue,
        v.NewValue
    FROM #MergeResults m
    CROSS APPLY (
{cross_values_sql}
    ) v (ColumnName, OldValue, NewValue)
    WHERE
        m.MergeAction = 'UPDATE'
        AND ISNULL(v.OldValue, '') <> ISNULL(v.NewValue, '');
END;
"""

    # 5. daily changes proc
    sp_daily = f"""
CREATE OR ALTER PROCEDURE dbo.Get_{TABLE_NAME}_DailyChanges
(
    @FromDateUtc DATETIME2(3),
    @ToDateUtc   DATETIME2(3)
)
AS
BEGIN
    SET NOCOUNT ON;

    SELECT
        ChangeLogId,
        ChangeDateUtc,
        ChangeSource,
        [{pk_col}],
        ColumnName,
        OldValue,
        NewValue
    FROM dbo.{TABLE_NAME}ChangeLog
    WHERE ChangeDateUtc >= @FromDateUtc
      AND ChangeDateUtc <  @ToDateUtc
    ORDER BY ChangeDateUtc, [{pk_col}], ColumnName;
END;
"""

    print("-- MAIN TABLE")
    print(create_main)
    print("-- STAGING TABLE")
    print(create_staging)
    print("-- CHANGE LOG TABLE")
    print(create_log)
    print("-- UPSERT + LOGGING PROC")
    print(sp_upsert)
    print("-- DAILY CHANGES PROC")
    print(sp_daily)


if __name__ == "__main__":
    main()
