CREATE OR ALTER PROCEDURE dbo.sp_DailyVolatilityMaster
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @Today DATE = CAST(GETDATE() AS DATE);
    
    PRINT '=== DAILY VOLATILITY DASHBOARD ===';
    PRINT CONCAT('Run Date: ', @Today);
    
    -- 1. REFRESH daily alerts
    EXEC dbo.sp_GenerateDailyVolatilityAlerts;
    
    PRINT CHAR(10) + '1. MARKET SNAPSHOT:' + CHAR(10);
    EXEC dbo.sp_SingleDaySnapshot;
    
    PRINT CHAR(10) + '2. TOP 20 LONGS (High/Extreme + UpTrend):' + CHAR(10);
    
    -- Fixed: Add TOP 20 to CTE
    WITH ranked_signals AS (
        SELECT TOP 20
            Symbol, AlertLevel, ZScore, VolatilityChange, VolRegime,
            (ABS(ZScore) * 0.6) + 
            (CASE WHEN AlertLevel IN ('High','Extreme') THEN 2 ELSE 0 END) * 0.4 AS ConvictionScore
        FROM dbo.DailyVolatilityAlerts
        WHERE ABS(ZScore) >= 2.0
        ORDER BY 
            CASE WHEN VolatilityChange > 0 THEN 1 ELSE 2 END,
            ABS(ZScore) DESC  -- Now legal with TOP 20
    )
    SELECT * FROM ranked_signals ORDER BY ConvictionScore DESC;
    
    PRINT CHAR(10) + '3. MARKET REGIME:' + CHAR(10);
    SELECT TOP 1 * FROM dbo.MarketShockIndex ORDER BY ReportDate DESC;
    
    PRINT CHAR(10) + '4. RISK SUMMARY:' + CHAR(10);
    SELECT 
        COUNT(*) AS TotalAlerts,
        SUM(CASE WHEN ABS(ZScore) >= 3 THEN 1 ELSE 0 END) AS ExtremeCount,
        AVG(ABS(ZScore)) AS MarketAvgAbsZScore,
        MAX(ABS(ZScore)) AS MaxSingleNameShock,
        SUM(CASE WHEN VolatilityChange > 0 THEN 1 ELSE 0 END)*1.0 / COUNT(*) AS UpTrendPct
    FROM dbo.DailyVolatilityAlerts;
    
    PRINT CHAR(10) + '=== END DASHBOARD ===';
END;
GO
