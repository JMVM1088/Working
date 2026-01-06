/*
This query gives you the "Efficiency" of your ensemble. Here is how to read the Capture_Ratio_Pct:

> 100% (The Efficient Alpha): Your strategy made more than the stock did. This usually happens because you stayed in for the big up-moves but your Regime Filter or Stability Filter successfully pulled you out during the big crashes.

50% - 90% (The Conservative Participant): You captured most of the move but missed some gains. This is very common for "Trend Following" systems that wait for confirmation before entering.

Negative % (The Counter-Trend): The stock went up, but your strategy lost money (or vice-versa). This indicates a "Whipsaw"—your indicators are lagging, causing you to buy at the top and sell at the bottom.

0% (The Sideline): You missed the entire move. This suggests your threshold is likely set too high.

Key Insights from this Query
Efficiency per Day: Compare Strategy_Return_Pct to ActiveDays. If you made 10% in only 5 active days, while the market took 60 days to make 12%, your Return-per-Day-Invested is massive. This is a sign of a very high-quality ensemble.

Filter Effectiveness: If a stock's Market_Return_Pct is negative (a falling stock) but your Strategy_Return_Pct is 0, your Capture Ratio will be 0. In this case, 0 is a perfect score. It proves your Regime filter successfully protected your capital from a downtrend.

*/
WITH DailyReturns AS (
    -- Step 1: Calculate daily price change (%)
    SELECT 
        s.Symbol,
        s.ReportDate,
        s.Final_Score,
        p.[Close] as ClosePrice,
        LAG(p.[Close]) OVER (PARTITION BY s.Symbol ORDER BY s.ReportDate) as PrevClose,
        (p.[Close] - LAG(p.[Close]) OVER (PARTITION BY s.Symbol ORDER BY s.ReportDate)) 
            / NULLIF(LAG(p.[Close]) OVER (PARTITION BY s.Symbol ORDER BY s.ReportDate), 0) as DailyPctChange
    FROM StrategyScores s (nolock)
    JOIN AI_Stock_prices p (nolock) ON s.Symbol = p.Symbol AND s.ReportDate = p.[Time]
    WHERE s.RunID LIKE 'BATCH_%'
),
PerformanceCalc AS (
    -- Step 2: Sum up returns for the benchmark vs the strategy
    SELECT 
        Symbol,
        -- Total Market Return (Buy & Hold)
        SUM(DailyPctChange) AS TotalMarketReturn,
        -- Strategy Return (Only days where we had an active signal)
        SUM(CASE WHEN Final_Score <> 0 THEN DailyPctChange ELSE 0 END) AS StrategyReturn,
        COUNT(*) as TotalDays,
        SUM(CASE WHEN Final_Score <> 0 THEN 1 ELSE 0 END) as ActiveDays
    FROM DailyReturns
    WHERE DailyPctChange IS NOT NULL
    GROUP BY Symbol
)
-- Step 3: Final Ratio Calculation
SELECT 
    Symbol,
    CAST(TotalMarketReturn * 100 AS DECIMAL(10,2)) as Market_Return_Pct,
    CAST(StrategyReturn * 100 AS DECIMAL(10,2)) as Strategy_Return_Pct,
    CAST((StrategyReturn / NULLIF(TotalMarketReturn, 0)) * 100 AS DECIMAL(10,2)) as Capture_Ratio_Pct,
    ActiveDays,
    TotalDays
FROM PerformanceCalc
ORDER BY Capture_Ratio_Pct DESC;


-===============================
WITH DailyReturns AS (
    SELECT 
        s.Symbol,
        s.ReportDate,
        s.Final_Score,
        p.[Close],
        LAG(p.[Close]) OVER (PARTITION BY s.Symbol ORDER BY s.ReportDate) as PrevClose,
        (p.[Close] - LAG(p.[Close]) OVER (PARTITION BY s.Symbol ORDER BY s.ReportDate)) 
            / NULLIF(LAG(p.[Close]) OVER (PARTITION BY s.Symbol ORDER BY s.ReportDate), 0) as DailyPctChange
    FROM StrategyScores s (nolock)
    JOIN AI_Stock_Prices p  (nolock) ON s.Symbol = p.Symbol AND s.ReportDate = p.[Time]
    WHERE s.RunID LIKE 'BATCH_%'
),
DownsideAnalysis AS (
    SELECT 
        Symbol,
        -- Total returns of the stock only on days it dropped > 2%
        SUM(CASE WHEN DailyPctChange < -0.02 THEN DailyPctChange ELSE 0 END) AS MarketDownsideSum,
        -- Our strategy's returns only on those specific "crash" days
        SUM(CASE WHEN DailyPctChange < -0.02 AND Final_Score <> 0 THEN DailyPctChange ELSE 0 END) AS StrategyDownsideSum,
        -- Overall Strategy and Market returns for context
        SUM(DailyPctChange) AS TotalMarketReturn,
        SUM(CASE WHEN Final_Score <> 0 THEN DailyPctChange ELSE 0 END) AS TotalStrategyReturn
    FROM DailyReturns
    WHERE DailyPctChange IS NOT NULL
    GROUP BY Symbol
)
SELECT 
    Symbol,
    CAST(TotalMarketReturn * 100 AS DECIMAL(10,2)) as Market_Return_Pct,
    CAST(TotalStrategyReturn * 100 AS DECIMAL(10,2)) as Strategy_Return_Pct,
    -- DOWNSIDE CAPTURE: Lower is better. 0% is perfect. 100% means you felt the full crash.
    CAST((StrategyDownsideSum / NULLIF(MarketDownsideSum, 0)) * 100 AS DECIMAL(10,2)) as Downside_Capture_Pct,
    -- OVERALL CAPTURE: Higher is better.
    CAST((TotalStrategyReturn / NULLIF(TotalMarketReturn, 0)) * 100 AS DECIMAL(10,2)) as Overall_Capture_Pct
FROM DownsideAnalysis
ORDER BY Downside_Capture_Pct ASC; -- Sort by best defenders first