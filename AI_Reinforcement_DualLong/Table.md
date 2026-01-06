-- ============================================================
-- 🚀 DUAL-HORIZON DQN - COMPLETE SQL SERVER SCHEMA
-- Full table definitions for long-only model with 20 features
-- ============================================================

USE [YourDatabaseName];
GO

-- ============================================================
-- TABLE 1: dqn_daily_predictions
-- ============================================================
-- Stores daily backtest predictions with all 20 features
-- One row per trading day per backtest run
-- ============================================================

IF OBJECT_ID('dqn_daily_predictions', 'U') IS NOT NULL
    DROP TABLE dqn_daily_predictions;
GO

CREATE TABLE dqn_daily_predictions (
    -- ===== PRIMARY KEY =====
    prediction_id BIGINT PRIMARY KEY IDENTITY(1,1),
    
    -- ===== RUN IDENTIFICATION =====
    run_id NVARCHAR(150) NOT NULL,                  -- Unique backtest run ID
    run_date DATETIME NOT NULL,                     -- When backtest was executed
    
    -- ===== TRADING DATE & SYMBOL =====
    [date] DATE NOT NULL,                           -- Trading date (YYYY-MM-DD)
    symbol NVARCHAR(10) NOT NULL,                   -- 'SPY', 'QQQ', etc.
    
    -- ===== PRICE DATA (OHLCV) =====
    price FLOAT NOT NULL,                           -- Close price ($)
    [high] FLOAT,                                   -- High price ($)
    [low] FLOAT,                                    -- Low price ($)
    volume BIGINT,                                  -- Trading volume (shares)
    
    -- ===== MODEL PREDICTION =====
    action INT NOT NULL,                            -- 0=FLAT, 1=LONG (long-only model)
    signal NVARCHAR(10) NOT NULL,                   -- 'FLAT', 'LONG' (no SHORT)
    
    -- ===== POSITION MANAGEMENT =====
    [position] FLOAT NOT NULL,                      -- Current position: 0.0 (FLAT) to 1.0 (FULL LONG)
    position_confidence FLOAT,                      -- Confidence score (0-1) for sizing
    position_change FLOAT NOT NULL,                 -- |target_pos - prev_pos|
    
    -- ===== P&L METRICS =====
    daily_pnl FLOAT NOT NULL,                       -- Daily profit/loss ($)
    trade_cost FLOAT NOT NULL,                      -- Transaction cost ($)
    
    -- ===== PORTFOLIO STATE =====
    equity FLOAT NOT NULL,                          -- Portfolio value ($), starting at 1.0
    equity_pct FLOAT NOT NULL,                      -- Portfolio value (100x), starting at 100
    
    -- ===== RETURNS =====
    daily_return FLOAT NOT NULL,                    -- Daily return (%)
    ret_5d_forecast FLOAT NOT NULL,                 -- 5-day forward return (%) from training
    ret_60d_forecast FLOAT NOT NULL,                -- 60-day forward return (%) from training
    
    -- ===== BASE FEATURES (8) =====
    
    -- Return metrics
    ret_1d FLOAT,                                   -- 1-day return
    ret_5d FLOAT,                                   -- 5-day return
    
    -- Volatility metrics
    vol_10d FLOAT,                                  -- 10-day volatility
    vol_20d FLOAT,                                  -- 20-day volatility
    
    -- Moving average distances
    dist_sma_20 FLOAT,                              -- Distance from 20-day SMA (normalized)
    dist_sma_50 FLOAT,                              -- Distance from 50-day SMA (normalized)
    dist_sma_200 FLOAT,                             -- Distance from 200-day SMA (normalized)
    
    -- Volume Z-score
    vol_z_20 FLOAT,                                 -- Volume Z-score (20-day)
    
    -- ===== ADVANCED FEATURES (12) =====
    
    -- MOMENTUM INDICATORS (4)
    rsi_14 FLOAT,                                   -- RSI 14-period (normalized -1 to 1)
    macd_line FLOAT,                                -- MACD line (normalized)
    macd_signal FLOAT,                              -- MACD signal line (normalized)
    macd_hist FLOAT,                                -- MACD histogram (normalized)
    
    -- VOLATILITY INDICATORS (3)
    atr_14 FLOAT,                                   -- Average True Range 14-period (normalized)
    vol_ratio FLOAT,                                -- Volatility ratio: vol_20d / vol_10d
    vol_trend FLOAT,                                -- Volume momentum (5-day pct change)
    
    -- TREND INDICATORS (2)
    adx_14 FLOAT,                                   -- ADX 14-period (0-1, trend strength)
    bb_position FLOAT,                              -- Bollinger Band position (0-1)
    
    -- PRICE ACTION INDICATORS (1)
    range_norm FLOAT,                               -- Price position in 20-day range (0-1)
    
    -- RETURN PATTERNS (1)
    ret_skew FLOAT,                                 -- Return skewness (20-day)
    
    -- VOLUME-PRICE INDICATORS (1)
    obv_ratio FLOAT,                                -- On-Balance Volume ratio (normalized)
    
    -- ===== TIMESTAMPS =====
    created_at DATETIME DEFAULT GETDATE()
)

-- Create indexes for fast queries
CREATE NONCLUSTERED INDEX idx_run_id 
    ON dqn_daily_predictions(run_id)
    INCLUDE ([date], signal, daily_pnl, equity, [position])

CREATE NONCLUSTERED INDEX idx_date 
    ON dqn_daily_predictions([date])
    INCLUDE (run_id, signal, daily_pnl, [position])

CREATE NONCLUSTERED INDEX idx_symbol_date 
    ON dqn_daily_predictions(symbol, [date])
    INCLUDE (price, signal, equity, [position])

CREATE NONCLUSTERED INDEX idx_features
    ON dqn_daily_predictions(run_id, [date])
    INCLUDE (rsi_14, macd_line, macd_hist, atr_14, adx_14, bb_position, obv_ratio, vol_20d)

GO

-- ============================================================
-- TABLE 2: dqn_prediction_summary
-- ============================================================
-- High-level summary statistics for each backtest run
-- One row per run
-- ============================================================

IF OBJECT_ID('dqn_prediction_summary', 'U') IS NOT NULL
    DROP TABLE dqn_prediction_summary;
GO

CREATE TABLE dqn_prediction_summary (
    -- ===== PRIMARY KEY =====
    summary_id INT PRIMARY KEY IDENTITY(1,1),
    
    -- ===== RUN IDENTIFICATION =====
    run_id NVARCHAR(150) NOT NULL UNIQUE,           -- Unique run identifier
    run_date DATETIME NOT NULL,                     -- When backtest was executed
    
    -- ===== SYMBOL & PERIOD =====
    symbol NVARCHAR(10) NOT NULL,                   -- 'SPY', 'QQQ', etc.
    period_start DATE NOT NULL,                     -- First trading date
    period_end DATE NOT NULL,                       -- Last trading date
    days INT NOT NULL,                              -- Total trading days
    
    -- ===== RETURNS =====
    total_return_pct FLOAT NOT NULL,                -- Total return (%)
    final_equity FLOAT NOT NULL,                    -- Final portfolio value ($)
    
    -- ===== RISK-ADJUSTED RETURNS =====
    sharpe_ratio FLOAT,                             -- Sharpe ratio (annualized)
    sortino_ratio FLOAT,                            -- Sortino ratio (optional)
    max_drawdown_pct FLOAT,                         -- Maximum drawdown (%)
    calmar_ratio FLOAT,                             -- Calmar ratio (optional)
    
    -- ===== TRADE STATISTICS =====
    total_trades INT NOT NULL,                      -- Total position changes
    dqn_trades INT,                                 -- DQN model trades
    random_trades INT,                              -- Random baseline trades
    buy_count INT,                                  -- LONG entries
    sell_count INT,                                 -- (not used in long-only)
    flat_days INT,                                  -- FLAT position days
    
    -- ===== WIN/LOSS STATISTICS =====
    winning_days INT NOT NULL,                      -- Days with positive P&L
    losing_days INT NOT NULL,                       -- Days with negative P&L
    win_rate FLOAT NOT NULL,                        -- Win rate (%)
    avg_win FLOAT,                                  -- Average winning day ($)
    avg_loss FLOAT,                                 -- Average losing day ($)
    best_day FLOAT,                                 -- Best day P&L ($)
    worst_day FLOAT,                                -- Worst day P&L ($)
    
    -- ===== COST ANALYSIS =====
    total_cost FLOAT,                               -- Total transaction costs ($)
    cost_pct_of_return FLOAT,                       -- Costs as % of return
    
    -- ===== RISK METRICS =====
    daily_volatility FLOAT,                         -- Daily volatility (%)
    profit_factor FLOAT,                            -- Total wins / Total losses
    
    -- ===== MODEL METADATA =====
    model_version NVARCHAR(50),                     -- 'longonly_2015_2024', 'retrained_2015_2024', etc.
    feature_count INT,                              -- 20 for advanced, 8 for base
    training_days INT,                              -- Days used for training
    training_period NVARCHAR(50),                   -- e.g., '2015-2024'
    backtest_version NVARCHAR(50),                  -- Script version
    notes NVARCHAR(MAX),                            -- Free-form notes
    
    -- ===== TIMESTAMPS =====
    created_at DATETIME DEFAULT GETDATE()
)

-- Create indexes
CREATE NONCLUSTERED INDEX idx_summary_run_id 
    ON dqn_prediction_summary(run_id)
    INCLUDE (period_start, period_end, total_return_pct, sharpe_ratio, model_version)

CREATE NONCLUSTERED INDEX idx_summary_model
    ON dqn_prediction_summary(model_version)
    INCLUDE (run_id, total_return_pct, sharpe_ratio, max_drawdown_pct, win_rate)

CREATE NONCLUSTERED INDEX idx_summary_return
    ON dqn_prediction_summary(total_return_pct DESC)
    INCLUDE (model_version, sharpe_ratio, max_drawdown_pct)

GO

-- ============================================================
-- TABLE 3: dqn_model_metadata
-- ============================================================
-- Metadata about trained models
-- One row per model version
-- ============================================================

IF OBJECT_ID('dqn_model_metadata', 'U') IS NOT NULL
    DROP TABLE dqn_model_metadata;
GO

CREATE TABLE dqn_model_metadata (
    -- ===== PRIMARY KEY =====
    model_id INT PRIMARY KEY IDENTITY(1,1),
    
    -- ===== MODEL IDENTIFICATION =====
    model_name NVARCHAR(100) NOT NULL UNIQUE,       -- e.g., 'dqn_spy_longonly_2015_2024'
    model_version NVARCHAR(50) NOT NULL,            -- '1.0_longonly'
    symbol NVARCHAR(10) NOT NULL,                   -- 'SPY'
    model_type NVARCHAR(50),                        -- 'longonly', 'retrained', 'advanced', 'base'
    
    -- ===== TRAINING CONFIGURATION =====
    training_start DATE NOT NULL,                   -- Training period start
    training_end DATE NOT NULL,                     -- Training period end
    training_days INT NOT NULL,                     -- Days in training set
    training_steps INT NOT NULL,                    -- Total training steps (e.g., 500K)
    
    -- ===== MODEL ARCHITECTURE =====
    learning_rate FLOAT,                            -- Learning rate (e.g., 5e-4)
    buffer_size INT,                                -- Replay buffer size (e.g., 100,000)
    batch_size INT,                                 -- Batch size (e.g., 128)
    gamma FLOAT,                                    -- Discount factor (e.g., 0.99)
    target_update_interval INT,                     -- Target network update freq (e.g., 500)
    hidden_layers NVARCHAR(100),                    -- Network architecture (e.g., '[512, 512, 512]')
    exploration_fraction FLOAT,                     -- Exploration fraction
    exploration_final_eps FLOAT,                    -- Final epsilon
    
    -- ===== FEATURES =====
    feature_count INT NOT NULL,                     -- 20 for advanced, 8 for base, 21 for obs space
    features NVARCHAR(MAX),                         -- JSON list of all features
    observation_size INT,                           -- Observation vector size (21)
    action_space INT,                               -- Number of actions (2 for long-only)
    
    -- ===== NORMALIZATION =====
    feat_mean NVARCHAR(MAX),                        -- JSON array of feature means (20 values)
    feat_std NVARCHAR(MAX),                         -- JSON array of feature std devs (20 values)
    
    -- ===== VALIDATION PERFORMANCE =====
    validation_return FLOAT,                        -- Validation return (%)
    validation_sharpe FLOAT,                        -- Validation Sharpe ratio
    validation_max_drawdown FLOAT,                  -- Validation max drawdown (%)
    validation_days INT,                            -- Validation period days
    
    -- ===== FILES =====
    model_file_path NVARCHAR(500),                  -- Path to .zip model file
    model_file_size INT,                            -- File size (bytes)
    feat_mean_path NVARCHAR(500),                   -- Path to feat_mean.npy
    feat_std_path NVARCHAR(500),                    -- Path to feat_std.npy
    
    -- ===== METADATA =====
    status NVARCHAR(20),                            -- 'ACTIVE', 'ARCHIVED', 'TEST'
    description NVARCHAR(MAX),                      -- Model description
    notes NVARCHAR(MAX),                            -- Training notes, issues
    
    -- ===== TIMESTAMPS =====
    created_at DATETIME DEFAULT GETDATE(),
    updated_at DATETIME DEFAULT GETDATE()
)

-- Create indexes
CREATE NONCLUSTERED INDEX idx_model_name 
    ON dqn_model_metadata(model_name)
    INCLUDE (model_version, feature_count, status, training_steps, training_start)

CREATE NONCLUSTERED INDEX idx_model_type
    ON dqn_model_metadata(model_type)
    INCLUDE (model_name, status, validation_return, training_start)

GO

-- ============================================================
-- TABLE 4: dqn_signal_analysis
-- ============================================================
-- Pre-aggregated signal performance statistics
-- Refreshed after each backtest
-- ============================================================

IF OBJECT_ID('dqn_signal_analysis', 'U') IS NOT NULL
    DROP TABLE dqn_signal_analysis;
GO

CREATE TABLE dqn_signal_analysis (
    -- ===== PRIMARY KEY =====
    analysis_id INT PRIMARY KEY IDENTITY(1,1),
    
    -- ===== RUN IDENTIFICATION =====
    run_id NVARCHAR(150) NOT NULL,
    
    -- ===== SIGNAL TYPE =====
    signal NVARCHAR(10) NOT NULL,                   -- 'FLAT', 'LONG'
    
    -- ===== COUNT & FREQUENCY =====
    signal_count INT NOT NULL,                      -- Days with this signal
    signal_pct FLOAT NOT NULL,                      -- Percentage of total days
    
    -- ===== PERFORMANCE =====
    total_pnl FLOAT NOT NULL,                       -- Total P&L ($)
    avg_pnl FLOAT NOT NULL,                         -- Average daily P&L ($)
    total_return FLOAT NOT NULL,                    -- Total return (%)
    avg_daily_return FLOAT NOT NULL,                -- Average daily return (%)
    
    -- ===== WIN STATISTICS =====
    winning_days INT NOT NULL,                      -- Days with positive P&L
    losing_days INT NOT NULL,                       -- Days with negative P&L
    win_rate FLOAT NOT NULL,                        -- Win rate (%)
    
    -- ===== RISK =====
    std_dev FLOAT,                                  -- Daily return std dev
    sharpe_ratio FLOAT,                             -- Sharpe ratio for this signal
    max_drawdown FLOAT,                             -- Max drawdown (%)
    
    -- ===== METADATA =====
    created_at DATETIME DEFAULT GETDATE()
)

-- Create index
CREATE NONCLUSTERED INDEX idx_signal_run 
    ON dqn_signal_analysis(run_id, signal)
    INCLUDE (total_pnl, win_rate, sharpe_ratio, avg_pnl)

GO

-- ============================================================
-- TABLE 5: dqn_feature_importance_analysis
-- ============================================================
-- Feature importance and correlation analysis
-- Updated after each analysis run
-- ============================================================

IF OBJECT_ID('dqn_feature_importance_analysis', 'U') IS NOT NULL
    DROP TABLE dqn_feature_importance_analysis;
GO

CREATE TABLE dqn_feature_importance_analysis (
    -- ===== PRIMARY KEY =====
    importance_id INT PRIMARY KEY IDENTITY(1,1),
    
    -- ===== RUN IDENTIFICATION =====
    run_id NVARCHAR(150) NOT NULL,
    analysis_date DATETIME NOT NULL DEFAULT GETDATE(),
    
    -- ===== FEATURE DETAILS =====
    feature_name NVARCHAR(50) NOT NULL,             -- e.g., 'rsi_14', 'macd_line'
    feature_group NVARCHAR(50),                     -- 'momentum', 'volatility', 'trend', etc.
    feature_type NVARCHAR(20),                      -- 'BASE' or 'ADVANCED'
    
    -- ===== CORRELATION ANALYSIS =====
    corr_with_signal FLOAT,                         -- Correlation with model signal
    corr_with_pnl FLOAT,                            -- Correlation with daily P&L
    corr_with_return FLOAT,                         -- Correlation with returns
    corr_with_5d_forward FLOAT,                     -- Correlation with 5-day forward returns
    corr_with_60d_forward FLOAT,                    -- Correlation with 60-day forward returns
    
    -- ===== IMPORTANCE SCORE =====
    importance_score FLOAT,                         -- Importance score (0-100)
    
    -- ===== STATISTICS =====
    mean_value FLOAT,                               -- Mean value
    std_dev FLOAT,                                  -- Standard deviation
    min_value FLOAT,
    max_value FLOAT,
    
    -- ===== RANKING =====
    rank_by_importance INT,                         -- 1-20 ranking
    percentile FLOAT,                               -- Percentile ranking (0-100)
    
    -- ===== STATISTICAL SIGNIFICANCE =====
    is_significant BIT,                             -- 1 if statistically significant
    p_value FLOAT,
    
    -- ===== METADATA =====
    created_at DATETIME DEFAULT GETDATE()
)

-- Create index
CREATE NONCLUSTERED INDEX idx_importance_analysis
    ON dqn_feature_importance_analysis(run_id, rank_by_importance)
    INCLUDE (feature_name, importance_score, is_significant, corr_with_pnl)

GO

-- ============================================================
-- TABLE 6: dqn_backtest_comparison
-- ============================================================
-- Compare multiple backtest runs (model versions)
-- ============================================================

IF OBJECT_ID('dqn_backtest_comparison', 'U') IS NOT NULL
    DROP TABLE dqn_backtest_comparison;
GO

CREATE TABLE dqn_backtest_comparison (
    -- ===== PRIMARY KEY =====
    comparison_id INT PRIMARY KEY IDENTITY(1,1),
    
    -- ===== RUN IDENTIFICATION =====
    run_id NVARCHAR(150) NOT NULL,
    
    -- ===== COMPARISON GROUP =====
    comparison_group NVARCHAR(100),                 -- e.g., 'spy_2024_models'
    comparison_date DATETIME DEFAULT GETDATE(),
    
    -- ===== BENCHMARK (Buy & Hold) =====
    benchmark_return FLOAT,                         -- Buy & Hold return (%)
    benchmark_sharpe FLOAT,                         -- Buy & Hold Sharpe ratio
    benchmark_max_dd FLOAT,                         -- Buy & Hold max drawdown (%)
    
    -- ===== DQN PERFORMANCE VS BENCHMARK =====
    excess_return FLOAT,                            -- DQN return - Benchmark return
    excess_sharpe FLOAT,                            -- DQN Sharpe - Benchmark Sharpe
    
    -- ===== RELATIVE METRICS =====
    outperformance_days INT,                        -- Days beating benchmark
    underperformance_days INT,                      -- Days losing to benchmark
    hit_rate FLOAT,                                 -- % of days outperforming
    
    -- ===== RISK COMPARISON =====
    dqn_max_drawdown FLOAT,                         -- DQN max drawdown (%)
    drawdown_reduction FLOAT,                       -- How much DQN reduced drawdown (%)
    
    -- ===== METADATA =====
    notes NVARCHAR(MAX),
    created_at DATETIME DEFAULT GETDATE()
)

-- Create index
CREATE NONCLUSTERED INDEX idx_comparison_run 
    ON dqn_backtest_comparison(run_id)
    INCLUDE (excess_return, hit_rate, comparison_date, benchmark_return)

GO

-- ============================================================
-- VIEWS FOR ANALYSIS
-- ============================================================

-- View 1: Latest Model Performance
CREATE OR ALTER VIEW vw_latest_model_performance AS
SELECT TOP 1
    s.run_id,
    s.symbol,
    m.model_name,
    m.model_type,
    m.training_start,
    m.training_end,
    m.feature_count,
    m.action_space,
    s.period_start,
    s.period_end,
    s.days,
    s.total_return_pct,
    s.sharpe_ratio,
    s.max_drawdown_pct,
    s.win_rate,
    s.profit_factor,
    s.total_trades,
    s.run_date
FROM dqn_prediction_summary s
LEFT JOIN dqn_model_metadata m ON s.model_version = m.model_type
ORDER BY s.run_date DESC
GO

-- View 2: Model Comparison (Return & Risk)
CREATE OR ALTER VIEW vw_model_comparison AS
SELECT
    m.model_type,
    m.training_start,
    m.training_end,
    m.feature_count,
    m.action_space,
    COUNT(DISTINCT s.run_id) as num_backtests,
    ROUND(AVG(s.total_return_pct), 2) as avg_return,
    ROUND(AVG(s.sharpe_ratio), 2) as avg_sharpe,
    ROUND(AVG(s.max_drawdown_pct), 2) as avg_max_dd,
    ROUND(AVG(s.win_rate), 1) as avg_win_rate,
    ROUND(MAX(s.total_return_pct), 2) as best_return,
    ROUND(MIN(s.total_return_pct), 2) as worst_return,
    ROUND(AVG(s.profit_factor), 2) as avg_profit_factor
FROM dqn_model_metadata m
LEFT JOIN dqn_prediction_summary s ON m.model_version = s.model_version
GROUP BY m.model_type, m.training_start, m.training_end, m.feature_count, m.action_space
GO

-- View 3: Daily Advanced Features
CREATE OR ALTER VIEW vw_daily_advanced_features AS
SELECT
    run_id,
    [date],
    price,
    signal,
    [position],
    position_confidence,
    daily_pnl,
    equity,
    ROUND(rsi_14, 3) as rsi,
    ROUND(macd_hist, 3) as macd,
    ROUND(atr_14, 3) as atr,
    ROUND(adx_14, 3) as adx,
    ROUND(bb_position, 3) as bb,
    ROUND(obv_ratio, 3) as obv,
    ROUND(vol_20d, 3) as vol
FROM dqn_daily_predictions
WHERE rsi_14 IS NOT NULL
GO

-- View 4: Performance by Year and Model
CREATE OR ALTER VIEW vw_annual_performance AS
SELECT
    s.model_version,
    YEAR(d.[date]) as year,
    COUNT(*) as trading_days,
    ROUND(SUM(d.daily_pnl), 4) as total_pnl,
    ROUND(SUM(CASE WHEN d.daily_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate_pct,
    ROUND(AVG(d.daily_pnl), 6) as avg_daily_pnl,
    ROUND(MAX(d.equity) / MIN(d.equity) - 1, 4) as period_return
FROM dqn_daily_predictions d
JOIN dqn_prediction_summary s ON d.run_id = s.run_id
GROUP BY s.model_version, YEAR(d.[date])
GO

-- View 5: Top Performing Runs
CREATE OR ALTER VIEW vw_top_performing_runs AS
SELECT TOP 20
    run_id,
    symbol,
    model_version,
    period_start,
    period_end,
    days,
    total_return_pct,
    sharpe_ratio,
    max_drawdown_pct,
    win_rate,
    total_trades,
    profit_factor,
    run_date
FROM dqn_prediction_summary
ORDER BY total_return_pct DESC
GO

-- ============================================================
-- STORED PROCEDURES
-- ============================================================

-- Procedure 1: Get Model Summary
CREATE OR ALTER PROCEDURE sp_get_model_summary
    @model_version NVARCHAR(50)
AS
BEGIN
    SELECT TOP 1
        m.model_name,
        m.model_type,
        m.training_start,
        m.training_end,
        m.training_days,
        m.training_steps,
        m.feature_count,
        m.action_space,
        m.learning_rate,
        m.batch_size,
        m.gamma,
        COUNT(s.run_id) as num_backtests,
        ROUND(AVG(s.total_return_pct), 2) as avg_return,
        ROUND(AVG(s.sharpe_ratio), 2) as avg_sharpe,
        ROUND(MAX(s.total_return_pct), 2) as best_return,
        ROUND(MIN(s.total_return_pct), 2) as worst_return,
        m.created_at as model_created_date
    FROM dqn_model_metadata m
    LEFT JOIN dqn_prediction_summary s ON m.model_version = s.model_version
    WHERE m.model_version = @model_version
    GROUP BY m.model_name, m.model_type, m.training_start, m.training_end, 
             m.training_days, m.training_steps, m.feature_count, m.action_space,
             m.learning_rate, m.batch_size, m.gamma, m.created_at
END
GO

-- Procedure 2: Compare Two Models
CREATE OR ALTER PROCEDURE sp_compare_two_models
    @model_version_1 NVARCHAR(50),
    @model_version_2 NVARCHAR(50)
AS
BEGIN
    SELECT
        m1.model_type as model_1_type,
        m2.model_type as model_2_type,
        m1.feature_count as model_1_features,
        m2.feature_count as model_2_features,
        ROUND(AVG(s1.total_return_pct), 2) as model_1_avg_return,
        ROUND(AVG(s2.total_return_pct), 2) as model_2_avg_return,
        ROUND(AVG(s1.total_return_pct) - AVG(s2.total_return_pct), 2) as return_difference,
        ROUND(AVG(s1.sharpe_ratio), 2) as model_1_sharpe,
        ROUND(AVG(s2.sharpe_ratio), 2) as model_2_sharpe,
        ROUND(AVG(s1.max_drawdown_pct), 2) as model_1_max_dd,
        ROUND(AVG(s2.max_drawdown_pct), 2) as model_2_max_dd,
        ROUND(AVG(s1.win_rate), 1) as model_1_win_rate,
        ROUND(AVG(s2.win_rate), 1) as model_2_win_rate,
        ROUND(AVG(s1.profit_factor), 2) as model_1_profit_factor,
        ROUND(AVG(s2.profit_factor), 2) as model_2_profit_factor
    FROM dqn_model_metadata m1
    LEFT JOIN dqn_prediction_summary s1 ON m1.model_version = s1.model_version
    CROSS JOIN dqn_model_metadata m2
    LEFT JOIN dqn_prediction_summary s2 ON m2.model_version = s2.model_version
    WHERE m1.model_version = @model_version_1 AND m2.model_version = @model_version_2
    GROUP BY m1.model_type, m2.model_type, m1.feature_count, m2.feature_count
END
GO

-- Procedure 3: Get Run Details
CREATE OR ALTER PROCEDURE sp_get_run_details
    @run_id NVARCHAR(150)
AS
BEGIN
    SELECT
        s.run_id,
        s.symbol,
        s.model_version,
        s.period_start,
        s.period_end,
        s.days,
        s.total_return_pct,
        s.final_equity,
        s.sharpe_ratio,
        s.max_drawdown_pct,
        s.total_trades,
        s.winning_days,
        s.losing_days,
        s.win_rate,
        s.profit_factor,
        s.daily_volatility,
        ROUND(COUNT(d.prediction_id), 0) as total_predictions,
        ROUND(SUM(CASE WHEN d.signal = 'LONG' THEN 1 ELSE 0 END), 0) as long_days,
        ROUND(SUM(CASE WHEN d.signal = 'FLAT' THEN 1 ELSE 0 END), 0) as flat_days,
        ROUND(AVG(d.[position]), 2) as avg_position,
        ROUND(AVG(d.position_confidence), 2) as avg_confidence,
        ROUND(SUM(d.trade_cost), 4) as total_costs,
        s.run_date
    FROM dqn_prediction_summary s
    LEFT JOIN dqn_daily_predictions d ON s.run_id = d.run_id
    WHERE s.run_id = @run_id
    GROUP BY s.run_id, s.symbol, s.model_version, s.period_start, s.period_end,
             s.days, s.total_return_pct, s.final_equity, s.sharpe_ratio, s.max_drawdown_pct,
             s.total_trades, s.winning_days, s.losing_days, s.win_rate, s.profit_factor,
             s.daily_volatility, s.run_date
END
GO

-- Procedure 4: Get Top Runs by Criteria
CREATE OR ALTER PROCEDURE sp_get_top_runs
    @top_n INT = 10,
    @order_by NVARCHAR(20) = 'return'  -- 'return', 'sharpe', 'dd'
AS
BEGIN
    IF @order_by = 'sharpe'
        SELECT TOP (@top_n) * FROM vw_top_performing_runs
        ORDER BY sharpe_ratio DESC
    ELSE IF @order_by = 'dd'
        SELECT TOP (@top_n) * FROM vw_top_performing_runs
        ORDER BY max_drawdown_pct ASC
    ELSE
        SELECT TOP (@top_n) * FROM vw_top_performing_runs
        ORDER BY total_return_pct DESC
END
GO

-- ============================================================
-- SAMPLE DATA - Insert Retrained Model Metadata
-- ============================================================

INSERT INTO dqn_model_metadata (
    model_name,
    model_version,
    symbol,
    model_type,
    training_start,
    training_end,
    training_days,
    training_steps,
    learning_rate,
    buffer_size,
    batch_size,
    gamma,
    target_update_interval,
    hidden_layers,
    exploration_fraction,
    exploration_final_eps,
    feature_count,
    features,
    observation_size,
    action_space,
    feat_mean,
    feat_std,
    status,
    description,
    notes
)
VALUES (
    'dqn_spy_longonly_2015_2024',
    'longonly_2015_2024',
    'SPY',
    'longonly',
    '2015-01-01',
    '2024-01-01',
    2261,
    500000,
    5e-4,
    100000,
    128,
    0.99,
    500,
    '[512, 512, 512]',
    0.15,
    0.05,
    20,
    '["ret_1d","ret_5d","vol_10d","vol_20d","dist_sma_20","dist_sma_50","dist_sma_200","vol_z_20","rsi_14","macd_line","macd_signal","macd_hist","atr_14","vol_ratio","vol_trend","adx_14","bb_position","range_norm","ret_skew","obv_ratio"]',
    21,
    2,
    '[]',
    '[]',
    'ACTIVE',
    'DQN model trained on 2015-2024 (long-only, no shorts)',
    'Confidence-based position sizing. Trained to eliminate SHORT trades which were unprofitable.'
)
GO

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check all tables exist
SELECT 
    SCHEMA_NAME(schema_id) as schema_name,
    name as table_name,
    create_date
FROM sys.tables
WHERE name LIKE 'dqn_%'
ORDER BY name
GO

-- Check all views exist
SELECT 
    SCHEMA_NAME(schema_id) as schema_name,
    name as view_name,
    create_date
FROM sys.views
WHERE name LIKE 'vw_%'
ORDER BY name
GO

-- Check all stored procedures exist
SELECT 
    SCHEMA_NAME(schema_id) as schema_name,
    name as procedure_name,
    create_date
FROM sys.procedures
WHERE name LIKE 'sp_%'
ORDER BY name
GO

-- Check table sizes and row counts
SELECT 
    t.name as table_name,
    (SELECT COUNT(*) FROM sys.partitions p WHERE p.object_id = t.object_id AND p.index_id < 2) as row_count,
    (SUM(a.total_pages) * 8 / 1024.0) as size_mb
FROM sys.tables t
INNER JOIN sys.indexes i ON t.object_id = i.object_id
INNER JOIN sys.partitions p ON i.object_id = p.object_id AND i.index_id = p.index_id
INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
WHERE t.name LIKE 'dqn_%'
GROUP BY t.name, t.object_id
ORDER BY size_mb DESC
GO

-- ============================================================
-- EXAMPLE QUERIES FOR ANALYSIS
-- ============================================================

-- Query 1: Latest backtest summary
-- SELECT TOP 1 * FROM dqn_prediction_summary ORDER BY run_date DESC

-- Query 2: Compare all models
-- SELECT * FROM vw_model_comparison

-- Query 3: View latest model performance
-- SELECT * FROM vw_latest_model_performance

-- Query 4: Get specific run details
-- EXEC sp_get_run_details @run_id = 'backtest_SPY_longonly_2015_2024_20251224_220634'

-- Query 5: Compare two models
-- EXEC sp_compare_two_models 'longonly_2015_2024', 'retrained_2015_2024'

-- Query 6: Top 10 performing runs
-- EXEC sp_get_top_runs @top_n = 10, @order_by = 'return'

-- Query 7: Annual performance
-- SELECT * FROM vw_annual_performance ORDER BY model_version, year

-- Query 8: Daily data for specific run
-- SELECT TOP 100 [date], signal, [position], position_confidence, 
--        daily_pnl, equity, rsi_14, macd_hist, adx_14, bb_position
-- FROM dqn_daily_predictions
-- WHERE run_id = 'backtest_SPY_longonly_2015_2024_20251224_220634'
-- ORDER BY [date]

-- ============================================================
 SELECT * FROM dqn_daily_predictions
   WHERE run_id = 'backtest_SPY_longonly_2015_2024_20251224_222047'

 SELECT * FROM [dqn_prediction_summary]
   WHERE run_id = 'backtest_SPY_longonly_2015_2024_20251224_222047'
