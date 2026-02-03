-- ============================================================================
-- Neon PostgreSQL Database Schema
-- Portfolio Forecasting & Optimization System
--
-- Optimized for Neon's 0.5GB free tier limit:
-- - Only essential tables for Streamlit dashboard
-- - No logs or debug tables
-- - Minimal indexes
-- ============================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- Table 1: Forecasts
-- ============================================================================
-- Stores individual predictions from each forecaster for each ticker
CREATE TABLE IF NOT EXISTS forecasts (
    id VARCHAR(36) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    as_of_date TIMESTAMP NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    forecaster VARCHAR(50) NOT NULL,
    predicted_price DECIMAL(12, 4) NOT NULL,
    predicted_return DECIMAL(10, 6) NOT NULL,
    actual_price DECIMAL(12, 4),
    actual_return DECIMAL(10, 6),
    error DECIMAL(10, 6),
    forecaster_config TEXT
);

-- Essential index for dashboard queries
CREATE INDEX IF NOT EXISTS idx_forecasts_lookup
    ON forecasts(created_at DESC, ticker, forecaster);

-- ============================================================================
-- Table 2: Portfolios
-- ============================================================================
-- Stores portfolio weights from each (forecaster, allocator) combination
CREATE TABLE IF NOT EXISTS portfolios (
    id VARCHAR(36) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    as_of_date TIMESTAMP NOT NULL,
    rebalance_date TIMESTAMP NOT NULL,
    forecaster VARCHAR(50) NOT NULL,
    allocator VARCHAR(50) NOT NULL,
    weights TEXT NOT NULL,  -- JSON: {"AAPL": 0.25, "MSFT": 0.20}
    expected_return DECIMAL(10, 6),
    expected_volatility DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 6),
    actual_return DECIMAL(10, 6),
    allocator_config TEXT
);

-- Essential index for dashboard queries
CREATE INDEX IF NOT EXISTS idx_portfolios_lookup
    ON portfolios(created_at DESC, forecaster, allocator);

-- ============================================================================
-- Table 3: Backtests
-- ============================================================================
-- Stores backtesting results for model comparison
CREATE TABLE IF NOT EXISTS backtests (
    id VARCHAR(36) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    forecaster VARCHAR(50) NOT NULL,
    allocator VARCHAR(50),
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    n_folds INTEGER NOT NULL,
    mae DECIMAL(10, 6),
    rmse DECIMAL(10, 6),
    mase DECIMAL(10, 6),
    directional_accuracy DECIMAL(5, 4),
    sharpe_ratio DECIMAL(10, 6),
    total_return DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    config TEXT,
    metrics TEXT
);

-- Essential index for dashboard queries
CREATE INDEX IF NOT EXISTS idx_backtests_lookup
    ON backtests(created_at DESC, forecaster);

-- ============================================================================
-- View: Latest Forecasts (for dashboard)
-- ============================================================================
CREATE OR REPLACE VIEW v_latest_forecasts AS
SELECT
    f.*,
    DATE(f.created_at) as run_date
FROM forecasts f
WHERE f.created_at >= NOW() - INTERVAL '7 days'
ORDER BY f.created_at DESC;

-- ============================================================================
-- View: Latest Portfolios (for dashboard)
-- ============================================================================
CREATE OR REPLACE VIEW v_latest_portfolios AS
SELECT
    p.*,
    DATE(p.created_at) as run_date
FROM portfolios p
WHERE p.created_at >= NOW() - INTERVAL '7 days'
ORDER BY p.created_at DESC;

-- ============================================================================
-- View: Model Leaderboard (for dashboard)
-- ============================================================================
CREATE OR REPLACE VIEW v_model_leaderboard AS
SELECT
    forecaster,
    AVG(mase) as avg_mase,
    AVG(directional_accuracy) as avg_dir_accuracy,
    AVG(mae) as avg_mae,
    COUNT(*) as n_runs,
    MAX(created_at) as last_run
FROM backtests
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY forecaster
ORDER BY avg_mase ASC NULLS LAST;

-- ============================================================================
-- Cleanup Function (call periodically to stay within 0.5GB limit)
-- ============================================================================
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 30)
RETURNS TABLE(
    deleted_forecasts BIGINT,
    deleted_portfolios BIGINT,
    deleted_backtests BIGINT
) AS $$
DECLARE
    cutoff_date TIMESTAMP := NOW() - (days_to_keep || ' days')::INTERVAL;
    df BIGINT;
    dp BIGINT;
    db BIGINT;
BEGIN
    -- Delete old forecasts
    DELETE FROM forecasts WHERE created_at < cutoff_date;
    GET DIAGNOSTICS df = ROW_COUNT;

    -- Delete old portfolios
    DELETE FROM portfolios WHERE created_at < cutoff_date;
    GET DIAGNOSTICS dp = ROW_COUNT;

    -- Delete old backtests
    DELETE FROM backtests WHERE created_at < cutoff_date;
    GET DIAGNOSTICS db = ROW_COUNT;

    RETURN QUERY SELECT df, dp, db;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Usage Examples
-- ============================================================================

-- Check database size (Neon limit: 0.5GB = 512MB)
-- SELECT pg_size_pretty(pg_database_size(current_database())) as db_size;

-- Cleanup old data (keep last 30 days)
-- SELECT * FROM cleanup_old_data(30);

-- Get latest forecasts for dashboard
-- SELECT * FROM v_latest_forecasts LIMIT 100;

-- Get model leaderboard
-- SELECT * FROM v_model_leaderboard;

-- ============================================================================
-- End of Schema
-- ============================================================================
