import pyodbc
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

class DatabaseConnection:
    """Manage SQL Server connection"""
    
    def __init__(self):
        self.server = os.getenv('DB_SERVER', 'localhost')
        self.database = os.getenv('DB_NAME', 'TradeBlog')
        self.username = os.getenv('DB_USER', 'sa')
        self.password = os.getenv('DB_PASSWORD')
        
        self.connection_string = (
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={self.server};'
            f'DATABASE={self.database};'
            f'UID={self.username};'
            f'PWD={self.password};'
        )
    
    def connect(self):
        """Establish connection"""
        try:
            conn = pyodbc.connect(self.connection_string)
            print("✅ Connected to SQL Server successfully")
            return conn
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None
    
    def execute_query(self, query: str, params: List = None) -> bool:
        """Execute query"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Query execution error: {e}")
            return False
    
    def fetch_query(self, query: str, params: List = None) -> List:
        """Fetch query results"""
        try:
            conn = self.connect()
            if params:
                df = pd.read_sql(query, conn, params=params)
            else:
                df = pd.read_sql(query, conn)
            
            conn.close()
            return df
        except Exception as e:
            print(f"❌ Fetch error: {e}")
            return None

# ============================================================================
# TRADEBLOG MANAGER
# ============================================================================

class TradeBlogManager:
    """Main trade logging system"""
    
    def __init__(self):
        self.db = DatabaseConnection()
    
    def insert_batch(self, batch_data: Dict) -> int:
        """Insert new batch entry into database"""
        
        query = """
        EXEC sp_InsertBatch
            @BatchName = ?,
            @EntryDate = ?,
            @ExpirationDate = ?,
            @PutLongStrike = ?,
            @PutShortStrike = ?,
            @CallShortStrike = ?,
            @CallLongStrike = ?,
            @EntryPrice = ?,
            @CreditCollected = ?,
            @SpreadWidth = ?,
            @NumberOfSpreads = ?,
            @EntryIVRank = ?,
            @EntryVIX = ?
        """
        
        params = [
            batch_data['batch_name'],
            batch_data['entry_date'],
            batch_data['expiration_date'],
            batch_data['put_long_strike'],
            batch_data['put_short_strike'],
            batch_data['call_short_strike'],
            batch_data['call_long_strike'],
            batch_data['entry_price'],
            batch_data['credit_collected'],
            batch_data['spread_width'],
            batch_data['number_of_spreads'],
            batch_data['entry_iv_rank'],
            batch_data['entry_vix'],
        ]
        
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            cursor.execute(query, params)
            batch_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"✅ Batch {batch_data['batch_name']} inserted with ID {batch_id}")
            return batch_id
        except Exception as e:
            print(f"❌ Error inserting batch: {e}")
            return None
    
    def insert_snapshot(self, batch_id: int, snapshot_data: Dict) -> int:
        """Insert trade snapshot (intraday or EOD)"""
        
        query = """
        EXEC sp_InsertTradeSnapshot
            @BatchID = ?,
            @SnapshotTime = ?,
            @IsEOD = ?,
            @SnapshotType = ?,
            @SPYPrice = ?,
            @IVRank = ?,
            @VIX = ?,
            @PutDelta = ?,
            @PutGamma = ?,
            @PutVega = ?,
            @PutTheta = ?,
            @CallDelta = ?,
            @CallGamma = ?,
            @CallVega = ?,
            @CallTheta = ?,
            @CurrentPnL = ?,
            @ProbabilityOfProfit = ?,
            @ExpectedValue = ?,
            @RecommendedAction = ?,
            @RecommendationReason = ?,
            @HardStopTriggered = ?,
            @HardStopReason = ?
        """
        
        params = [
            batch_id,
            snapshot_data['snapshot_time'],
            snapshot_data['is_eod'],
            snapshot_data['snapshot_type'],
            snapshot_data['spy_price'],
            snapshot_data['iv_rank'],
            snapshot_data['vix'],
            snapshot_data['put_delta'],
            snapshot_data['put_gamma'],
            snapshot_data['put_vega'],
            snapshot_data['put_theta'],
            snapshot_data['call_delta'],
            snapshot_data['call_gamma'],
            snapshot_data['call_vega'],
            snapshot_data['call_theta'],
            snapshot_data['current_pnl'],
            snapshot_data['probability_of_profit'],
            snapshot_data['expected_value'],
            snapshot_data['recommended_action'],
            snapshot_data['recommendation_reason'],
            snapshot_data['hard_stop_triggered'],
            snapshot_data['hard_stop_reason'],
        ]
        
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            cursor.execute(query, params)
            snapshot_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            snapshot_type = snapshot_data['snapshot_type']
            print(f"✅ {snapshot_type} snapshot inserted with ID {snapshot_id}")
            return snapshot_id
        except Exception as e:
            print(f"❌ Error inserting snapshot: {e}")
            return None
    
    def get_latest_batch_status(self, batch_id: int) -> pd.DataFrame:
        """Get most recent status of a batch"""
        
        query = "EXEC sp_GetLatestBatchStatus @BatchID = ?"
        
        try:
            result = self.db.fetch_query(query, [batch_id])
            if result is not None and len(result) > 0:
                print(f"✅ Retrieved latest status for Batch {batch_id}")
                return result
            else:
                print(f"⚠️  No data found for Batch {batch_id}")
                return None
        except Exception as e:
            print(f"❌ Error retrieving batch status: {e}")
            return None
    
    def log_hard_stop(self, batch_id: int, stop_type: str, stop_value: float,
                      spy_price: float, notes: str = "") -> int:
        """Log hard stop event"""
        
        query = """
        EXEC sp_LogHardStop
            @BatchID = ?,
            @StopType = ?,
            @StopValue = ?,
            @SPYPriceAtStop = ?,
            @Notes = ?
        """
        
        params = [batch_id, stop_type, stop_value, spy_price, notes]
        
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            cursor.execute(query, params)
            hard_stop_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"🛑 Hard stop {stop_type} logged for Batch {batch_id}")
            return hard_stop_id
        except Exception as e:
            print(f"❌ Error logging hard stop: {e}")
            return None
    
    def log_trade_action(self, batch_id: int, action_data: Dict) -> int:
        """Log trade action"""
        
        query = """
        EXEC sp_LogTradeAction
            @BatchID = ?,
            @ActionType = ?,
            @ActionDescription = ?,
            @ExecutedPrice = ?,
            @ExecutedQuantity = ?,
            @SPYPriceAtAction = ?,
            @PnLAtAction = ?,
            @WasSuccessful = ?,
            @Notes = ?
        """
        
        params = [
            batch_id,
            action_data['action_type'],
            action_data['action_description'],
            action_data['executed_price'],
            action_data['executed_quantity'],
            action_data['spy_price_at_action'],
            action_data['pnl_at_action'],
            action_data['was_successful'],
            action_data['notes'],
        ]
        
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            cursor.execute(query, params)
            action_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"✅ Trade action logged for Batch {batch_id}")
            return action_id
        except Exception as e:
            print(f"❌ Error logging trade action: {e}")
            return None
    
    def close_batch(self, batch_id: int, closed_data: Dict) -> bool:
        """Close batch"""
        
        query = """
        EXEC sp_CloseBatch
            @BatchID = ?,
            @ClosedDate = ?,
            @ClosedPrice = ?,
            @ClosedReason = ?,
            @FinalPnL = ?
        """
        
        params = [
            batch_id,
            closed_data['closed_date'],
            closed_data['closed_price'],
            closed_data['closed_reason'],
            closed_data['final_pnl'],
        ]
        
        return self.db.execute_query(query, params)
    
    def record_roll(self, roll_data: Dict) -> int:
        """Record roll transaction"""
        
        query = """
        EXEC sp_RecordRoll
            @OriginalBatchID = ?,
            @RollDate = ?,
            @ClosedSide = ?,
            @ClosedAtPrice = ?,
            @ClosedAtLoss = ?,
            @NewCreditCollected = ?,
            @RollingCost = ?,
            @RollReason = ?
        """
        
        params = [
            roll_data['original_batch_id'],
            roll_data['roll_date'],
            roll_data['closed_side'],
            roll_data['closed_at_price'],
            roll_data['closed_at_loss'],
            roll_data['new_credit_collected'],
            roll_data['rolling_cost'],
            roll_data['roll_reason'],
        ]
        
        try:
            conn = self.db.connect()
            cursor = conn.cursor()
            cursor.execute(query, params)
            roll_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"✅ Roll recorded with ID {roll_id}")
            return roll_id
        except Exception as e:
            print(f"❌ Error recording roll: {e}")
            return None
    
    def get_daily_portfolio_summary(self, snapshot_date: str) -> pd.DataFrame:
        """Get daily portfolio summary"""
        
        query = "EXEC sp_GetDailyPortfolioSummary @SnapshotDate = ?"
        
        try:
            result = self.db.fetch_query(query, [snapshot_date])
            if result is not None and len(result) > 0:
                print(f"✅ Retrieved portfolio summary for {snapshot_date}")
                return result
            else:
                print(f"⚠️  No summary data for {snapshot_date}")
                return None
        except Exception as e:
            print(f"❌ Error retrieving portfolio summary: {e}")
            return None
    
    def generate_trade_journal_report(self, batch_id: int) -> pd.DataFrame:
        """Generate complete trade journal for a batch"""
        
        query = """
        SELECT
            b.BatchName,
            b.EntryDate,
            b.ExpirationDate,
            b.EntryPrice,
            b.CreditCollected,
            ts.SnapshotTime,
            ts.SPYPrice,
            ts.CurrentPnL,
            ts.ProbabilityOfProfit,
            ts.ExpectedValue,
            ts.RecommendedAction,
            ts.OverallStatus,
            ta.ActionType,
            ta.ActionDate
        FROM [dbo].[Trade_Batches] b
        LEFT JOIN [dbo].[Trade_Snapshots] ts ON b.BatchID = ts.BatchID
        LEFT JOIN [dbo].[Trade_Actions] ta ON b.BatchID = ta.BatchID
        WHERE b.BatchID = ?
        ORDER BY ts.SnapshotTime DESC, ta.ActionDate DESC
        """
        
        try:
            result = self.db.fetch_query(query, [batch_id])
            if result is not None:
                print(f"✅ Generated trade journal for Batch {batch_id}")
                return result
            else:
                print(f"⚠️  No journal data for Batch {batch_id}")
                return None
        except Exception as e:
            print(f"❌ Error generating journal: {e}")
            return None
    
    def export_to_csv(self, batch_id: int, filename: str = None) -> str:
        """Export trade journal to CSV"""
        
        if filename is None:
            filename = f"batch_{batch_id}_journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = self.generate_trade_journal_report(batch_id)
        
        if df is not None:
            df.to_csv(filename, index=False)
            print(f"✅ Trade journal exported to {filename}")
            return filename
        else:
            print(f"❌ Failed to export journal")
            return None

# ============================================================================
# AUTOMATED INTRADAY CAPTURE
# ============================================================================

class IntraDayCapture:
    """Automatic intraday snapshot capture every 2 hours"""
    
    def __init__(self, tradeblog_manager: TradeBlogManager,
                 decision_engine, metrics_calculator):
        self.manager = tradeblog_manager
        self.decision_engine = decision_engine
        self.metrics_calculator = metrics_calculator
    
    def capture_snapshot(self, batch_id: int, position_data: Dict,
                        is_eod: bool = False) -> int:
        """Capture snapshot at this moment"""
        
        # Calculate all metrics
        snapshot_time = datetime.now()
        spy_price = position_data['spy_price']
        
        # Build snapshot data
        snapshot_data = {
            'snapshot_time': snapshot_time,
            'is_eod': 1 if is_eod else 0,
            'snapshot_type': 'EOD' if is_eod else 'INTRADAY_2H',
            'spy_price': spy_price,
            'iv_rank': position_data['iv_rank'],
            'vix': position_data.get('vix', 0),
            
            'put_delta': position_data['put_delta'],
            'put_gamma': position_data.get('put_gamma', 0),
            'put_vega': position_data.get('put_vega', 0),
            'put_theta': position_data.get('put_theta', 0),
            
            'call_delta': position_data['call_delta'],
            'call_gamma': position_data.get('call_gamma', 0),
            'call_vega': position_data.get('call_vega', 0),
            'call_theta': position_data.get('call_theta', 0),
            
            'current_pnl': position_data['current_pnl'],
            'probability_of_profit': position_data.get('pop', 56.3),
            'expected_value': position_data.get('ev', 0),
            
            'recommended_action': position_data['recommendation'],
            'recommendation_reason': position_data.get('reason', ''),
            
            'hard_stop_triggered': 1 if position_data.get('hard_stop') else 0,
            'hard_stop_reason': position_data.get('hard_stop_reason', ''),
        }
        
        # Insert to database
        snapshot_id = self.manager.insert_snapshot(batch_id, snapshot_data)
        
        return snapshot_id
    
    def daily_automated_capture(self, batch_id: int, positions_data: List[Dict]) -> None:
        """Run daily automated capture"""
        
        print("\n" + "="*70)
        print("DAILY AUTOMATED TRADEBLOG CAPTURE")
        print("="*70)
        
        # 2-hour intraday captures
        capture_times = [
            "10:00 AM",  # 2 hours after open
            "12:00 PM",  # 4 hours
            "2:00 PM",   # 6 hours
        ]
        
        print("\n📊 Intraday Captures (Every 2 Hours):")
        for pos in positions_data:
            snapshot_id = self.capture_snapshot(batch_id, pos, is_eod=False)
            if snapshot_id:
                print(f"  ✅ {pos['batch_name']}: Snapshot {snapshot_id}")
        
        # EOD capture
        print("\n📈 EOD Capture (4:00 PM):")
        for pos in positions_data:
            snapshot_id = self.capture_snapshot(batch_id, pos, is_eod=True)
            if snapshot_id:
                print(f"  ✅ {pos['batch_name']}: EOD Snapshot {snapshot_id}")
        
        # Generate daily portfolio summary
        portfolio_summary = self.manager.get_daily_portfolio_summary(
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if portfolio_summary is not None:
            print("\n📊 Portfolio Summary:")
            print(portfolio_summary.to_string())

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Entry point"""
    
    print("\n" + "="*70)
    print("TRADEBLOG DATABASE SYSTEM")
    print("="*70)
    
    # Initialize
    tradeblog = TradeBlogManager()
    
    # Example: Insert new batch
    batch_data = {
        'batch_name': 'BATCH_1',
        'entry_date': datetime(2026, 1, 22).date(),
        'expiration_date': datetime(2026, 3, 7).date(),
        'put_long_strike': 665.0,
        'put_short_strike': 670.0,
        'call_short_strike': 710.0,
        'call_long_strike': 715.0,
        'entry_price': 689.50,
        'credit_collected': 140.0,
        'spread_width': 500.0,
        'number_of_spreads': 20,
        'entry_iv_rank': 58.0,
        'entry_vix': 16.5,
    }
    
    batch_id = tradeblog.insert_batch(batch_data)
    
    if batch_id:
        # Insert snapshots
        snapshot_data = {
            'snapshot_time': datetime.now(),
            'is_eod': 0,
            'snapshot_type': 'INTRADAY_2H',
            'spy_price': 691.40,
            'iv_rank': 58.0,
            'vix': 16.5,
            'put_delta': -0.25,
            'put_gamma': 0.0025,
            'put_vega': 12.50,
            'put_theta': 48.0,
            'call_delta': -0.25,
            'call_gamma': 0.0025,
            'call_vega': 12.50,
            'call_theta': 48.0,
            'current_pnl': 35.0,
            'probability_of_profit': 56.3,
            'expected_value': 65.20,
            'recommended_action': 'HOLD',
            'recommendation_reason': 'Let theta work',
            'hard_stop_triggered': 0,
            'hard_stop_reason': '',
        }
        
        snapshot_id = tradeblog.insert_snapshot(batch_id, snapshot_data)
        
        # Get latest status
        status = tradeblog.get_latest_batch_status(batch_id)
        if status is not None:
            print("\nLatest Batch Status:")
            print(status)
        
        # Export to CSV
        tradeblog.export_to_csv(batch_id)

if __name__ == '__main__':
    main()